from __future__ import annotations
import time
import cv2
import numpy as np
from .base import Enrollment, FaceSwapEngine, FrameResult
from app.providers import describe_providers, execution_providers
from app.config import settings
from app.compositor import SemanticCompositor
from app.enhance import FaceRestorer, transfer_tone


class InSwapperEngine(FaceSwapEngine):
    """Optional licensed InsightFace model adapter for an NVIDIA deployment."""
    name = "inswapper"

    def __init__(self, model_path: str, threshold: float):
        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise RuntimeError("Install requirements-gpu.txt on the GPU worker") from exc
        self.threshold = threshold
        # Use whatever accelerator this host really exposes instead of assuming CUDA.
        self.providers = list(execution_providers())
        self.provider = describe_providers(self.providers)
        self.accelerated = self.provider != "cpu"
        self.analyzer = FaceAnalysis(name="buffalo_l", providers=self.providers)
        # ctx_id selects the CUDA device; it must be < 0 when CUDA is not in play,
        # otherwise InsightFace tries to bind a GPU context that does not exist.
        self.analyzer.prepare(ctx_id=0 if "CUDAExecutionProvider" in self.providers else -1,
                              det_size=(640, 640))
        self.swapper = insightface.model_zoo.get_model(model_path, providers=self.providers)
        self.compositor = None
        if settings.parser_model_path.exists():
            self.compositor = SemanticCompositor(
                str(settings.parser_model_path), settings.parser_feather,
                settings.parser_include_ears,
            )
        # Optional: without a model the swap is simply softer, which is the honest
        # behaviour rather than a startup failure.
        self.restorer = None
        if settings.restoration_model_path.exists():
            self.restorer = FaceRestorer(
                str(settings.restoration_model_path), settings.restoration_visibility,
            )

    def _faces(self, rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return sorted(self.analyzer.get(bgr), key=lambda f: f.bbox[0])

    def enroll(self, rgb):
        faces = self._faces(rgb)
        if len(faces) != 1:
            raise ValueError("Enrollment requires exactly one clearly visible face.")
        return Enrollment(identity=faces[0], preview=rgb)

    def verify_self(self, rgb, identity):
        faces = self._faces(rgb)
        if len(faces) != 1:
            return False, 0.0
        a = identity.normed_embedding
        b = faces[0].normed_embedding
        score = float(np.dot(a, b))
        return score >= self.threshold, score

    def observe_liveness(self, rgb):
        faces = self._faces(rgb)
        if len(faces) != 1:
            return False, None
        points = np.asarray(faces[0].kps, dtype=np.float32)
        eye_mid = (points[0] + points[1]) * .5
        eye_distance = max(1.0, float(np.linalg.norm(points[1] - points[0])))
        # A robust low-cost yaw proxy for challenge response, normalized by eye span.
        yaw = float((points[2, 0] - eye_mid[0]) / eye_distance)
        return True, yaw

    @staticmethod
    def _bbox_alpha(rgb, bbox, feather=.035):
        """A feathered mask over the face box, for when there is no parser model."""
        height, width = rgb.shape[:2]
        x1, y1, x2, y2 = SemanticCompositor._expanded_bbox(bbox, width, height)
        alpha = np.zeros((height, width), np.float32)
        alpha[y1:y2, x1:x2] = 1.0
        blur = max(3, int(min(x2 - x1, y2 - y1) * feather) | 1)
        return cv2.GaussianBlur(alpha, (blur, blur), 0)

    def active_stages(self) -> set[str]:
        stages = set()
        if self.restorer is not None:
            stages.add("restoration")
        if self.compositor is not None:
            stages.add("parser")
        return stages

    def stage_faults(self) -> dict[str, str]:
        faults = {}
        if self.restorer is not None and self.restorer.fault:
            faults["restoration"] = self.restorer.fault
        if self.compositor is not None and self.compositor.fault:
            faults["parser"] = self.compositor.fault
        return faults

    def _quality(self, overrides):
        """This session's quality values, falling back to the configured defaults."""
        overrides = overrides or {}
        return (
            float(overrides.get("tone_transfer_strength", settings.tone_transfer_strength)),
            float(overrides.get("parser_feather", settings.parser_feather)),
            float(overrides.get("restoration_visibility", settings.restoration_visibility)),
        )

    def process(self, rgb, identity, verified, overrides=None):
        tone, feather, visibility = self._quality(overrides)
        start = time.perf_counter()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.analyzer.get(bgr)
        if not faces:
            return FrameResult(rgb, False, verified, (time.perf_counter()-start)*1000)
        target = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        result = self.swapper.get(bgr, target, identity, paste_back=True)
        out = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # Restoration rebuilds what the 128px swap lost, before the mask decides how much
        # of it reaches the frame. Skipped entirely when this session has it at zero.
        if self.restorer is not None and visibility > 0:
            out = self.restorer.enhance(out, target.bbox, visibility)
        # A parser that has been reporting no face pixels is worse than no parser: it
        # composites nothing, so the swap silently does nothing. Fall back to the box.
        if self.compositor is not None and self.compositor.fault is None:
            out, alpha = self.compositor.blend(rgb, out, target.bbox, feather)
        else:
            alpha = self._bbox_alpha(rgb, target.bbox, feather)
        # Then put the result into the target's own lighting, which is the other thing a
        # viewer reads as "pasted in" even when the geometry is perfect.
        if tone > 0:
            out = transfer_tone(rgb, out, alpha, tone)
        cv2.putText(out, "SYNTHETIC", (12, out.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (235, 225, 255), 1, cv2.LINE_AA)
        return FrameResult(out, True, verified, (time.perf_counter()-start)*1000)
