from __future__ import annotations
import time
import cv2
import numpy as np
from .base import Enrollment, FaceSwapEngine, FrameResult
from .providers import describe_providers, execution_providers
from app.config import settings
from app.compositor import SemanticCompositor


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

    def process(self, rgb, identity, verified):
        start = time.perf_counter()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.analyzer.get(bgr)
        if not faces:
            return FrameResult(rgb, False, verified, (time.perf_counter()-start)*1000)
        target = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        result = self.swapper.get(bgr, target, identity, paste_back=True)
        out = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        if self.compositor is not None:
            out, _ = self.compositor.blend(rgb, out, target.bbox)
        cv2.putText(out, "SYNTHETIC", (12, out.shape[0]-14), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (235, 225, 255), 1, cv2.LINE_AA)
        return FrameResult(out, True, verified, (time.perf_counter()-start)*1000)
