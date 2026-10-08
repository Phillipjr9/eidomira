from __future__ import annotations
import logging
import time
import cv2
import numpy as np
from .base import Enrollment, FaceSwapEngine, FrameResult
from app.providers import describe_providers, execution_providers
from app.config import settings
from app.compositor import SemanticCompositor
from app.enhance import FaceRestorer, transfer_tone

log = logging.getLogger("eidomira.inswapper")


class _AlignedFace:
    """A face whose keypoints are the canonical arcface template.

    `INSwapper.get` derives its own alignment from `target.kps` before it runs the model.
    Handing it the template makes that alignment the identity, so the crop it processes is
    exactly the crop it was handed — which is what the phased path in `app/boost.py` needs,
    because it has already done the alignment itself and is varying it deliberately.
    """

    __slots__ = ("kps", "bbox")

    def __init__(self, kps, bbox):
        self.kps = kps
        self.bbox = bbox


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
        self._boost_disabled = False

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

    def _swap_face(self, bgr, target, identity):
        """The swapped frame, boosted when asked for and possible.

        A boost failure is reported once and then stops being attempted: a deployment whose
        installed library does not match the assumptions `_boosted_swap` documents should
        lose the boost, not every frame to an exception.
        """
        if int(settings.swap_pixel_boost) > 1 and not self._boost_disabled:
            try:
                boosted = self._boosted_swap(bgr, target, identity)
                if boosted is not None:
                    return boosted
            except Exception as exc:
                self._boost_disabled = True
                log.warning(
                    "pixel boost failed (%s: %s); using one pass for the rest of this "
                    "session. Set STUDIO_SWAP_PIXEL_BOOST=1 to stop trying entirely.",
                    type(exc).__name__, exc,
                )
        return self.swapper.get(bgr, target, identity, paste_back=True)

    def _quality(self, overrides):
        """This session's quality values, falling back to the configured defaults."""
        overrides = overrides or {}
        return (
            float(overrides.get("tone_transfer_strength", settings.tone_transfer_strength)),
            float(overrides.get("parser_feather", settings.parser_feather)),
            float(overrides.get("restoration_visibility", settings.restoration_visibility)),
        )

    def _boosted_swap(self, bgr, target, identity):
        """The swapped face at `settings.swap_pixel_boost` times 128, pasted into the frame.

        Returns None when the face is small enough that the crop had to *upsample* it: there
        is no detail in a 128 crop to recover, and `scale`² passes would buy nothing for
        `scale`² times the cost. The transform's linear term is the crop's scale, so it says
        which case we are in without a second detection pass.

        **This path has never run against a real model.** No swap weights exist on this
        machine and `insightface` is not installed, so what is exercised by the tests is the
        plumbing around a stand-in: the alignment it builds, the number of passes it makes,
        the canvas size and the paste-back. Three things about the real library are assumed
        and cannot be checked here:

        1. `face_align.estimate_norm(kps, 128)` returns the same 2x3 affine `norm_crop2`
           uses internally, so a crop warped with it is the crop the model expects.
        2. The template keypoints make `INSwapper.get`'s own alignment the identity.
        3. `INSwapper.get(..., paste_back=False)` returns the 128x128 aligned result.

        If any of those is wrong the boost produces a scrambled or misaligned face rather
        than a sharp one — which is why it is off by default, why it falls back to the
        single pass on the first exception, and why the way to turn it on is to point
        `tools/quality_report.py` at a real pair and look at the number.
        """
        from insightface.utils import face_align

        from app.boost import ALIGN, boosted_face, paste_back, shifted_transform

        scale = int(settings.swap_pixel_boost)
        base = np.asarray(face_align.estimate_norm(target.kps, ALIGN, mode="arcface"),
                          np.float64)
        # The linear term is crop pixels per frame pixel. Above 1 the crop enlarges a face
        # that was already small in the frame, so the detail the boost recovers was never
        # thrown away and the passes would buy nothing for four times the cost. Below 1 the
        # crop shrank a larger face into 128 px, and the phases are what gets it back.
        if min(abs(base[0, 0]), abs(base[1, 1])) >= 1.0:
            return None
        # The template for this crop size, taken from the library rather than hardcoded:
        # insightface defines it at 112 and scales it, and a copy here would drift.
        template = np.asarray(face_align.arcface_dst, np.float32) * (ALIGN / 112.0)
        aligned = _AlignedFace(template, target.bbox)

        def warp_crop(dx, dy):
            return cv2.warpAffine(
                bgr, shifted_transform(base, dx, dy), (ALIGN, ALIGN),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )

        def swap_once(crop):
            return self.swapper.get(crop, aligned, identity, paste_back=False)

        canvas = boosted_face(warp_crop, swap_once, scale)
        layer, alpha = paste_back(bgr, canvas, base, scale)
        # `paste_back` hands back a layer that is zero outside the canvas, because a layer
        # with a smeared border is the kind of thing that gets composited by accident. The
        # frame the rest of the pipeline expects is the room with the boosted face on it,
        # so the layer goes back over the room here, where the alpha says it is valid.
        return np.where(alpha[..., None] > 0, layer, bgr).astype(np.uint8)

    def process(self, rgb, identity, verified, overrides=None):
        tone, feather, visibility = self._quality(overrides)
        start = time.perf_counter()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = self.analyzer.get(bgr)
        if not faces:
            return FrameResult(rgb, False, verified, (time.perf_counter()-start)*1000)
        target = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        result = self._swap_face(bgr, target, identity)
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
