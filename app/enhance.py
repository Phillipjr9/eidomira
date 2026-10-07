"""Post-swap refinement: the stage that decides whether a swap looks real.

The swap model is not where the quality gap is. `inswapper_128` outputs a 128x128 face,
so a raw swap is soft no matter which detector or mask sits around it. Every
production-grade pipeline in 2026 stacks the same things on top, and this module is two of
them:

* **Face restoration** rebuilds the detail the 128px swap loses, blended back at partial
  strength because full strength looks airbrushed.
* **Tone transfer** matches the swapped face's colour and contrast to the target's own
  lighting, which is the other thing a viewer notices instantly — a face pasted from a
  differently lit room.

Both are deliberately model-agnostic. Restoration is inactive until a licensed model file
is present, and the choice of model is a licensing decision rather than a technical one:

===================  ==========================  ==============================
Model                Licence                     Usable in a paid product
===================  ==========================  ==============================
GFPGAN v1.4          Apache-2.0                  yes
GPEN-BFR-512         Apache-2.0 (code)           yes, verify the weights
CodeFormer           NTU S-Lab License 1.0       **no** — non-commercial
===================  ==========================  ==============================

`inswapper_128` itself is non-commercial and needs a licence from InsightFace, who sell
one. See `docs/quality-and-licensing.md`.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

#: Reinhard-style transfer clipped to a sane range. An unbounded standard-deviation ratio
#: turns a noisy source into noise amplified, which reads as grain over the whole face.
DEFAULT_GAIN_LIMIT = (0.8, 1.25)


def transfer_tone(reference_rgb: np.ndarray, source_rgb: np.ndarray, mask: np.ndarray,
                  strength: float = 1.0,
                  gain_limit: tuple[float, float] = DEFAULT_GAIN_LIMIT) -> np.ndarray:
    """Match `source_rgb`'s colour statistics to `reference_rgb` inside `mask`.

    `reference_rgb` is the target frame before it was swapped, so the swapped face ends up
    sitting in the target's own lighting rather than the source photograph's. Statistics
    are taken over the solid interior of the mask, and the correction is faded out across
    the feather so no new edge is introduced where the old one was removed.

    Returns a new frame; the input is not modified. Pixels outside `mask` are untouched.
    """
    if reference_rgb.shape != source_rgb.shape:
        raise ValueError(f"frames differ: {reference_rgb.shape} vs {source_rgb.shape}")
    if mask.shape[:2] != source_rgb.shape[:2]:
        raise ValueError(f"mask {mask.shape[:2]} does not match frame {source_rgb.shape[:2]}")

    strength = float(np.clip(strength, 0.0, 1.0))
    solid = mask > 0.5
    if strength <= 0.0 or int(solid.sum()) < 32:
        return source_rgb.copy()

    # Measure inside the feather, not across it: the transition ring is part target and
    # part source, so including it pulls both sets of statistics toward each other and
    # quietly weakens the correction it is supposed to make.
    interior = cv2.erode(solid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if int(interior.sum()) < 32:
        interior = solid

    reference_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    corrected = source_lab.copy()
    for channel in range(3):
        reference_values = reference_lab[..., channel][interior]
        source_values = source_lab[..., channel][interior]
        reference_mean, reference_std = float(reference_values.mean()), float(reference_values.std())
        source_mean, source_std = float(source_values.mean()), float(source_values.std())
        gain = float(np.clip(reference_std / max(source_std, 1e-3), *gain_limit))
        corrected[..., channel] = (source_lab[..., channel] - source_mean) * gain + reference_mean

    corrected_rgb = cv2.cvtColor(
        np.clip(corrected, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    alpha = (mask.astype(np.float32) * strength)[..., None]
    blended = source_rgb.astype(np.float32) * (1.0 - alpha) + corrected_rgb.astype(np.float32) * alpha
    return np.rint(np.clip(blended, 0, 255)).astype(np.uint8)


class FaceRestorer:
    """Rebuilds detail the 128px swap lost, in the region it was lost in.

    Inactive until a model file exists, so a deployment without one keeps working: the
    swap is simply softer, which is the current behaviour rather than an error.
    """

    def __init__(self, model_path: str, visibility: float = .75,
                 providers: list[str] | None = None, size: int | None = None):
        self.visibility = float(np.clip(visibility, 0.0, 1.0))
        self.size = size
        self.providers = providers
        #: First fault seen, if any. Kept so a failure that happens once per frame is
        #: still visible to whoever is debugging rather than being swallowed silently.
        self.fault: str | None = None
        if self.size is None:
            self.size = self._read_size(model_path)
        self.session = self._load(model_path)

    def _load(self, model_path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the deployment
            raise RuntimeError("onnxruntime is required for face restoration") from exc
        if self.providers is None:
            from app.providers import execution_providers
            self.providers = list(execution_providers())
        try:
            return ort.InferenceSession(model_path, providers=self.providers)
        except Exception as exc:
            raise RuntimeError(f"could not load the restoration model {model_path!r}") from exc

    @staticmethod
    def _read_size(model_path: str) -> int:
        """The model's square input size, when the graph declares a fixed one."""
        import onnxruntime as ort

        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        last = shape[-1]
        return int(last) if isinstance(last, int) else 512

    def _fault(self, reason: str, exc: Exception | None = None) -> np.ndarray:
        """Record a fault once and keep the stream alive.

        Restoration is an improvement, not a requirement: a frame that cannot be restored
        should still go out. But it must not go out *silently*, because that is how a
        broken model looks exactly like a working one.
        """
        message = reason if exc is None else f"{reason}: {type(exc).__name__}: {exc}"
        if self.fault is not None:
            return None
        self.fault = message
        logging.getLogger("eidomira.enhance").warning(
            "face restoration disabled for this session — %s", message)
        return None

    def enhance(self, frame_rgb: np.ndarray, bbox) -> np.ndarray:
        """Restore the face inside `bbox` and blend it back at `visibility` strength."""
        if self.visibility <= 0.0:
            return frame_rgb
        height, width = frame_rgb.shape[:2]
        x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return frame_rgb

        crop = frame_rgb[y1:y2, x1:x2]
        resized = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]

        inputs = self.session.get_inputs()
        name = inputs[0].name
        try:
            raw = self.session.run(None, {name: tensor})[0]
        except Exception as exc:
            # a session that wants a different layout than the usual NCHW is not worth a
            # crash mid-stream; leave the frame as the swap produced it, but say so once
            self._fault("the model rejected an input frame", exc)
            return frame_rgb

        restored = np.squeeze(np.asarray(raw))
        if restored.ndim == 3 and restored.shape[0] == 3:
            restored = np.transpose(restored, (1, 2, 0))
        if restored.ndim != 3 or restored.shape[2] != 3:
            self._fault(f"the model returned an unexpected shape {restored.shape}")
            return frame_rgb
        if restored.dtype != np.uint8:
            restored = np.clip(restored * (255.0 if restored.max() <= 1.0 else 1.0),
                               0, 255).astype(np.uint8)
        restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        restored = cv2.resize(restored, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)

        # Feather the paste so the restored crop does not arrive with its own edges.
        alpha = np.ones((y2 - y1, x2 - x1), np.float32) * self.visibility
        blur = max(3, (min(y2 - y1, x2 - x1) // 8) | 1)
        alpha = cv2.GaussianBlur(alpha, (blur, blur), 0)[..., None]

        out = frame_rgb.copy()
        region = out[y1:y2, x1:x2].astype(np.float32)
        out[y1:y2, x1:x2] = np.rint(
            region * (1.0 - alpha) + restored.astype(np.float32) * alpha
        ).astype(np.uint8)
        return out
