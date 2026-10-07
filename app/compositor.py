from __future__ import annotations

import cv2
import numpy as np


# Common 19-class CelebAMask-HQ/BiSeNet layout.
SKIN = {1, 2, 3, 4, 5, 10, 11, 12, 13}
EARS = {7, 8}
PROTECTED_OCCLUDERS = {6, 9, 14, 15, 16, 17, 18}  # glasses, jewelry, neck, clothes, hair, hat


class SemanticCompositor:
    """Face-local semantic blending with occluder preservation.

    Expected model input is NCHW RGB normalized to [-1, 1]. Output may be
    NCHW logits or a single label map. The class layout follows the commonly
    distributed 19-class face parsing models.
    """

    def __init__(self, model_path: str, feather: float = .035, include_ears: bool = True):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for semantic compositing") from exc
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[-1]) if isinstance(shape[-1], int) else 512
        self.feather = float(np.clip(feather, .005, .15))
        self.classes = SKIN | (EARS if include_ears else set())

    @staticmethod
    def _expanded_bbox(bbox, width, height, expansion=.22):
        x1, y1, x2, y2 = map(float, bbox)
        w, h = x2 - x1, y2 - y1
        x1 -= w * expansion; x2 += w * expansion
        y1 -= h * expansion; y2 += h * expansion
        return (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))

    def mask(self, original_rgb: np.ndarray, bbox) -> np.ndarray:
        h, w = original_rgb.shape[:2]
        x1, y1, x2, y2 = self._expanded_bbox(bbox, w, h)
        crop = original_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((h, w), np.float32)
        resized = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 127.5 - 1.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        raw = self.session.run(None, {self.input_name: tensor})[0]
        if raw.ndim == 4 and raw.shape[1] > 1:
            labels = np.argmax(raw[0], axis=0).astype(np.uint8)
        else:
            labels = np.squeeze(raw).astype(np.uint8)
        local = np.isin(labels, list(self.classes)).astype(np.uint8) * 255
        # Close tiny neural holes, but leave parsed glasses/hair excluded.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel)
        local = cv2.resize(local, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        blur = max(3, int(min(x2-x1, y2-y1) * self.feather) | 1)
        local = cv2.GaussianBlur(local, (blur, blur), 0)
        full = np.zeros((h, w), np.float32)
        full[y1:y2, x1:x2] = local.astype(np.float32) / 255.0
        return full

    def blend(self, original_rgb: np.ndarray, swapped_rgb: np.ndarray, bbox):
        alpha = self.mask(original_rgb, bbox)[..., None]
        result = original_rgb.astype(np.float32) * (1.0 - alpha)
        result += swapped_rgb.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8), alpha[..., 0]
