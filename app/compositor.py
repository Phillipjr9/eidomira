from __future__ import annotations

import cv2
import numpy as np

from app.providers import execution_providers


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
        # Same resolution as the swap adapter: hardcoding CUDA here meant a DirectML,
        # CoreML or ROCm host ran the parser on the CPU without saying so.
        self.providers = list(execution_providers())
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[-1]) if isinstance(shape[-1], int) else 512
        self.feather = float(np.clip(feather, .005, .15))
        self.classes = SKIN | (EARS if include_ears else set())
        # Everything the parser labels that we have deliberately decided not to cover. The
        # morphological close fills small holes, so any exclusion missing from this set is
        # silently undone — which is why `parser_include_ears: false` had no effect.
        self.protected = PROTECTED_OCCLUDERS | (set() if include_ears else EARS)

    @staticmethod
    def _masks(raw: np.ndarray, classes, protected) -> tuple[np.ndarray, np.ndarray]:
        """Turn a parser's raw output into (blendable, protected) boolean masks.

        Models in the wild emit multi-class logits, single-channel logits, probabilities, or
        an already-argmaxed label map. Only the last of those survives a plain uint8 cast:
        negative logits wrap into garbage class ids, and probabilities in [0, 1] collapse to
        zero. Neither raises. The mask just comes back empty, `blend` returns the original
        frame, and the swap silently does nothing while still reporting success.
        """
        array = np.asarray(raw)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]                                    # batch of one, however padded
        if array.ndim == 3 and array.shape[0] > 1:
            # argmax is right for logits and for probabilities alike
            labels = np.argmax(array, axis=0)
            return np.isin(labels, list(classes)), np.isin(labels, list(protected))
        if array.ndim == 3:
            array = array[0]                                    # single channel
        if np.issubdtype(array.dtype, np.integer):
            return np.isin(array, list(classes)), np.isin(array, list(protected))
        empty = np.zeros_like(array, dtype=bool)
        if array.size and float(array.min()) >= 0.0 and float(array.max()) <= 1.0:
            return array >= 0.5, empty                          # probabilities
        return array > 0.0, empty                               # logits: positive means yes

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
        blendable, protected = self._masks(raw, self.classes, self.protected)
        local = blendable.astype(np.uint8) * 255
        # Close tiny neural holes, but leave parsed glasses/hair excluded.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel)
        # The close is what removes speckle, and it also fills straight over the classes the
        # parser deliberately excluded — painting the swapped face across glasses and hair.
        # PROTECTED_OCCLUDERS was declared for this and never used until now.
        if protected.shape != local.shape:
            protected = cv2.resize(protected.astype(np.uint8), (local.shape[1], local.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        local[protected] = 0
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
        # Round rather than truncate: truncation darkens the feathered seam by up to a level.
        return np.rint(np.clip(result, 0, 255)).astype(np.uint8), alpha[..., 0]
