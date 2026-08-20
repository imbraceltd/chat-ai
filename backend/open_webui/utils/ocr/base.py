"""Real-OCR engine abstraction for Document AI confidence grounding.

This is intentionally separate from ``utils/vision`` (which is LLM-based "OCR" and
only carries a heuristic page-level confidence). Engines here wrap a *real* OCR
backend that emits per-line/word text together with a bounding box and a
calibrated confidence score, which we use to deterministically ground the values
extracted by the vision LLM.

Coordinate convention: every ``OcrToken.box`` is ``[x1, y1, x2, y2]`` normalized
to ``0..1`` of the page image, so boxes are resolution-independent regardless of
the DPI at which the page was rendered.
"""

import base64
import binascii
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class OcrToken:
    """A single OCR-detected text span on a page."""

    text: str
    box: List[float]  # [x1, y1, x2, y2] normalized to 0..1 of the page image
    confidence: float  # 0..1
    page: int


def decode_image(image_base64: str) -> Tuple["object", int, int]:
    """Decode a base64 PNG into an RGB numpy array plus (width, height).

    Imported lazily so that ``utils.ocr`` stays importable even where numpy/PIL
    are unavailable until an engine actually runs. Raises on malformed input.
    """
    import numpy as np
    from PIL import Image

    try:
        raw = base64.b64decode(image_base64)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Invalid base64 image data: {e}") from e

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    width, height = image.size
    return np.asarray(image), width, height


def image_dimensions(image_base64: str) -> Tuple[int, int]:
    """Return ``(width, height)`` of a base64 image without decoding pixels to numpy.

    Used by the external-OCR-API path to normalize the server's pixel boxes.
    """
    from PIL import Image

    raw = base64.b64decode(image_base64)
    with Image.open(io.BytesIO(raw)) as image:
        return image.size


def normalize_box(
    points_or_xyxy, width: int, height: int
) -> List[float]:
    """Normalize a box to ``[x1, y1, x2, y2]`` in 0..1.

    Accepts either a 4-point polygon ``[[x,y], ...]`` (RapidOCR/PaddleOCR/Azure)
    or an axis-aligned ``[x1, y1, x2, y2]``. Clamps to the page bounds.
    """
    if width <= 0 or height <= 0:
        return [0.0, 0.0, 0.0, 0.0]

    flat = list(points_or_xyxy)
    if flat and isinstance(flat[0], (list, tuple)):
        xs = [float(p[0]) for p in flat]
        ys = [float(p[1]) for p in flat]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    else:
        x1, y1, x2, y2 = (float(v) for v in flat[:4])

    def _clamp(v: float, hi: int) -> float:
        return max(0.0, min(1.0, v / hi))

    return [_clamp(x1, width), _clamp(y1, height), _clamp(x2, width), _clamp(y2, height)]


def _preprocess_enabled() -> bool:
    """Whether OCR image preprocessing is on (config ``DOCUMENT_AI_OCR_PREPROCESS``)."""
    try:
        from open_webui.config import DOCUMENT_AI_OCR_PREPROCESS

        return bool(getattr(DOCUMENT_AI_OCR_PREPROCESS, "value", DOCUMENT_AI_OCR_PREPROCESS))
    except Exception:
        return True


def preprocess_for_ocr(
    img,
    *,
    binarize: bool = False,
    target_long_side: int = 2600,
    max_scale: float = 2.0,
):
    """Geometry-preserving OCR preprocessing. Returns a processed ndarray (or the input
    unchanged on any failure — never raises).

    Geometry is preserved (no deskew/crop, only uniform upscale), so the normalized boxes
    the engine produces still map onto the original page; the uniform scale cancels under
    normalization.

    Resolution is the safe, broadly-beneficial lever here: the pages are crisp digital PDF
    renders, on which denoise/contrast filters *destroy* thin strokes and RapidOCR/PaddleOCR
    already normalize internally — so for those (``binarize=False``) we ONLY upscale toward a
    higher effective resolution. ``binarize=True`` (Tesseract) additionally grayscales +
    CLAHE + adaptive-thresholds, which genuinely helps that engine.
    """
    try:
        import cv2
        import numpy as np  # noqa: F401  (ensures numpy is present)

        if img is None or getattr(img, "size", 0) == 0:
            return img

        out = img
        if binarize:
            gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY) if out.ndim == 3 else out
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        h, w = out.shape[:2]
        long_side = max(h, w)
        if long_side < target_long_side:
            scale = min(target_long_side / long_side, max_scale)
            if scale > 1.01:
                out = cv2.resize(
                    out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4
                )
        return out
    except Exception as e:
        log.warning(f"OCR preprocessing failed, using original image: {e}")
        return img


def maybe_preprocess(img, *, binarize: bool = False):
    """Apply :func:`preprocess_for_ocr` when enabled by config, else return ``img``."""
    if not _preprocess_enabled():
        return img
    return preprocess_for_ocr(img, binarize=binarize)


class OcrEngine(ABC):
    """Base class for real-OCR engines.

    Contract: ``extract_tokens`` must never raise into the caller — on any failure
    it returns ``[]`` so the surrounding extraction continues without grounded
    confidence. Concrete engines import their heavy backend lazily, keep the
    backend as a process-wide singleton, run the sync backend in a worker thread,
    and emit ``OcrToken``s with normalized boxes.
    """

    @abstractmethod
    async def extract_tokens(
        self, images_base64: List[str], page_numbers: Optional[List[int]] = None
    ) -> List[OcrToken]:
        """Extract OCR tokens from base64 PNG page images."""
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        """Stable identifier for this engine (e.g. ``"rapidocr"``)."""
        raise NotImplementedError
