"""RapidOCR engine (default).

Two modes, chosen at runtime:

- **External API** (when ``RAPIDOCR_API_URL`` is configured): POST each page image to a
  ``rapidocr_api`` server (``POST /ocr`` with form field ``image_data``). No local model
  is loaded, so the heavy ``rapidocr_onnxruntime`` package isn't required in this process.
- **Local ONNX** (default): ``rapidocr_onnxruntime`` (pinned in ``requirements.txt``),
  loaded once as a process-wide singleton and imported lazily.
"""

import asyncio
import logging
import os
from typing import List, Optional

from open_webui.utils.ocr.base import (
    OcrEngine,
    OcrToken,
    decode_image,
    image_dimensions,
    maybe_preprocess,
    normalize_box,
)

log = logging.getLogger(__name__)

_engine = None  # cached RapidOCR() instance (model load is the expensive part)
_semaphore = None
_MAX_CONCURRENCY = 4


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    return _semaphore


def get_api_url() -> str:
    """Resolved external RapidOCR API endpoint, or '' when running locally.

    Accepts a base URL (``http://host:9003``) or a full endpoint ending in ``/ocr``.
    """
    try:
        from open_webui.config import RAPIDOCR_API_URL

        url = str(getattr(RAPIDOCR_API_URL, "value", RAPIDOCR_API_URL) or "").strip()
    except Exception:
        url = ""
    if not url:
        return ""
    return url if url.rstrip("/").endswith("/ocr") else url.rstrip("/") + "/ocr"


def _get_api_timeout() -> float:
    try:
        return float(os.getenv("RAPIDOCR_API_TIMEOUT", "60"))
    except (TypeError, ValueError):
        return 60.0


def _tokens_from_api_result(result, width: int, height: int, page: int) -> List[OcrToken]:
    """Map a rapidocr_api /ocr JSON response into OcrTokens.

    Response is a dict keyed "0","1",... each ``{rec_txt, dt_boxes(4 pts), score}``; ``{}``
    when nothing is detected. Boxes are in pixel space and normalized against (width, height).
    """
    if not isinstance(result, dict) or not result:
        return []
    tokens: List[OcrToken] = []
    for item in result.values():
        if not isinstance(item, dict):
            continue
        text = item.get("rec_txt")
        if not text:
            continue
        box_pts = item.get("dt_boxes")
        try:
            score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        tokens.append(
            OcrToken(
                text=str(text),
                box=normalize_box(box_pts, width, height) if box_pts else [0.0, 0.0, 0.0, 0.0],
                confidence=score,
                page=page,
            )
        )
    return tokens


def _get_engine():
    """Lazily build (and cache) the local RapidOCR backend. Heavy import lives here."""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR  # lazy: only when running locally

        _engine = RapidOCR()
        log.info("RapidOCR engine initialized (local ONNX)")
    return _engine


class RapidOcrEngine(OcrEngine):
    def get_name(self) -> str:
        return "rapidocr"

    def _run_one(self, image_base64: str, page: int) -> List[OcrToken]:
        """Synchronous single-image OCR against the local model (runs in a worker thread)."""
        try:
            img, width, height = decode_image(image_base64)
        except Exception as e:
            log.warning(f"RapidOCR: failed to decode page {page}: {e}")
            return []

        img = maybe_preprocess(img)  # natural-image detector: no binarization
        height, width = img.shape[:2]  # dims may change after upscale; boxes normalize to these

        engine = _get_engine()
        result, _elapse = engine(img)
        if not result:
            return []

        tokens: List[OcrToken] = []
        for line in result:
            # RapidOCR returns [box(4 points), text, score]
            try:
                box, text, score = line[0], line[1], line[2]
            except (IndexError, TypeError):
                continue
            if not text:
                continue
            tokens.append(
                OcrToken(
                    text=str(text),
                    box=normalize_box(box, width, height),
                    confidence=float(score),
                    page=page,
                )
            )
        return tokens

    async def _run_one_api(self, api_url: str, image_base64: str, page: int) -> List[OcrToken]:
        """Single-image OCR against an external rapidocr_api server (POST /ocr).

        Uploads the image as a multipart file (``image_file``) rather than the base64
        ``image_data`` form field. File parts stream to disk on the server, so they avoid
        its per-field size cap ("Part exceeded maximum size of 1024KB") that page-sized
        base64 images exceed; raw bytes are also ~33% smaller than base64.
        """
        import base64 as b64
        import httpx

        # Box coords come back in the server's pixel space; normalize against the same image.
        try:
            img_bytes = b64.b64decode(image_base64)
            width, height = image_dimensions(image_base64)
        except Exception as e:
            log.warning(f"RapidOCR API: failed to decode page {page}: {e}")
            return []

        async with httpx.AsyncClient(timeout=_get_api_timeout()) as client:
            resp = await client.post(
                api_url, files={"image_file": (f"page-{page}.png", img_bytes, "image/png")}
            )
            resp.raise_for_status()
            result = resp.json()

        return _tokens_from_api_result(result, width, height, page)

    async def extract_tokens(
        self, images_base64: List[str], page_numbers: Optional[List[int]] = None
    ) -> List[OcrToken]:
        if not images_base64:
            return []
        pages = page_numbers or list(range(1, len(images_base64) + 1))
        semaphore = _get_semaphore()
        api_url = get_api_url()
        if api_url:
            log.info(f"RapidOCR using external API: {api_url}")

        async def _process(image_base64: str, page: int) -> List[OcrToken]:
            async with semaphore:
                try:
                    if api_url:
                        return await self._run_one_api(api_url, image_base64, page)
                    return await asyncio.to_thread(self._run_one, image_base64, page)
                except Exception as e:
                    log.warning(f"RapidOCR: page {page} failed: {e}")
                    return []

        results = await asyncio.gather(
            *(_process(img, page) for img, page in zip(images_base64, pages))
        )
        return [tok for page_tokens in results for tok in page_tokens]
