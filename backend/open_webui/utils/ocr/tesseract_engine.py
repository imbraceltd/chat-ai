"""Tesseract engine (optional, install-on-demand).

Selected only when ``DOCUMENT_AI_OCR_ENGINE=tesseract``. Requires the operator to
install ``pytesseract`` and the ``tesseract`` system binary. Word-level text,
confidence and geometry come from ``image_to_data``.
"""

import asyncio
import logging
import os
from typing import List, Optional

from open_webui.utils.ocr.base import (
    OcrEngine,
    OcrToken,
    decode_image,
    maybe_preprocess,
    normalize_box,
)

log = logging.getLogger(__name__)

_semaphore = None
_MAX_CONCURRENCY = 4
_LANG = os.getenv("DOCUMENT_AI_OCR_LANG", "eng")  # tesseract uses 3-letter codes


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    return _semaphore


class TesseractOcrEngine(OcrEngine):
    def get_name(self) -> str:
        return "tesseract"

    def _run_one(self, image_base64: str, page: int) -> List[OcrToken]:
        import pytesseract  # lazy: only when selected
        from pytesseract import Output

        try:
            img, width, height = decode_image(image_base64)
        except Exception as e:
            log.warning(f"Tesseract: failed to decode page {page}: {e}")
            return []

        img = maybe_preprocess(img, binarize=True)  # tesseract benefits from binarization
        height, width = img.shape[:2]  # dims may change after upscale; boxes normalize to these

        data = pytesseract.image_to_data(img, lang=_LANG, output_type=Output.DICT)
        n = len(data.get("text", []))
        tokens: List[OcrToken] = []
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 0:  # tesseract uses -1 for non-text regions
                continue
            left = float(data["left"][i])
            top = float(data["top"][i])
            w = float(data["width"][i])
            h = float(data["height"][i])
            tokens.append(
                OcrToken(
                    text=text,
                    box=normalize_box([left, top, left + w, top + h], width, height),
                    confidence=conf / 100.0,
                    page=page,
                )
            )
        return tokens

    async def extract_tokens(
        self, images_base64: List[str], page_numbers: Optional[List[int]] = None
    ) -> List[OcrToken]:
        if not images_base64:
            return []
        pages = page_numbers or list(range(1, len(images_base64) + 1))
        semaphore = _get_semaphore()

        async def _process(image_base64: str, page: int) -> List[OcrToken]:
            async with semaphore:
                try:
                    return await asyncio.to_thread(self._run_one, image_base64, page)
                except Exception as e:
                    log.warning(f"Tesseract: page {page} failed: {e}")
                    return []

        results = await asyncio.gather(
            *(_process(img, page) for img, page in zip(images_base64, pages))
        )
        return [tok for page_tokens in results for tok in page_tokens]
