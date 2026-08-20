"""PaddleOCR engine (optional, install-on-demand).

Selected only when ``DOCUMENT_AI_OCR_ENGINE=paddleocr``. Requires the operator to
install ``paddleocr`` + ``paddlepaddle`` (heavy) — these are imported lazily here
so they never load at boot or when another engine is active.
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

_engine = None
_semaphore = None
_MAX_CONCURRENCY = 2  # PaddleOCR is heavier; keep concurrency conservative
_LANG = os.getenv("DOCUMENT_AI_OCR_LANG", "en")


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    return _semaphore


def _get_engine():
    global _engine
    if _engine is None:
        from paddleocr import PaddleOCR  # lazy: only when selected

        _engine = PaddleOCR(use_angle_cls=True, lang=_LANG, show_log=False)
        log.info(f"PaddleOCR engine initialized (lang={_LANG})")
    return _engine


class PaddleOcrEngine(OcrEngine):
    def get_name(self) -> str:
        return "paddleocr"

    def _run_one(self, image_base64: str, page: int) -> List[OcrToken]:
        try:
            img, width, height = decode_image(image_base64)
        except Exception as e:
            log.warning(f"PaddleOCR: failed to decode page {page}: {e}")
            return []

        img = maybe_preprocess(img)  # natural-image detector: no binarization
        height, width = img.shape[:2]  # dims may change after upscale; boxes normalize to these

        engine = _get_engine()
        result = engine.ocr(img, cls=True)
        # PaddleOCR returns one list per image: [[ [box, (text, score)], ... ]]
        if not result:
            return []
        page_result = result[0]
        if not page_result:
            return []

        tokens: List[OcrToken] = []
        for line in page_result:
            try:
                box, (text, score) = line[0], line[1]
            except (IndexError, TypeError, ValueError):
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
                    log.warning(f"PaddleOCR: page {page} failed: {e}")
                    return []

        results = await asyncio.gather(
            *(_process(img, page) for img, page in zip(images_base64, pages))
        )
        return [tok for page_tokens in results for tok in page_tokens]
