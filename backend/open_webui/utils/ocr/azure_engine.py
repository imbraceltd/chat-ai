"""Azure Document Intelligence engine (optional, cloud).

Selected only when ``DOCUMENT_AI_OCR_ENGINE=azure``. The ``azure-ai-documentintelligence``
SDK is already pinned; this engine is gated on the existing ``DOCUMENT_INTELLIGENCE_ENDPOINT``
/ ``DOCUMENT_INTELLIGENCE_KEY`` config. Uses the ``prebuilt-read`` model, whose words carry
both ``confidence`` and a ``polygon``.

NOTE: this calls the SDK directly. The existing ``AzureAIDocumentIntelligenceLoader`` in
``retrieval/loaders/main.py`` returns plain text only (no per-word confidence/geometry), so
it is not reusable here.
"""

import asyncio
import base64
import logging
from typing import List, Optional

from open_webui.utils.ocr.base import OcrEngine, OcrToken, normalize_box

log = logging.getLogger(__name__)

_semaphore = None
_MAX_CONCURRENCY = 4


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)
    return _semaphore


def _config_value(cfg) -> str:
    """Read a PersistentConfig (or plain value) as a string."""
    return str(getattr(cfg, "value", cfg) or "")


def is_available() -> bool:
    from open_webui.config import (
        DOCUMENT_INTELLIGENCE_ENDPOINT,
        DOCUMENT_INTELLIGENCE_KEY,
    )

    return bool(
        _config_value(DOCUMENT_INTELLIGENCE_ENDPOINT)
        and _config_value(DOCUMENT_INTELLIGENCE_KEY)
    )


class AzureDocIntelligenceEngine(OcrEngine):
    def get_name(self) -> str:
        return "azure"

    def _run_one(self, image_base64: str, page: int) -> List[OcrToken]:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        from azure.core.credentials import AzureKeyCredential
        from open_webui.config import (
            DOCUMENT_INTELLIGENCE_ENDPOINT,
            DOCUMENT_INTELLIGENCE_KEY,
        )

        endpoint = _config_value(DOCUMENT_INTELLIGENCE_ENDPOINT)
        key = _config_value(DOCUMENT_INTELLIGENCE_KEY)
        if not endpoint or not key:
            return []

        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            log.warning(f"Azure DI: failed to decode page {page}: {e}")
            return []

        client = DocumentIntelligenceClient(endpoint, AzureKeyCredential(key))
        poller = client.begin_analyze_document(
            "prebuilt-read",
            AnalyzeDocumentRequest(bytes_source=image_bytes),
        )
        result = poller.result()

        tokens: List[OcrToken] = []
        for az_page in result.pages or []:
            pw = float(az_page.width or 0)
            ph = float(az_page.height or 0)
            for word in getattr(az_page, "words", None) or []:
                text = (word.content or "").strip()
                if not text:
                    continue
                # polygon is a flat [x1, y1, x2, y2, ...] in the page's unit
                polygon = list(word.polygon or [])
                points = [
                    [polygon[i], polygon[i + 1]] for i in range(0, len(polygon) - 1, 2)
                ]
                box = normalize_box(points, int(pw), int(ph)) if points and pw and ph else [0.0, 0.0, 0.0, 0.0]
                tokens.append(
                    OcrToken(
                        text=text,
                        box=box,
                        confidence=float(word.confidence or 0.0),
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
                    log.warning(f"Azure DI: page {page} failed: {e}")
                    return []

        results = await asyncio.gather(
            *(_process(img, page) for img, page in zip(images_base64, pages))
        )
        return [tok for page_tokens in results for tok in page_tokens]
