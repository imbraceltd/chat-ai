"""Real-OCR engines for grounding Document AI extraction confidence.

This package is intentionally boot-clean: importing it pulls in only the factory,
the dataclasses, and the (stdlib-only) reconciliation logic. Concrete engine
backends (RapidOCR/PaddleOCR/Tesseract/Azure) are imported lazily by the factory
when — and only when — an engine is actually selected.
"""

from open_webui.utils.ocr.base import OcrEngine, OcrToken
from open_webui.utils.ocr.factory import create_ocr_engine
from open_webui.utils.ocr.reconcile import aggregate_confidence, reconcile_fields

__all__ = [
    "OcrEngine",
    "OcrToken",
    "create_ocr_engine",
    "reconcile_fields",
    "aggregate_confidence",
]
