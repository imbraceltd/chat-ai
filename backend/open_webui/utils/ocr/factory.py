"""Engine factory — env-driven and boot-clean.

The active engine is resolved from (in order): an explicit per-request override →
the ``DOCUMENT_AI_OCR_ENGINE`` config → ``"rapidocr"``. Only the resolved engine's
module is imported (lazily, inside ``_build``), so no OCR backend loads at server
boot and only the selected engine's dependencies need be installed.

If the resolved engine's package or config is missing, this logs a clear error and
returns ``None`` — OCR confidence is then disabled for that request while the
extraction itself still proceeds.
"""

import importlib.util
import logging
from typing import Optional

from open_webui.utils.ocr.base import OcrEngine

log = logging.getLogger(__name__)

# engine name -> the top-level package that must be importable for it to work
_REQUIRED_PACKAGE = {
    "rapidocr": "rapidocr_onnxruntime",
    "paddleocr": "paddleocr",
    "tesseract": "pytesseract",
    "azure": "azure.ai.documentintelligence",
}

_ENGINE_CACHE = {}  # name -> built OcrEngine (successful builds only)


def _resolve_name(name: Optional[str]) -> str:
    if name and name.strip():
        return name.strip().lower()
    try:
        from open_webui.config import DOCUMENT_AI_OCR_ENGINE

        cfg = str(getattr(DOCUMENT_AI_OCR_ENGINE, "value", DOCUMENT_AI_OCR_ENGINE) or "")
        if cfg.strip():
            return cfg.strip().lower()
    except Exception:
        pass
    return "rapidocr"


def _package_installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def _rapidocr_api_configured() -> bool:
    """True when an external RapidOCR API server is configured (RAPIDOCR_API_URL)."""
    try:
        from open_webui.config import RAPIDOCR_API_URL

        return bool(str(getattr(RAPIDOCR_API_URL, "value", RAPIDOCR_API_URL) or "").strip())
    except Exception:
        return False


def _build(name: str) -> Optional[OcrEngine]:
    if name == "rapidocr":
        from open_webui.utils.ocr.rapidocr_engine import RapidOcrEngine

        return RapidOcrEngine()
    if name == "paddleocr":
        from open_webui.utils.ocr.paddleocr_engine import PaddleOcrEngine

        return PaddleOcrEngine()
    if name == "tesseract":
        from open_webui.utils.ocr.tesseract_engine import TesseractOcrEngine

        return TesseractOcrEngine()
    if name == "azure":
        from open_webui.utils.ocr.azure_engine import AzureDocIntelligenceEngine, is_available

        if not is_available():
            log.error(
                "OCR engine 'azure' selected but DOCUMENT_INTELLIGENCE_ENDPOINT/KEY are not "
                "configured; OCR confidence disabled."
            )
            return None
        return AzureDocIntelligenceEngine()
    return None


def create_ocr_engine(name: Optional[str] = None) -> Optional[OcrEngine]:
    """Resolve and build the configured OCR engine, or ``None`` if unavailable."""
    resolved = _resolve_name(name)

    if resolved in _ENGINE_CACHE:
        return _ENGINE_CACHE[resolved]

    required = _REQUIRED_PACKAGE.get(resolved)
    if required is None:
        log.error(
            f"Unknown OCR engine '{resolved}'. Valid: {sorted(_REQUIRED_PACKAGE)}. "
            "OCR confidence disabled."
        )
        return None

    # rapidocr can run against an external RapidOCR API server, in which case the local
    # onnxruntime package isn't required in this process.
    skip_pkg_check = resolved == "rapidocr" and _rapidocr_api_configured()
    if not skip_pkg_check and not _package_installed(required):
        log.error(
            f"OCR engine '{resolved}' selected but package '{required}' is not installed. "
            "Install it in the deployment or set DOCUMENT_AI_OCR_ENGINE to an available engine. "
            "OCR confidence disabled."
        )
        return None

    try:
        engine = _build(resolved)
    except Exception as e:
        log.error(f"Failed to initialize OCR engine '{resolved}': {e}. OCR confidence disabled.")
        return None

    if engine is not None:
        _ENGINE_CACHE[resolved] = engine
    return engine
