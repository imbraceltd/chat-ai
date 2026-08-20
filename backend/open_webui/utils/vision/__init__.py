"""
Vision OCR package for extracting text from PDFs using various vision LLM providers.

This package provides a unified interface for performing OCR on PDF documents
by converting pages to images and using vision-capable language models.

Key components:
- VisionOCRProvider: Abstract base class for all vision providers
- PDFOCRProcessor: Main processor that handles PDF conversion and orchestration
- OpenAIVisionOCR: OpenAI Vision API implementation
- OllamaVisionOCR: Ollama vision model implementation

Quick usage:
    from open_webui.utils.vision import create_ocr_processor
    
    processor = create_ocr_processor("openai", api_key="your-key")
    result = await processor.process_pdf_from_url("https://example.com/doc.pdf")
"""

from .vision import (
    VisionOCRProvider,
    PDFOCRProcessor,
    OCRPage,
    OCRResult,
    create_ocr_processor
)

__all__ = [
    "VisionOCRProvider",
    "PDFOCRProcessor", 
    "OCRPage",
    "OCRResult",
    "create_ocr_processor"
]
