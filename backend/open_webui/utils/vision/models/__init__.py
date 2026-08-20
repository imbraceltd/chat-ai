"""
Vision model implementations for different LLM providers.

This module contains specific implementations of the VisionOCRProvider interface
for various vision-capable language model providers.

Available providers:
- OpenAI Vision API (gpt-4o, gpt-4-vision-preview)
- Ollama vision models (llama3.2-vision, llava, etc.)
- More providers can be added by implementing the VisionOCRProvider interface
"""

from .openai import OpenAIVisionOCR, extract_text_from_pdf_url_openai, extract_text_from_pdf_buffer_openai
from .ollama import OllamaVisionOCR, extract_text_from_pdf_url_ollama, extract_text_from_pdf_buffer_ollama

__all__ = [
    "OpenAIVisionOCR",
    "OllamaVisionOCR", 
    "extract_text_from_pdf_url_openai",
    "extract_text_from_pdf_buffer_openai",
    "extract_text_from_pdf_url_ollama",
    "extract_text_from_pdf_buffer_ollama"
]
