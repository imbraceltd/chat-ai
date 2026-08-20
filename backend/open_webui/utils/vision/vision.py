import base64
import io
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import aiohttp
import asyncio

# Try to import pdf2image first, fall back to PyMuPDF if Poppler is not available
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class OCRPage:
    """Represents a single page extracted from a PDF with OCR results."""
    page_number: int
    extracted_text: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OCRResult:
    """Complete OCR result for a PDF document."""
    pages: List[OCRPage]
    total_pages: int
    success: bool
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None


class VisionOCRProvider(ABC):
    """Abstract base class for vision OCR providers."""
    
    @abstractmethod
    async def extract_text_from_images(
        self, 
        images_base64: List[str], 
        page_numbers: List[int],
        additional_prompt: Optional[str] = None
    ) -> List[OCRPage]:
        """
        Extract text from a list of base64-encoded images.
        
        Args:
            images_base64: List of base64-encoded PNG images
            page_numbers: Corresponding page numbers for each image
            additional_prompt: Optional additional instructions for the OCR model
            
        Returns:
            List of OCRPage objects with extracted text
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass


class PDFOCRProcessor:
    """Main processor for PDF OCR operations."""
    
    def __init__(self, provider: VisionOCRProvider):
        self.provider = provider
    
    async def process_pdf_from_url(
        self, 
        pdf_url: str, 
        max_pages: Optional[int] = None,
        dpi: int = 300,
        additional_prompt: Optional[str] = None
    ) -> OCRResult:
        """
        Process a PDF from URL and extract text using OCR.
        
        Args:
            pdf_url: URL of the PDF file
            max_pages: Maximum number of pages to process (None for all pages)
            dpi: DPI for image conversion (higher = better quality, larger files)
            additional_prompt: Optional additional instructions for the OCR model
            
        Returns:
            OCRResult containing all extracted pages
        """
        import time
        start_time = time.time()
        
        try:
            # Download PDF from URL
            pdf_data = await self._download_pdf(pdf_url)
            
            # Convert PDF to images
            images_data = await self._pdf_to_images(pdf_data, max_pages, dpi)
            
            # Extract text using vision provider
            page_numbers = [data["page_number"] for data in images_data]
            images_base64 = [data["image_base64"] for data in images_data]
            
            ocr_pages = await self.provider.extract_text_from_images(
                images_base64, page_numbers, additional_prompt
            )
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                pages=ocr_pages,
                total_pages=len(ocr_pages),
                success=True,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return OCRResult(
                pages=[],
                total_pages=0,
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )
    
    async def _download_pdf(self, url: str) -> bytes:
        """Download PDF from URL."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download PDF: HTTP {response.status}")
                return await response.read()
    
    async def _pdf_to_images(
        self, 
        pdf_data: bytes, 
        max_pages: Optional[int] = None, 
        dpi: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Convert PDF to images using pdf2image or PyMuPDF as fallback.
        
        Args:
            pdf_data: PDF file data as bytes
            max_pages: Maximum number of pages to convert
            dpi: DPI for image conversion
            
        Returns:
            List of dictionaries containing page_number and image_base64
        """
        def _convert_pdf_with_pdf2image():
            """Convert PDF using pdf2image (requires Poppler)."""
            if not PDF2IMAGE_AVAILABLE:
                raise ImportError("pdf2image is not available")
                
            # Convert PDF to PIL Images using pdf2image
            images = convert_from_bytes(
                pdf_data,
                dpi=dpi,
                first_page=1,
                last_page=max_pages,  # pdf2image handles None correctly
                fmt='PNG'
            )
            
            images_data = []
            
            for page_num, image in enumerate(images):
                # Convert PIL Image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                img_bytes = img_buffer.getvalue()
                
                # Convert to base64
                image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                images_data.append({
                    "page_number": page_num + 1,  # 1-indexed
                    "image_base64": image_base64
                })
            
            return images_data
        
        def _convert_pdf_with_pymupdf():
            """Convert PDF using PyMuPDF (no external dependencies)."""
            if not PYMUPDF_AVAILABLE:
                raise ImportError("PyMuPDF is not available")
                
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            images_data = []
            
            # Calculate DPI scaling factor (PyMuPDF uses matrix scaling)
            zoom = dpi / 72.0  # 72 DPI is default
            mat = fitz.Matrix(zoom, zoom)
            
            page_count = len(pdf_document)
            end_page = min(max_pages, page_count) if max_pages else page_count
            
            for page_num in range(end_page):
                page = pdf_document[page_num]
                
                # Render page to image
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                # Convert to base64
                image_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                images_data.append({
                    "page_number": page_num + 1,  # 1-indexed
                    "image_base64": image_base64
                })
            
            pdf_document.close()
            return images_data
        
        def _convert_pdf_sync():
            """Synchronous PDF conversion with fallback logic."""
            # Try PyMuPDF first (no external dependencies required)
            try:
                return _convert_pdf_with_pymupdf()
            except (ImportError, Exception) as e:
                print(f"PyMuPDF failed ({e}), falling back to pdf2image...")
                
                # Fall back to pdf2image (requires Poppler)
                try:
                    return _convert_pdf_with_pdf2image()
                except (ImportError, Exception) as e2:
                    raise Exception(f"Both PDF conversion methods failed. PyMuPDF: {e}, pdf2image: {e2}. Please install PyMuPDF or Poppler (for pdf2image).")
        
        # Run the synchronous function in a thread pool
        return await asyncio.to_thread(_convert_pdf_sync)
    
    async def process_pdf_from_buffer(
        self, 
        pdf_buffer: bytes, 
        max_pages: Optional[int] = None,
        dpi: int = 300,
        additional_prompt: Optional[str] = None
    ) -> OCRResult:
        """
        Process a PDF from buffer data and extract text using OCR.
        
        Args:
            pdf_buffer: PDF file data as bytes
            max_pages: Maximum number of pages to process (None for all pages)
            dpi: DPI for image conversion (higher = better quality, larger files)
            additional_prompt: Optional additional instructions for the OCR model
            
        Returns:
            OCRResult containing all extracted pages
        """
        import time
        start_time = time.time()
        
        try:
            # Convert PDF to images directly from buffer
            images_data = await self._pdf_to_images(pdf_buffer, max_pages, dpi)
            
            # Extract text using vision provider
            page_numbers = [data["page_number"] for data in images_data]
            images_base64 = [data["image_base64"] for data in images_data]
            
            ocr_pages = await self.provider.extract_text_from_images(
                images_base64, page_numbers, additional_prompt
            )
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                pages=ocr_pages,
                total_pages=len(ocr_pages),
                success=True,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return OCRResult(
                pages=[],
                total_pages=0,
                success=False,
                error_message=str(e),
                processing_time_seconds=processing_time
            )

# Convenience function to create processor with different providers
def create_ocr_processor(provider_name: str, **provider_kwargs) -> PDFOCRProcessor:
    """
    Create an OCR processor with the specified provider.
    
    Args:
        provider_name: Name of the provider ('openai', 'ollama', etc.)
        **provider_kwargs: Additional arguments for the provider
        
    Returns:
        Configured PDFOCRProcessor
    """
    from .models.openai import OpenAIVisionOCR
    from .models.ollama import OllamaVisionOCR
    
    if provider_name.lower() == "openai":
        provider = OpenAIVisionOCR(**provider_kwargs)
    elif provider_name.lower() == "ollama":
        provider = OllamaVisionOCR(**provider_kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
    
    return PDFOCRProcessor(provider)