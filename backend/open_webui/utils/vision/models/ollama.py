import base64
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from ..vision import VisionOCRProvider, OCRPage


class OllamaVisionOCR(VisionOCRProvider):
    """Ollama Vision implementation for OCR using vision-capable models."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2-vision",
        timeout: int = 300,  # 5 minutes timeout for vision processing
        temperature: float = 0.1
    ):
        """
        Initialize Ollama Vision OCR provider.
        
        Args:
            base_url: Ollama server base URL
            model: Ollama vision model to use (e.g., llava, llama3.2-vision)
            timeout: Request timeout in seconds
            temperature: Temperature for text generation
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
    
    async def extract_text_from_images(
        self, 
        images_base64: List[str], 
        page_numbers: List[int],
        additional_prompt: Optional[str] = None
    ) -> List[OCRPage]:
        """
        Extract text from images using Ollama Vision API.
        
        Args:
            images_base64: List of base64-encoded PNG images
            page_numbers: Corresponding page numbers for each image
            additional_prompt: Optional additional instructions for the OCR model
            
        Returns:
            List of OCRPage objects with extracted text
        """
        if len(images_base64) != len(page_numbers):
            raise ValueError("Number of images must match number of page numbers")
        
        tasks = []
        for image_base64, page_number in zip(images_base64, page_numbers):
            task = self._process_single_image(image_base64, page_number, additional_prompt)
            tasks.append(task)
        
        # Process images with limited concurrency to avoid overwhelming the server
        semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent requests
        
        async def process_with_semaphore(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        ocr_pages = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error page
                ocr_pages.append(OCRPage(
                    page_number=page_numbers[i],
                    extracted_text=f"Error processing page: {str(result)}",
                    confidence=0.0,
                    metadata={"error": str(result)}
                ))
            else:
                ocr_pages.append(result)
        
        return ocr_pages
    
    async def _process_single_image(
        self, 
        image_base64: str, 
        page_number: int, 
        additional_prompt: Optional[str] = None
    ) -> OCRPage:
        """
        Process a single image for OCR using Ollama.
        
        Args:
            image_base64: Base64-encoded PNG image
            page_number: Page number of this image
            additional_prompt: Optional additional instructions
            
        Returns:
            OCRPage with extracted text
        """
        # Build the prompt
        base_prompt = (
            "Please extract all text from this image accurately and completely. "
            "Maintain the original structure, formatting, and layout as much as possible. "
            "Include all headers, paragraphs, lists, tables, captions, and any other text elements. "
            "For tables, preserve the structure with clear column and row organization. "
            "Return only the extracted text content without any additional commentary or descriptions."
        )
        
        if additional_prompt:
            prompt = f"{base_prompt}\n\nAdditional instructions: {additional_prompt}"
        else:
            prompt = base_prompt
        
        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 4096  # Max tokens for response
                }
            }
            
            # Make request to Ollama API
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama API error: HTTP {response.status}")
                    
                    result = await response.json()
            
            # Extract text from response
            extracted_text = result.get("response", "")
            
            # Calculate confidence based on response quality
            confidence = self._calculate_confidence(result)
            
            metadata = {
                "model": self.model,
                "total_duration": result.get("total_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "prompt_eval_count": result.get("prompt_eval_count", 0),
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0),
            }
            
            return OCRPage(
                page_number=page_number,
                extracted_text=extracted_text,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            raise Exception(f"Ollama API error for page {page_number}: {str(e)}")
    
    def _calculate_confidence(self, response: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on Ollama response.
        
        Args:
            response: Ollama API response
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence for Ollama
        confidence = 0.75
        
        # Adjust based on response length
        response_text = response.get("response", "")
        response_length = len(response_text)
        
        if response_length > 1000:
            confidence = min(confidence + 0.1, 0.95)
        elif response_length < 50:
            confidence = max(confidence - 0.3, 0.2)
        
        # Adjust based on processing time (faster might indicate less thorough processing)
        total_duration = response.get("total_duration", 0)
        if total_duration > 0:
            # Convert nanoseconds to seconds
            duration_seconds = total_duration / 1_000_000_000
            if duration_seconds > 30:  # Longer processing might indicate more careful analysis
                confidence = min(confidence + 0.05, 0.95)
            elif duration_seconds < 5:  # Very fast might be less thorough
                confidence = max(confidence - 0.1, 0.3)
        
        return confidence
    
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        return f"Ollama ({self.model})"
    
    async def check_model_availability(self) -> bool:
        """
        Check if the specified model is available on the Ollama server.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [model.get("name", "") for model in data.get("models", [])]
                        return any(self.model in model_name for model_name in models)
            return False
        except Exception:
            return False


# Convenience function for quick usage
async def extract_text_from_pdf_url_ollama(
    pdf_url: str,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2-vision",
    max_pages: Optional[int] = None,
    dpi: int = 300,
    additional_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to extract text from PDF URL using Ollama.
    
    Args:
        pdf_url: URL of the PDF file
        base_url: Ollama server base URL
        model: Ollama vision model to use
        max_pages: Maximum pages to process
        dpi: DPI for image conversion
        additional_prompt: Additional OCR instructions
        
    Returns:
        Dictionary with extracted text and metadata
    """
    from ..vision import PDFOCRProcessor
    
    provider = OllamaVisionOCR(
        base_url=base_url,
        model=model
    )
    
    # Check if model is available
    if not await provider.check_model_availability():
        return {
            "success": False,
            "error_message": f"Model '{model}' is not available on Ollama server at {base_url}",
            "total_pages": 0,
            "pages": []
        }
    
    processor = PDFOCRProcessor(provider)
    result = await processor.process_pdf_from_url(
        pdf_url=pdf_url,
        max_pages=max_pages,
        dpi=dpi,
        additional_prompt=additional_prompt
    )
    
    # Convert to dictionary format
    return {
        "success": result.success,
        "total_pages": result.total_pages,
        "processing_time_seconds": result.processing_time_seconds,
        "error_message": result.error_message,
        "pages": [
            {
                "page_number": page.page_number,
                "extracted_text": page.extracted_text,
                "confidence": page.confidence,
                "metadata": page.metadata
            }
            for page in result.pages
        ]
    }


# Convenience function for quick usage with file buffers
async def extract_text_from_pdf_buffer_ollama(
    pdf_buffer: bytes,
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2-vision",
    max_pages: Optional[int] = None,
    dpi: int = 300,
    additional_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to extract text from PDF buffer using Ollama.
    
    Args:
        pdf_buffer: PDF file data as bytes
        base_url: Ollama server base URL
        model: Ollama vision model to use
        max_pages: Maximum pages to process
        dpi: DPI for image conversion
        additional_prompt: Additional OCR instructions
        
    Returns:
        Dictionary with extracted text and metadata
    """
    from ..vision import PDFOCRProcessor
    
    provider = OllamaVisionOCR(
        base_url=base_url,
        model=model
    )
    
    # Check if model is available
    if not await provider.check_model_availability():
        return {
            "success": False,
            "error_message": f"Model '{model}' is not available on Ollama server at {base_url}",
            "total_pages": 0,
            "pages": []
        }
    
    processor = PDFOCRProcessor(provider)
    result = await processor.process_pdf_from_buffer(
        pdf_buffer=pdf_buffer,
        max_pages=max_pages,
        dpi=dpi,
        additional_prompt=additional_prompt
    )
    
    # Convert to dictionary format
    return {
        "success": result.success,
        "total_pages": result.total_pages,
        "processing_time_seconds": result.processing_time_seconds,
        "error_message": result.error_message,
        "pages": [
            {
                "page_number": page.page_number,
                "extracted_text": page.extracted_text,
                "confidence": page.confidence,
                "metadata": page.metadata
            }
            for page in result.pages
        ]
    }