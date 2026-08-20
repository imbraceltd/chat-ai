import base64
import asyncio
from logging import log
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI
from ..vision import VisionOCRProvider, OCRPage


class OpenAIVisionOCR(VisionOCRProvider):
    """OpenAI Vision API implementation for OCR."""
    
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4o", 
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1
    ):
        """
        Initialize OpenAI Vision OCR provider.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4o)
            base_url: Optional custom base URL for OpenAI API
            max_tokens: Maximum tokens for response
            temperature: Temperature for text generation
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize async client
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = AsyncOpenAI(**client_kwargs)
    
    async def extract_text_from_images(
        self, 
        images_base64: List[str], 
        page_numbers: List[int],
        additional_prompt: Optional[str] = None
    ) -> List[OCRPage]:
        """
        Extract text from images using OpenAI Vision API.
        
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
        
        # Process all images concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
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
        Process a single image for OCR.
        
        Args:
            image_base64: Base64-encoded PNG image
            page_number: Page number of this image
            additional_prompt: Optional additional instructions
            
        Returns:
            OCRPage with extracted text
        """
        # Build the prompt
        base_prompt = (
            "Please extract all text from this image in a structured and readable format. "
            "Maintain the original layout and formatting as much as possible. "
            "Include headers, paragraphs, lists, tables, and any other text elements. "
            "If there are tables, please format them clearly with proper alignment. "
            "Return only the extracted text without any additional commentary."
        )
        
        if additional_prompt:
            prompt = f"{base_prompt}\n\nAdditional instructions: {additional_prompt}"
        else:
            prompt = base_prompt
        
        try:
            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract text and metadata
            extracted_text = response.choices[0].message.content or ""
            
            # Calculate confidence based on finish reason and response quality
            confidence = self._calculate_confidence(response)
            
            metadata = {
                "model": self.model,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            }
            
            return OCRPage(
                page_number=page_number,
                extracted_text=extracted_text,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error for page {page_number}: {str(e)}")
    
    def _calculate_confidence(self, response) -> float:
        """
        Calculate confidence score based on OpenAI response.
        
        Args:
            response: OpenAI chat completion response
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.8
        
        # Adjust based on finish reason
        if response.choices[0].finish_reason == "stop":
            confidence = 0.9
        elif response.choices[0].finish_reason == "length":
            confidence = 0.7  # Response was cut off
        else:
            confidence = 0.6
        
        # Adjust based on response length (longer responses generally indicate more content)
        response_length = len(response.choices[0].message.content or "")
        if response_length > 1000:
            confidence = min(confidence + 0.1, 1.0)
        elif response_length < 100:
            confidence = max(confidence - 0.2, 0.1)
        
        return confidence
    
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        return "OpenAI Vision"


# Convenience function for quick usage
async def extract_text_from_pdf_url_openai(
    pdf_url: str,
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    max_pages: Optional[int] = None,
    dpi: int = 300,
    additional_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to extract text from PDF URL using OpenAI.
    
    Args:
        pdf_url: URL of the PDF file
        api_key: OpenAI API key
        model: OpenAI model to use
        base_url: Optional custom base URL
        max_pages: Maximum pages to process
        dpi: DPI for image conversion
        additional_prompt: Additional OCR instructions
        
    Returns:
        Dictionary with extracted text and metadata
    """
    from ..vision import PDFOCRProcessor
    
    provider = OpenAIVisionOCR(
        api_key=api_key,
        model=model,
        base_url=base_url
    )
    
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
async def extract_text_from_pdf_buffer_openai(
    pdf_buffer: bytes,
    api_key: str,
    model: str = "gpt-4o",
    base_url: Optional[str] = None,
    max_pages: Optional[int] = None,
    dpi: int = 300,
    additional_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick function to extract text from PDF buffer using OpenAI.
    
    Args:
        pdf_buffer: PDF file data as bytes
        api_key: OpenAI API key
        model: OpenAI model to use
        base_url: Optional custom base URL
        max_pages: Maximum pages to process
        dpi: DPI for image conversion
        additional_prompt: Additional OCR instructions
        
    Returns:
        Dictionary with extracted text and metadata
    """
    from ..vision import PDFOCRProcessor
    
    provider = OpenAIVisionOCR(
        api_key=api_key,
        model=model,
        base_url=base_url
    )
    
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