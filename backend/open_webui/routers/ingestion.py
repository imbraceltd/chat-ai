import asyncio
import logging
import os
import uuid
import tempfile
from typing import Optional, Dict, Any, Protocol
from pathlib import Path
import mimetypes

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel

from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    PyPDFLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader,
)

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.auth import get_verified_user

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


class IngestionResult(BaseModel):
    """Result of file ingestion processing."""

    content: str
    file_type: str
    metadata: Optional[Dict[str, Any]] = None


class Ingestor(Protocol):
    """Protocol for file ingestors."""

    def can_ingest(self, mime_type: str) -> bool:
        """Check if this ingestor can handle the given MIME type."""
        ...

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        """Process the file bytes and return an IngestionResult."""
        ...


class TextIngestor:
    """Ingestor for plain text files."""

    def can_ingest(self, mime_type: str) -> bool:
        return mime_type in [
            "text/plain",
            "text/markdown",
            "text/x-python",
            "application/json",
        ]

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode="w+b", suffix=".txt", delete=False
            ) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                loader = TextLoader(temp_file_path, autodetect_encoding=True)
                docs = loader.load()
                content = docs[0].page_content if docs else ""
                return IngestionResult(content=content, file_type="text")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing text file: {str(e)}",
            )


class CSVIngestor:
    """Ingestor for CSV files."""

    def can_ingest(self, mime_type: str) -> bool:
        return mime_type == "text/csv"

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                loader = CSVLoader(temp_file_path)
                docs = loader.load()
                content = "\n".join(doc.page_content for doc in docs)
                return IngestionResult(content=content, file_type="csv")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing CSV file: {str(e)}",
            )


class ExcelIngestor:
    """Ingestor for Excel files."""

    def can_ingest(self, mime_type: str) -> bool:
        return mime_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                loader = UnstructuredExcelLoader(temp_file_path)
                docs = loader.load()
                content = "\n".join(doc.page_content for doc in docs)
                return IngestionResult(content=content, file_type="excel")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing Excel file: {str(e)}",
            )


class PDFIngestor:
    """Ingestor for PDF files."""

    def can_ingest(self, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        """Extract PDF content.

        Prefers the pdfplumber-based extractor (tables + text) used by `embedPDF`.
        Falls back to `PyPDFLoader` if the extractor fails or is unavailable.
        """
        from open_webui.utils.pdf_tabular_data_extractor import PDF_MAX_FILE_SIZE, PDF_MAX_FILE_SIZE_MB
        from pypdf import PdfReader

        file_size = len(file_bytes)
        if file_size > PDF_MAX_FILE_SIZE:
            raise ValueError(
                f"PDF '{filename}' is {file_size / 1024 / 1024:.1f}MB, "
                f"exceeding the {PDF_MAX_FILE_SIZE_MB}MB limit."
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        try:
            # Check page count to decide extraction strategy
            reader = PdfReader(temp_file_path)
            page_count = len(reader.pages)
            del reader

            if page_count > 100:
                # Large PDF — use PyMuPDF (fitz) for faster extraction with table support
                log.info(
                    "PDF has %s pages (>100); using PyMuPDF extractor for ingestion: %s",
                    page_count, filename,
                )
                try:
                    from open_webui.utils.pdf_tabular_data_extractor import extract_pdf_with_langchain_loader
                    with open(temp_file_path, "rb") as f:
                        extracted = extract_pdf_with_langchain_loader(f, pdf_name=filename)

                    table_elements = extracted.get("table_elements") or []
                    text_elements = extracted.get("text_elements") or []

                    table_texts = [getattr(e, "text", str(e)) for e in table_elements]
                    text_texts = [getattr(e, "text", str(e)) for e in text_elements]

                    content_parts: list[str] = []
                    if table_texts:
                        content_parts.append("# Extracted Tables\n\n" + "\n\n".join(table_texts))
                    if text_texts:
                        content_parts.append("# Extracted Text\n\n" + "\n\n".join(text_texts))

                    content = "\n\n".join(content_parts).strip()

                    log.info("Extracted %s tables and %s text chunks from PDF via PyMuPDF.",
                             len(table_texts), len(text_texts))

                    return IngestionResult(
                        content=content,
                        file_type="pdf",
                        metadata={
                            "extractor": "pymupdf",
                            "pages": page_count,
                            "tables_extracted": len(table_texts),
                            "text_chunks_extracted": len(text_texts),
                        },
                    )
                except Exception as extraction_error:
                    log.warning(
                        "PyMuPDF extraction failed for large PDF; falling back to PyPDFLoader: %s",
                        extraction_error,
                    )
                    loader = PyPDFLoader(temp_file_path, extract_images=True)
                    docs = loader.load()
                    content = "\n".join(doc.page_content for doc in docs)
                    log.info("Extracted %s pages from PDF via PyPDFLoader.", len(docs))
                    return IngestionResult(
                        content=content,
                        file_type="pdf",
                        metadata={"extractor": "pypdf", "pages": page_count},
                    )

            # --- Preferred path for <=100 pages: pdfplumber (tables + text) ---
            try:
                from open_webui.utils.pdf_tabular_data_extractor import embedPDF
                log.info("Using pdfplumber-based extractor for PDF ingestion.")
                with open(temp_file_path, "rb") as f:
                    extracted = embedPDF(f, pdf_name=filename)

                table_elements = extracted.get("table_elements") or []
                text_elements = extracted.get("text_elements") or []

                # Normalize into strings
                table_texts = [getattr(e, "text", str(e)) for e in table_elements]
                text_texts = [getattr(e, "text", str(e)) for e in text_elements]

                # For immediate embedding, return a single content blob.
                # Put tables first so they are easy to discover when users ask for data.
                content_parts: list[str] = []
                if table_texts:
                    content_parts.append("# Extracted Tables\n\n" + "\n\n".join(table_texts))
                if text_texts:
                    content_parts.append("# Extracted Text\n\n" + "\n\n".join(text_texts))

                content = "\n\n".join(content_parts).strip()

                log.info("Extracted %s tables and %s text chunks from PDF.",
                         len(table_texts), len(text_texts))

                return IngestionResult(
                    content=content,
                    file_type="pdf",
                    metadata={
                        "extractor": "pdfplumber",
                        "tables_extracted": extracted.get("tables_embedded", len(table_texts)),
                        "text_chunks_extracted": extracted.get("text_chunks_embedded", len(text_texts)),
                    },
                )

            except Exception as extraction_error:
                log.warning(
                    "pdfplumber table extraction failed; falling back to PyPDFLoader: %s",
                    extraction_error,
                )

            # --- Fallback path: PyPDFLoader text extraction only
            loader = PyPDFLoader(temp_file_path, extract_images=True)
            docs = loader.load()
            content = "\n".join(doc.page_content for doc in docs)
            log.info("Extracted %s pages from PDF.", len(docs))
            return IngestionResult(
                content=content,
                file_type="pdf",
                metadata={"extractor": "pypdf"},
            )

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass


class DOCXIngestor:
    """Ingestor for Microsoft Word documents (.docx only)."""

    def can_ingest(self, mime_type: str) -> bool:
        return (
            mime_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                loader = Docx2txtLoader(temp_file_path)
                docs = loader.load()
                content = docs[0].page_content if docs else ""
                return IngestionResult(content=content, file_type="docx")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing DOCX file: {str(e)}",
            )


class PPTIngestor:
    """Ingestor for Microsoft PowerPoint documents."""

    def can_ingest(self, mime_type: str) -> bool:
        return mime_type in [
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ]

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                loader = UnstructuredPowerPointLoader(temp_file_path)
                docs = loader.load()
                content = "\n".join(doc.page_content for doc in docs)
                return IngestionResult(content=content, file_type="pptx")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing PPTX file: {str(e)}",
            )


class ImageIngestor:
    """Ingestor for image files using OCR."""

    def can_ingest(self, mime_type: str) -> bool:
        return mime_type.startswith("image/")

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        try:
            import pytesseract
            from PIL import Image
            import io

            # Create a BytesIO object from the bytes
            buffer = io.BytesIO(file_bytes)
            image = Image.open(buffer)
            content = pytesseract.image_to_string(image)
            return IngestionResult(
                content=content,
                file_type="image",
                metadata={
                    "image_size": image.size,
                    "image_mode": image.mode,
                },
            )
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OCR libraries (pytesseract, PIL) not available",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing image file: {str(e)}",
            )


class VideoIngestor:
    """Ingestor for video files (basic metadata only)."""

    def can_ingest(self, mime_type: str) -> bool:
        return mime_type.startswith("video/")

    def ingest(self, file_bytes: bytes, filename: str) -> IngestionResult:
        try:
            import cv2
            import tempfile
            import os

            # Create temporary file for OpenCV (it requires file path)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name

            try:
                cap = cv2.VideoCapture(temp_file_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()

                content = f"Video file detected. Duration: {duration:.2f} seconds, Frames: {frame_count}, FPS: {fps:.2f}"
                return IngestionResult(
                    content=content,
                    file_type="video",
                    metadata={
                        "duration": duration,
                        "frame_count": frame_count,
                        "fps": fps,
                    },
                )
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except OSError:
                    pass

        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="OpenCV library not available for video processing",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error processing video file: {str(e)}",
            )


class IngestionService:
    """Service for handling file ingestion with multiple ingestors."""

    def __init__(self):
        self._ingestors: list[Ingestor] = [
            TextIngestor(),
            CSVIngestor(),
            ExcelIngestor(),
            PDFIngestor(),
            DOCXIngestor(),
            PPTIngestor(),
            ImageIngestor(),
            VideoIngestor(),
        ]

    def get_ingestor(self, mime_type: str) -> Optional[Ingestor]:
        """Get the appropriate ingestor for a MIME type."""
        for ingestor in self._ingestors:
            if ingestor.can_ingest(mime_type):
                return ingestor
        return None

    def ingest_file(
        self, file_bytes: bytes, mime_type: str, filename: str = ""
    ) -> IngestionResult:
        """Ingest a file using the appropriate ingestor."""
        ingestor = self.get_ingestor(mime_type)
        if not ingestor:
            return IngestionResult(
                content=f"Unsupported file type: {mime_type}",
                file_type="unknown",
            )
        return ingestor.ingest(file_bytes, filename)


# Global service instance
ingestion_service = IngestionService()


# API Router
router = APIRouter()


class IngestionResponse(BaseModel):
    """Response model for file ingestion."""

    content: str
    file_type: str
    metadata: Optional[Dict[str, Any]] = None


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
):
    """
    Ingest a file and extract its content for immediate embedding.
    Supports: text, CSV, Excel, PDF, DOCX, PPTX, images (OCR), videos (metadata)
    """
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must have a filename",
            )

        # Get MIME type
        mime_type, _ = mimetypes.guess_type(file.filename)
        if not mime_type:
            # Fallback: try to determine from file content
            file.file.seek(0)
            sample = file.file.read(1024)
            file.file.seek(0)

            if sample.startswith(b"%PDF"):
                mime_type = "application/pdf"
            elif sample.startswith(b"PK\x03\x04"):  # ZIP signature (used by DOCX, XLSX)
                if file.filename.lower().endswith(".docx"):
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif file.filename.lower().endswith(".xlsx"):
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    mime_type = "application/zip"
            else:
                mime_type = "application/octet-stream"

        # Read file content directly
        file_content = await file.read()

        # Run CPU-heavy ingestion in a thread pool to avoid blocking the event loop
        result = await asyncio.to_thread(
            ingestion_service.ingest_file, file_content, mime_type, file.filename
        )

        return IngestionResponse(
            content=result.content,
            file_type=result.file_type,
            metadata=result.metadata,
        )

    except Exception as e:
        log.exception(f"Error ingesting file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}",
        )
