import asyncio
import io
import logging
import re
import uuid
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from open_webui.internal.vector_store import afrom_documents as vector_store_afrom_documents
from langchain.schema import Document

from open_webui.models.files import FileForm, Files
from open_webui.models.ai_agents.rag import rag_model
from open_webui.models.ai_agents.files import files_model
from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import (
    LLM_STUDIO_CONFIG,
    MONGODB_CONFIG,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
    RAG_EMBEDDING_CONFIG,
    RAG_OPENAI_API_KEY,
    RAG_OLLAMA_BASE_URL,
    ENABLE_PDF_OCR,
    VLLM_CONFIG,
    BEDROCK_CONFIG,
)
from open_webui.internal.mongo_db import mongodb_client
from open_webui.utils.pdf_tabular_data_extractor import (
    embedPDF,
)  # Import the embedPDF function
from open_webui.utils.vision.vision import create_ocr_processor  # Import vision OCR
from open_webui.storage.provider import Storage
from open_webui.config import STORAGE_PROVIDER

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAG", logging.INFO))

# Configuration from MongoDB config
DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")
RAG_COLLECTION_NAME = "rag"
NEW_RAG_COLLECTION_NAME = "new_rag"
OLLAMA_URL = OLLAMA_BASE_URL
OLLAMA_EMBEDDING_MODEL = RAG_EMBEDDING_CONFIG.get("embeddingModel", "llama3.2")
CHUNK_SIZE = int(RAG_EMBEDDING_CONFIG.get("chunkSize", "512"))
CHUNK_OVERLAP = int(RAG_EMBEDDING_CONFIG.get("chunkOverlap", "150"))
BEDROCK_EMBEDDING_MODEL = RAG_EMBEDDING_CONFIG.get(
    "bedrockEmbeddingModel", "amazon.titan-embed-text-v1"
)

LMSTUDIO_EMBEDDING_MODEL = LLM_STUDIO_CONFIG.get(
    "embeddingModel", "text-embedding-nomic-embed-text-v1.5"
)
LMSTUDIO_HOST = LLM_STUDIO_CONFIG.get("host", "")
VLLM_HOST = VLLM_CONFIG.get("host", "")
VLLM_EMBEDDING_MODEL = VLLM_CONFIG.get("embeddingModel", "llama3.2")
VLLM_API_KEY = VLLM_CONFIG.get("api_key", "")
VLLM_EMBEDDING_HOST = VLLM_CONFIG.get("embeddingHost", "") or VLLM_HOST

# Cutoff date for new collection logic (matches JavaScript)
CUTOFF_DATE = datetime.fromisoformat("2025-10-11T00:00:00+00:00")

# Generic, domain-agnostic document families. "other" is the catch-all so
# arbitrary uploaded file types are covered without code-side keyword heuristics.
# Domain-specific taxonomies (e.g. banking call_report/credit_review) belong in
# an opt-in extraction profile, NOT this global vocabulary.
DOC_FAMILIES = [
    "email",
    "report",
    "spreadsheet",
    "presentation",
    "contract",
    "form",
    "article",
    "manual",
    "code",
    "other",
]


def derive_doc_metadata(original_name: str, sample_text: str = "") -> Dict[str, Any]:
    """Derive generic, filterable metadata for ANY uploaded document at ingest.

    Domain-agnostic (Tier 2 "semantic" metadata): an LLM extracts a generic
    `doc_family`, a short `title`, an optional ISO `doc_date` (+ derived
    `doc_year`), `language`, and a small list of salient `tags`. Stored in the
    rag `data` JSONB so retrieval can narrow/boost the candidate set BEFORE
    vector ranking via data->>'doc_family' / data->>'doc_date' /
    data->>'language' / data->'tags' — improving relevance and shrinking returned
    context when many files match.

    Intentionally NOT hard-coded to any business domain: there is no fixed
    company/account vocabulary here. Domain-specific structured fields are added
    by an opt-in profile elsewhere. Returns {} when disabled
    (RAG_LLM_METADATA=false), there is no text sample, or the call fails.
    """
    import json
    import os

    meta: Dict[str, Any] = {}

    if (
        os.getenv("RAG_LLM_METADATA", "true").lower() != "true"
        or not sample_text.strip()
    ):
        return meta

    try:
        from open_webui.llm.agent import (
            get_llm_provider,
            CONFIG,
            RETRIEVER_PROVIDER,
        )
        from langchain.schema import HumanMessage

        provider = get_llm_provider(RETRIEVER_PROVIDER or "openai", CONFIG)
        llm = provider.create_retriever_model(model=None, temperature=0.0)
        prompt = (
            "Extract generic metadata for this document. Respond with ONLY a "
            "JSON object, no prose. Schema:\n"
            '{"doc_family": one of ' + json.dumps(DOC_FAMILIES) + ", "
            '"title": "<short human-readable title, else empty>", '
            '"doc_date": "<primary date as YYYY-MM-DD if clearly present, else empty>", '
            '"language": "<ISO 639-1 two-letter code, else empty>", '
            '"tags": ["<up to 6 lowercase salient keyword/topic/entity tags>"]}\n\n'
            f"Filename: {original_name}\n"
            f"Content (first part):\n{sample_text[:2000]}"
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = (getattr(resp, "content", "") or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))

            fam = str(parsed.get("doc_family", "")).strip().lower()
            if fam in DOC_FAMILIES:
                meta["doc_family"] = fam

            title = str(parsed.get("title", "")).strip()
            if title:
                meta["title"] = title[:200]

            date = str(parsed.get("doc_date", "")).strip()
            if re.fullmatch(r"(?:19|20)\d{2}-\d{2}-\d{2}", date):
                meta["doc_date"] = date
                meta["doc_year"] = date[:4]  # convenience: indexable year

            lang = str(parsed.get("language", "")).strip().lower()
            if re.fullmatch(r"[a-z]{2}", lang):
                meta["language"] = lang

            tags = parsed.get("tags")
            if isinstance(tags, list):
                clean = []
                for t in tags:
                    t = str(t).strip().lower()
                    if t and t not in clean:
                        clean.append(t)
                if clean:
                    meta["tags"] = clean[:6]
    except Exception as e:
        log.warning(f"LLM doc-metadata extraction failed for {original_name}: {e}")

    return meta


def _strip_null_bytes(value: Any) -> Any:
    """Recursively remove NUL (``0x00``) bytes from strings.

    Extracted document text (esp. from OCR / malformed PDFs) can contain NUL
    characters. Postgres text/JSONB columns reject them with
    ``invalid byte sequence for encoding "UTF8": 0x00``, which fails the whole
    upload. NUL carries no meaning in our content, so we drop it before insert.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_strip_null_bytes(v) for v in value]
    if isinstance(value, dict):
        return {k: _strip_null_bytes(v) for k, v in value.items()}
    return value


class FileUploadError(Exception):
    """Custom exception for file upload errors."""

    pass


class UnsupportedFileTypeError(FileUploadError):
    """Exception for unsupported file types."""

    pass


class RAGUploadService:
    """Service for uploading and processing RAG documents."""

    def __init__(self):
        self.db_name = DB_NAME
        self.rag_collection_name = RAG_COLLECTION_NAME
        self.new_rag_collection_name = NEW_RAG_COLLECTION_NAME

    def _get_embeddings_model(self):
        """Get the embeddings model via the shared provider factory, following
        LLM_EMBEDDING_PROVIDER (which inherits LLM_DEFAULT_PROVIDER).

        This keeps the ingest/write path in sync with the query/read path (see
        routers/rag.py), so uploaded documents and query vectors always use the
        same embedding model. Replaces the old RAG_EMBEDDING_TYPE if/else."""
        from open_webui.llm.agent import get_llm_provider, EMBEDDING_PROVIDER, CONFIG

        provider_name = EMBEDDING_PROVIDER or "openai"
        log.info(f"Creating embedding model via provider '{provider_name}'")
        return get_llm_provider(provider_name, CONFIG).create_embedding_model()

    def _get_text_splitter(self):
        """Get configured text splitter."""
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    async def _get_collection_name(
        self, assistant_id: str, organization_id: str
    ) -> str:
        """Determine which collection to use based on assistant creation date."""
        try:
            # Get assistant details from repository
            assistant_details = await self._get_assistant_details(
                organization_id, assistant_id
            )

            if assistant_details and assistant_details.get("created_at"):
                # Handle different date formats
                created_at_str = assistant_details["created_at"]
                if created_at_str.endswith("Z"):
                    created_at_str = created_at_str.replace("Z", "+00:00")

                created_at = datetime.fromisoformat(created_at_str)
                if created_at >= CUTOFF_DATE:
                    return self.new_rag_collection_name

            return self.rag_collection_name

        except Exception as e:
            log.warning(f"Could not determine collection, using default: {e}")
            return self.rag_collection_name

    async def _get_assistant_details(
        self, organization_id: str, assistant_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get assistant details from repository."""
        try:
            # This matches the JavaScript repository.assistantV2.getByAssistantIdAndOrganizationId
            query = {"organization_id": organization_id, "assistant_id": assistant_id}
            result = await mongodb_client.query_one(
                self.db_name, "assistants_v2", query  # Adjust collection name as needed
            )
            return result
        except Exception as e:
            log.warning(f"Could not get assistant details: {e}")
            return None

    async def _extract_pdf_content_with_embedder(
        self, file_buffer: bytes, filename: str, enable_ocr: bool
    ) -> Optional[list]:
        """Extract PDF content using the enhanced PDF extractor with table support.

        For PDFs with more than 200 pages the caller's PyPDFLoader fallback is
        used instead of pdfplumber to avoid excessive memory/CPU consumption.
        """
        try:
            from pypdf import PdfReader as _PdfReader

            # Quick page-count check (pypdf is lightweight, doesn't parse page content)
            reader = _PdfReader(io.BytesIO(file_buffer))
            page_count = len(reader.pages)
            del reader

            if page_count > 100:
                log.info(
                    "PDF has %s pages (>100); using PyMuPDF extractor: %s",
                    page_count, filename,
                )
                from open_webui.utils.pdf_tabular_data_extractor import (
                    extract_pdf_with_langchain_loader,
                )

                pdf_file = io.BytesIO(file_buffer)
                result = await asyncio.to_thread(
                    extract_pdf_with_langchain_loader, pdf_file, filename
                )

                if result.get("status") != "success":
                    log.warning("PyMuPDF extraction failed, falling back to PyPDFLoader")
                    return None

                docs = []
                for index, table_element in enumerate(
                    result.get("table_elements", [])
                ):
                    content = (
                        table_element.text
                        if hasattr(table_element, "text")
                        else str(table_element)
                    )
                    docs.append(
                        {
                            "pageContent": content,
                            "metadata": {
                                "id": str(uuid.uuid4()),
                                "blobType": "application/pdf",
                                "fileName": filename,
                                "fileSize": len(file_buffer),
                                "type": "table_element",
                                "elementIndex": index,
                                "extractionMethod": "pymupdf_extractor",
                                "isTable": True,
                                "source": "pdf_tabular_data_extractor",
                            },
                        }
                    )
                table_count = len(docs)
                for index, text_element in enumerate(
                    result.get("text_elements", [])
                ):
                    content = (
                        text_element.text
                        if hasattr(text_element, "text")
                        else str(text_element)
                    )
                    docs.append(
                        {
                            "pageContent": content,
                            "metadata": {
                                "id": str(uuid.uuid4()),
                                "blobType": "application/pdf",
                                "fileName": filename,
                                "fileSize": len(file_buffer),
                                "type": "text_element",
                                "elementIndex": table_count + index,
                                "extractionMethod": "pymupdf_extractor",
                                "isTable": False,
                                "source": "pdf_tabular_data_extractor",
                            },
                        }
                    )
                log.info(
                    f"PyMuPDF extracted {table_count} tables and "
                    f"{len(docs) - table_count} text chunks from {filename}"
                )
                return docs

            log.info(
                "Using enhanced PDF extractor with table detection and summarization..."
            )

            # Create a BytesIO object from the buffer
            pdf_file = io.BytesIO(file_buffer)

            # Run CPU-heavy PDF extraction in a thread pool to avoid blocking the event loop
            result = await asyncio.to_thread(embedPDF, pdf_file, filename)

            if result.get("status") != "success":
                log.error("PDF extraction failed")
                return None

            log.info(
                f"PDF extraction successful: {result.get('tables_embedded', 0)} tables, {result.get('text_chunks_embedded', 0)} text chunks"
            )

            docs = []

            # Process table elements
            table_elements = result.get("table_elements", [])
            if table_elements:
                for index, table_element in enumerate(table_elements):
                    # Extract content from Element object
                    content = (
                        table_element.text
                        if hasattr(table_element, "text")
                        else str(table_element)
                    )

                    docs.append(
                        {
                            "pageContent": content,
                            "metadata": {
                                "id": str(uuid.uuid4()),
                                "blobType": "application/pdf",
                                "fileName": filename,
                                "fileSize": len(file_buffer),
                                "type": "table_element",
                                "elementIndex": index,
                                "extractionMethod": "enhanced_pdf_extractor",
                                "isTable": True,
                                "source": "pdf_tabular_data_extractor",
                            },
                        }
                    )

            # Process text elements
            text_elements = result.get("text_elements", [])
            if text_elements:
                for index, text_element in enumerate(text_elements):
                    # Extract content from Element object
                    content = (
                        text_element.text
                        if hasattr(text_element, "text")
                        else str(text_element)
                    )

                    docs.append(
                        {
                            "pageContent": content,
                            "metadata": {
                                "id": str(uuid.uuid4()),
                                "blobType": "application/pdf",
                                "fileName": filename,
                                "fileSize": len(file_buffer),
                                "type": "text_element",
                                "elementIndex": index
                                + len(
                                    table_elements
                                ),  # Continue numbering after tables
                                "extractionMethod": "enhanced_pdf_extractor",
                                "isTable": False,
                                "source": "pdf_tabular_data_extractor",
                            },
                        }
                    )

            log.info(
                f"Processed {len(table_elements)} table elements and {len(text_elements)} text elements"
            )

            # Add OCR processing for images in PDF
            if enable_ocr:
                try:
                    log.info("Processing PDF with vision OCR for image elements...")

                    # Determine which OCR provider to use based on available API keys
                    ocr_processor = None

                    # Use OpenAI for OCR if API key is available
                    ocr_processor = create_ocr_processor(
                        provider_name="openai",
                        api_key=OPENAI_API_KEY,
                        model="gpt-4o",  # Use cost-effective model for OCR
                    )
                    log.info("Using OpenAI for OCR processing")

                    if ocr_processor:
                        # Process PDF from buffer for OCR
                        ocr_results = await ocr_processor.process_pdf_from_buffer(
                            file_buffer
                        )

                        if ocr_results and ocr_results.success:
                            ocr_pages = ocr_results.pages
                            log.info(
                                f"OCR successfully processed {len(ocr_pages)} image pages"
                            )

                            # Add OCR results to docs
                            for index, ocr_page in enumerate(ocr_pages):
                                page_number = ocr_page.page_number
                                extracted_text = ocr_page.extracted_text
                                log.info(
                                    f"OCR Page {page_number} extracted text length: {extracted_text}"
                                )

                                if (
                                    extracted_text.strip()
                                ):  # Only add non-empty extractions
                                    docs.append(
                                        {
                                            "pageContent": extracted_text,
                                            "metadata": {
                                                "id": str(uuid.uuid4()),
                                                "blobType": "application/pdf",
                                                "fileName": filename,
                                                "fileSize": len(file_buffer),
                                                "type": "image_ocr_element",
                                                "elementIndex": index
                                                + len(table_elements)
                                                + len(text_elements),
                                                "pageNumber": page_number,
                                                "extractionMethod": "vision_ocr",
                                                "isTable": False,
                                                "isOCR": True,
                                                "confidence": ocr_page.confidence,
                                                "source": "pdf_vision_ocr_extractor",
                                            },
                                        }
                                    )

                            log.info(
                                f"Added {len([p for p in ocr_pages if p.extracted_text.strip()])} OCR image elements to documents"
                            )
                        else:
                            error_msg = (
                                ocr_results.error_message
                                if ocr_results
                                else "Unknown error"
                            )
                            log.warning(f"OCR processing failed: {error_msg}")

                except Exception as ocr_error:
                    log.error(f"OCR processing failed: {ocr_error}")
                    # Continue with existing documents even if OCR fails
                    import traceback

                    traceback.print_exc()
            else:
                log.info("PDF OCR processing is disabled")

            return docs

        except Exception as e:
            log.error(f"Enhanced PDF extraction failed: {e}")
            import traceback

            traceback.print_exc()
            return None

    async def _load_document(
        self, file_buffer: bytes, mimetype: str, filename: str, enable_ocr: bool
    ) -> list:
        """Load document based on file type using langchain_community loaders."""
        docs = []

        try:
            if mimetype == "text/csv":
                # Create temporary file for CSV with proper cleanup
                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".csv", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                # Load the CSV outside the context manager to ensure file is closed
                try:
                    loader = CSVLoader(tmp_file_path)
                    docs = loader.load()
                finally:
                    # Clean up temp file after loading is complete
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")

            elif mimetype == "application/pdf":
                # Use the enhanced PDF extractor with table support
                enhanced_docs = await self._extract_pdf_content_with_embedder(
                    file_buffer, filename, enable_ocr
                )
                if enhanced_docs:
                    log.info(f"Enhanced PDF extraction successful for {filename}")
                    return enhanced_docs

                # Fallback to standard PyPDFLoader if enhanced extraction fails
                log.info(
                    "Enhanced PDF extractor failed, attempting fallback to PyPDFLoader"
                )

                from langchain_community.document_loaders import PyPDFLoader

                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".pdf", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                finally:
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")

                # Add consistent metadata structure (fallback)
                processed_docs = []
                for index, doc in enumerate(docs):
                    processed_docs.append(
                        {
                            "pageContent": doc.page_content,
                            "metadata": {
                                **doc.metadata,
                                "id": str(uuid.uuid4()),
                                "blobType": "application/pdf",
                                "fileName": filename,
                                "fileSize": len(file_buffer),
                                "type": "pdf_fallback",
                                "extractionMethod": "pdfloader_fallback",
                                "chunkIndex": index,
                            },
                        }
                    )

                log.info(f"PDF processed using fallback PyPDFLoader: {filename}")
                return processed_docs

            elif mimetype == "application/json":
                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".json", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    loader = JSONLoader(
                        tmp_file_path, jq_schema=".", text_content=False
                    )
                    docs = loader.load()
                finally:
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")

            elif (
                mimetype
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".docx", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    loader = Docx2txtLoader(tmp_file_path)
                    docs = loader.load()
                finally:
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")

            elif mimetype == "text/plain":
                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".txt", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    loader = TextLoader(tmp_file_path)
                    docs = loader.load()
                finally:
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")

            elif (
                mimetype
                == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            ):
                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".pptx", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    loader = UnstructuredPowerPointLoader(tmp_file_path)
                    docs = loader.load()
                finally:
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")
            elif (
                mimetype
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".xlsx", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    loader = UnstructuredExcelLoader(tmp_file_path)
                    docs = loader.load()
                finally:
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")
            elif mimetype == "application/vnd.ms-excel":
                with tempfile.NamedTemporaryFile(
                    mode="w+b", suffix=".xls", delete=False
                ) as tmp_file:
                    tmp_file.write(file_buffer)
                    tmp_file.flush()
                    tmp_file_path = tmp_file.name

                try:
                    loader = UnstructuredExcelLoader(tmp_file_path)
                    docs = loader.load()
                finally:
                    try:
                        Path(tmp_file_path).unlink()
                    except (OSError, PermissionError) as e:
                        log.warning(f"Could not delete temp file {tmp_file_path}: {e}")
            else:
                log.error(f"{mimetype} is not supported")
                raise UnsupportedFileTypeError(f"File type {mimetype} is not supported")

            # Convert to standard format if not already done
            if docs and not isinstance(docs[0], dict):
                return [
                    {"pageContent": doc.page_content, "metadata": doc.metadata}
                    for doc in docs
                ]

            return docs

        except Exception as e:
            if isinstance(e, UnsupportedFileTypeError):
                raise
            log.error(f"Error loading document: {e}")
            raise FileUploadError("Failed to load files")

    @staticmethod
    def _build_storage_filename(
        organization_id: str, file_id: str, filename: str, max_bytes: int = 200
    ) -> str:
        """Build a filesystem-safe, length-bounded storage key.

        Handles (double) URL-encoded names and strips path separators, then
        truncates the name part while preserving the extension so the final
        component stays under the filesystem limit (255 bytes -> Errno 36).
        """
        import os
        from urllib.parse import unquote

        # Decode up to two layers of URL-encoding (e.g. "%2520" -> "%20" -> " ").
        decoded = filename or ""
        for _ in range(2):
            new_decoded = unquote(decoded)
            if new_decoded == decoded:
                break
            decoded = new_decoded

        # Keep only the base name; never let separators create sub-paths.
        decoded = decoded.replace("\\", "/").split("/")[-1].strip() or "file"

        prefix = f"{organization_id}_{file_id}_"
        stem, ext = os.path.splitext(decoded)

        # Cap the extension defensively (a malformed name can have a huge "ext").
        if len(ext.encode("utf-8")) > 32:
            ext = ""

        # Budget for the stem in bytes (UTF-8), reserving room for prefix + ext.
        budget = max_bytes - len(prefix.encode("utf-8")) - len(ext.encode("utf-8"))
        if budget < 1:
            budget = 1

        stem_bytes = stem.encode("utf-8")[:budget]
        # Avoid splitting a multi-byte UTF-8 character at the truncation boundary.
        safe_stem = stem_bytes.decode("utf-8", errors="ignore") or "file"

        return f"{prefix}{safe_stem}{ext}"

    async def _save_rag_file(
        self,
        file_id: str,
        mimetype: str,
        organization_id: str,
        file_buffer: bytes,
        filename: str,
    ) -> str:
        """Save RAG file using configured Storage provider and return URL."""
        try:
            # Create a BytesIO object from the buffer for the Storage provider
            file_like_object = io.BytesIO(file_buffer)

            # Create a structured path for better organization.
            # Filenames may be (double) URL-encoded and arbitrarily long, which can
            # exceed the filesystem's per-component limit (255 bytes -> Errno 36).
            # Sanitize and bound the length while preserving the extension.
            safe_filename = self._build_storage_filename(
                organization_id, file_id, filename
            )
            unique_filename = safe_filename

            log.info(
                f"Uploading file using {STORAGE_PROVIDER} Storage provider: {unique_filename}"
            )
            log.info(f"File size: {len(file_buffer)} bytes")

            # Upload file using the configured Storage provider
            # The Storage provider handles local, S3, GCS, or Azure based on STORAGE_PROVIDER config
            _, file_path = Storage.upload_file(file_like_object, unique_filename)
            # Verify upload was successful
            if not file_path:
                raise FileUploadError("Storage provider returned empty file path")

            log.info(f"File uploaded successfully to {STORAGE_PROVIDER}: {file_path}")

            return file_path

        except Exception as e:
            log.error(f"Error saving RAG file using Storage provider: {e}")
            import traceback

            traceback.print_exc()
            log.error(f"Failed to save file to {STORAGE_PROVIDER}: {e}")
            raise FileUploadError("Failed to save file")

    async def upload(
        self,
        organization_id: str = "",
        assistant_id: str = "",
        uploaded_file: Dict[str, Any] = None,
        boarditem_id: str = "",
        board_id: str = "",
        folder_id: str = "",
        enable_ocr: bool = False,
        no_split: bool = False,
        file_id: str = "",
    ) -> Dict[str, Any]:
        """
        Upload and process a file for RAG.

        Args:
            organization_id: Organization ID
            assistant_id: Assistant ID
            uploaded_file: Dict containing file data with keys:
                - buffer: bytes - File content
                - mimetype: str - MIME type
                - originalname: str - Original filename
                - size: int - File size in bytes
            boarditem_id: Board item ID (optional)
            board_id: Board ID (optional)

        Returns:
            Dict with file information including id, bytes, filename, etc.
        """
        try:
            if not uploaded_file:
                raise ValueError("No file provided")

            file_buffer = uploaded_file.get("buffer")
            mimetype = uploaded_file.get("mimetype")
            original_name = uploaded_file.get("originalname")
            file_size = uploaded_file.get("size", 0)

            if not all([file_buffer, mimetype, original_name]):
                raise ValueError("Missing required file information")

            # Determine collection name based on assistant creation date
            # collection_name = await self._get_collection_name(assistant_id, organization_id)

            # Get MongoDB collection for vector store (MongoDB mode only)
            from open_webui.internal.document_store import DB_TYPE as _DOC_DB_TYPE
            if _DOC_DB_TYPE == "postgresql":
                rag_collection = None
            else:
                log.info(f"Connecting to MongoDB collection: {mongodb_client._connected}")
                db_client = await mongodb_client.get_client()
                db = db_client[self.db_name]
                rag_collection = db[RAG_COLLECTION_NAME]

            # Get embeddings model
            embeddings = self._get_embeddings_model()

            # Get text splitter
            text_splitter = self._get_text_splitter()

            # Use provided file_id (e.g. data_board file _id) or generate one
            file_id = file_id or str(uuid.uuid4())
            created_at = datetime.utcnow().isoformat()

            # Load and process document
            log.info(f"Loading document: {original_name} ({mimetype})")
            docs = await self._load_document(
                file_buffer, mimetype, original_name, enable_ocr
            )

            if not docs:
                raise FileUploadError("No content extracted from file")

            # Save file and get URL (matching JavaScript fileManager.saveRAGFile)
            uploaded_file_url = await self._save_rag_file(
                file_id, mimetype, organization_id, file_buffer, original_name
            )

            # Split documents into chunks (unless no_split is True)
            if no_split:
                log.info("Skipping document splitting (no_split=True)")
                split_docs = [
                    Document(page_content=doc["pageContent"], metadata=doc["metadata"])
                    for doc in docs
                ]
            else:
                log.info("Splitting documents into chunks...")

                # Separate table elements from text elements
                table_docs = [
                    Document(page_content=doc["pageContent"], metadata=doc["metadata"])
                    for doc in docs
                    if doc.get("metadata", {}).get("isTable")
                ]
                text_docs = [
                    Document(page_content=doc["pageContent"], metadata=doc["metadata"])
                    for doc in docs
                    if not doc.get("metadata", {}).get("isTable")
                ]

                # Only split text documents; keep table documents intact
                split_text_docs = text_splitter.split_documents(text_docs) if text_docs else []
                split_docs = split_text_docs + table_docs

            # Derive generic, domain-agnostic metadata (see derive_doc_metadata):
            # an LLM extracts doc_family / title / doc_date / language / tags over
            # the first extracted chunk. Flows into the `data` JSONB and is
            # filterable via data->>'doc_family' / data->>'doc_date' /
            # data->>'language' / data->'tags'.
            # Run in a thread: derive_doc_metadata makes a blocking llm.invoke
            # call, and upload() runs on the shared event loop. Calling it
            # directly freezes the loop for the whole (retrying) upstream call,
            # hanging every other endpoint until it returns.
            doc_meta = await asyncio.to_thread(
                derive_doc_metadata,
                original_name,
                split_docs[0].page_content if split_docs else "",
            )
            log.info(f"Derived doc metadata for {original_name}: {doc_meta}")

            # Tier 1 universal (deterministic, no LLM) — applies to ANY file type.
            _file_ext = (
                original_name.rsplit(".", 1)[-1].lower()
                if "." in original_name
                else ""
            )

            # Add metadata to each chunk (matching JavaScript logic)
            csv_all_splits = []
            _last_page = None  # carry-forward for chunks split mid-page
            for doc in split_docs:
                # Clean metadata by removing 'id' and '_id' keys to avoid conflicts
                clean_metadata = {
                    k: v for k, v in doc.metadata.items() if k not in ["id", "_id"]
                }

                # Page-level metadata: every extracted chunk is prefixed with
                # "Page N, ..." (see pdf_tabular_data_extractor). Capture N as
                # structured, filterable metadata (data->>'page') so retrieval can be
                # scoped/cited at page granularity. Prefer an explicit element page if
                # propagated; else parse the text prefix with carry-forward.
                _page_match = re.search(r"Page\s+(\d+)", doc.page_content[:40])
                if _page_match:
                    _last_page = _page_match.group(1)
                _chunk_page = (
                    doc.metadata.get("page")
                    if doc.metadata.get("page") is not None
                    else _last_page
                )

                processed_doc = Document(
                    page_content=_strip_null_bytes(doc.page_content),
                    metadata=_strip_null_bytes({
                        **clean_metadata,
                        **doc_meta,
                        **({"page": str(_chunk_page)} if _chunk_page is not None else {}),
                        # Tier 1 chunk-level universal metadata (deterministic).
                        "content_kind": "table" if doc.metadata.get("isTable") else "prose",
                        **({"file_ext": _file_ext} if _file_ext else {}),
                        "org_id": organization_id,
                        "original_name": original_name,
                        "assistant_id": assistant_id,
                        "file_id": file_id,
                        "bytes": file_size,
                        "created_at": created_at,
                        "board_id": board_id,
                        "boarditem_id": boarditem_id,
                        "folder_id": folder_id,
                        "url": uploaded_file_url,
                        "key": boarditem_id or file_id,
                    }),
                )
                csv_all_splits.append(processed_doc)

            # Create vector store and add documents (matching JavaScript MongoDBAtlasVectorSearch.fromDocuments)
            log.info(f"Adding {len(csv_all_splits)} chunks to vector store...")

            await vector_store_afrom_documents(
                csv_all_splits,
                embeddings,
                collection=rag_collection,
                index_name="vector_index",
                text_key="text",
                embedding_key="embedding",
            )

            # Get model name for logging (matching JavaScript embeddings?.modelName)
            model_name = getattr(embeddings, "model", "unknown")
            log.info(
                f"Finished uploading file to vector store using {model_name}: {original_name}"
            )

            # Save file details
            if not board_id or not boarditem_id:
                await files_model.create(
                    {
                        "id": file_id,
                        "organization_id": organization_id,
                        "assistant_id": assistant_id,
                        "filename": original_name,
                        "file_size": file_size,
                        "file_type": mimetype,
                        "board_id": board_id,
                        "boarditem_id": boarditem_id,
                        "url": uploaded_file_url,
                    }
                )

            # Return response matching JavaScript structure
            return {
                "id": file_id,
                "bytes": file_size,
                "filename": original_name,
                "assistant_id": assistant_id,
                "created_at": created_at,
                "board_id": board_id,
                "boarditem_id": boarditem_id,
                "url": uploaded_file_url,
                "extraction_method": (
                    "enhanced_pdf_extractor_with_ocr"
                    if mimetype == "application/pdf" and ENABLE_PDF_OCR.value
                    else (
                        "enhanced_pdf_extractor"
                        if mimetype == "application/pdf"
                        else "standard"
                    )
                ),
                "tables_extracted": (
                    len(
                        [
                            doc
                            for doc in docs
                            if doc.get("metadata", {}).get("isTable", False)
                        ]
                    )
                    if mimetype == "application/pdf"
                    else 0
                ),
                "text_chunks_extracted": (
                    len(
                        [
                            doc
                            for doc in docs
                            if not doc.get("metadata", {}).get("isTable", False)
                            and not doc.get("metadata", {}).get("isOCR", False)
                        ]
                    )
                    if mimetype == "application/pdf"
                    else len(docs)
                ),
                "ocr_images_extracted": (
                    len(
                        [
                            doc
                            for doc in docs
                            if doc.get("metadata", {}).get("isOCR", False)
                        ]
                    )
                    if mimetype == "application/pdf"
                    else 0
                ),
            }

        except Exception as error:
            log.error(f"Upload error: {error}")
            raise FileUploadError("Upload failed")

    async def remove(
        self, file_id: str = "", organization_id: str = ""
    ) -> Dict[str, Any]:
        """
        Remove a RAG file from both regular and new RAG collections.

        Args:
            file_id: The file ID to remove
            organization_id: The organization ID

        Returns:
            Dict containing removal results from both collections
        """
        try:

            log.info(
                f"Removing RAG file: {file_id} for organization: {organization_id}"
            )

            # Remove from regular RAG collection (matches JavaScript repository.rag.remove)
            regular_result = await rag_model.remove(file_id, organization_id)

            # Remove from new RAG collection (matches JavaScript repository.rag.removeNewRag)
            new_rag_result = await rag_model.remove_new_rag(file_id, organization_id)

            await files_model.hard_delete(file_id)

            # Log removal results
            regular_count = regular_result.get("deleted_count", 0)
            new_rag_count = new_rag_result.get("deleted_count", 0)
            total_removed = regular_count + new_rag_count

            log.info(f"Removed {regular_count} documents from regular RAG collection")
            log.info(f"Removed {new_rag_count} documents from new RAG collection")
            log.info(f"Total documents removed: {total_removed}")

            # Return the regular result (matches JavaScript behavior)
            return {
                "deleted_count": total_removed,
                "regular_rag_deleted": regular_count,
                "new_rag_deleted": new_rag_count,
                "file_id": file_id,
                "organization_id": organization_id,
            }

        except Exception as error:
            log.error(f"Error removing RAG file {file_id}: {error}")
            raise FileUploadError("Failed to remove file")

    async def getByAssistantId(
        self, organization_id: str, assistant_id: str
    ) -> Dict[str, Any]:
        """
        Get RAG files by assistant ID.

        Args:
            organization_id: Organization ID
            assistant_id: Assistant ID

        Returns:
            Dict with file information
        """
        try:
            # This matches the JavaScript repository.rag.getByAssistantId
            query = {
                "organization_id": organization_id,
                "assistant_id": assistant_id,
                "deleted_at": None,
            }
            result = await files_model.search(query)
            return result or []
        except Exception as e:
            log.error(f"Error getting RAG files by assistant ID: {e}")
            raise FileUploadError("Failed to get files")

    async def getById(self, file_id: str, organization_id: str) -> Dict[str, Any]:
        """
        Get RAG file by ID.

        Args:
            file_id: The file ID to retrieve
            organization_id: The organization ID

        Returns:
            Dict with file information
        """
        try:

            result = await files_model.get_by_id(file_id=file_id)
            return result or {}
        except Exception as e:
            log.error(f"Error getting RAG file by ID {file_id}: {e}")
            raise FileUploadError("Failed to get file")

    async def getFileContent(
        self,
        file_id: str,
    ) -> Dict[str, Any]:
        """
        Get the content of a RAG file by ID.

        Args:
            file_id: The file ID to retrieve content for

        Returns:
            Dict with file content and metadata
        """
        try:
            # Get file details from files_model
            file_details = await files_model.get_by_id(file_id=file_id)

            if not file_details:
                raise FileUploadError(f"File with ID {file_id} not found")

            # Use Storage provider to get the file content
            file_path = Storage.get_file(file_details["url"])

            result = {
                **(file_details or {}),  # Handle None case
                "file_path": file_path,
            }

            return result

        except Exception as e:
            log.error(f"Error getting RAG file content {file_id}: {e}")
            raise FileUploadError("Failed to get file content")


# Create global instance (following rag.py pattern)
rag_upload_service = RAGUploadService()


# Export convenience function following the pattern in rag.py
async def upload(
    organization_id: str = "",
    assistant_id: str = "",
    uploaded_file: Dict[str, Any] = None,
    boarditem_id: str = "",
    board_id: str = "",
    folder_id: str = "",
    file_id: str = "",
    enable_ocr: bool = False,
    no_split: bool = False,
) -> Dict[str, Any]:
    """
    Upload and process a file for RAG.

    Convenience function that wraps the service method.
    """
    return await rag_upload_service.upload(
        organization_id=organization_id,
        assistant_id=assistant_id,
        uploaded_file=uploaded_file,
        boarditem_id=boarditem_id,
        board_id=board_id,
        folder_id=folder_id,
        file_id=file_id,
        enable_ocr=enable_ocr,
        no_split=no_split,
    )


async def remove(file_id: str = "", organization_id: str = "") -> Dict[str, Any]:
    """
    Remove a RAG file from both regular and new RAG collections.

    Convenience function that wraps the service method.
    """
    log.info(
        f"Removing RAG file with ID: {file_id} for organization: {organization_id}"
    )
    return await rag_upload_service.remove(
        file_id=file_id, organization_id=organization_id
    )


async def getByAssistantId(organization_id: str, assistant_id: str) -> Dict[str, Any]:
    """
    Get RAG files by assistant ID.

    Convenience function that wraps the service method.
    """
    log.info(
        f"Getting RAG files for assistant ID: {assistant_id} in organization: {organization_id}"
    )
    return await rag_upload_service.getByAssistantId(
        organization_id=organization_id, assistant_id=assistant_id
    )


async def getById(file_id: str, organization_id: str) -> Dict[str, Any]:
    """
    Get RAG file by ID.

    Convenience function that wraps the service method.
    """
    log.info(f"Getting RAG file with ID: {file_id} for organization: {organization_id}")
    return await rag_upload_service.getById(
        file_id=file_id, organization_id=organization_id
    )


async def getFileContent(
    file_id: str,
) -> Dict[str, Any]:
    """
    Get the content of a RAG file by ID.

    Convenience function that wraps the service method.
    """
    log.info(f"Getting content for RAG file with ID: {file_id}")
    return await rag_upload_service.getFileContent(file_id=file_id)


# Export all following the pattern in rag.py
__all__ = [
    "RAGUploadService",
    "rag_upload_service",
    "upload",
    "remove",
    "getByAssistantId",
    "getById",
    "getFileContent",
    "FileUploadError",
    "UnsupportedFileTypeError",
]
