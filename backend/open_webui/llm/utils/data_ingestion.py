from abc import ABC, abstractmethod
from typing import List, Dict, Any, AsyncIterable, Optional, Union
from pathlib import Path
import logging
import hashlib
import uuid

logger = logging.getLogger(__name__)


# --- Data Models -------------------------------------------------------------


class Document:
    """
    Shared document DTO for all ingestors and indexers.
    Follows the schema: id, source_uri, text, mime_type, embedding, metadata, routing_tags
    """

    def __init__(
        self,
        id: str,
        source_uri: str,
        text: str,
        mime_type: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        routing_tags: Optional[List[str]] = None,
    ):
        self.id = id
        self.source_uri = source_uri
        self.text = text
        self.mime_type = mime_type
        self.embedding = embedding or []
        self.metadata = metadata or {}
        self.routing_tags = routing_tags or []


# --- Interfaces --------------------------------------------------------------


class IIngestor(ABC):
    """
    Interface for data source ingestion (PDF, Image, SQL, etc.)
    """

    @abstractmethod
    async def load_and_preprocess(self, source: str) -> Document:
        """
        Load and preprocess the source into a Document.
        """
        pass


class IEmbeddingService(ABC):
    """
    Interface for embedding text or images.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def embed_image(self, bytes_data: bytes) -> List[float]:
        pass


class IIndexer(ABC):
    """
    Interface for storing/upserting documents into an index.
    """

    @abstractmethod
    async def upsert(self, doc: Document) -> None:
        pass

    @abstractmethod
    async def upsert_batch(self, docs: List[Document]) -> None:
        pass


class IChunker(ABC):
    """
    Interface for streaming chunks of a large source (to avoid OOM).
    """

    @abstractmethod
    async def stream_chunks(self, source: str) -> AsyncIterable[str]:
        pass


# --- Concrete Ingestors ------------------------------------------------------


class TextIngestor(IIngestor):
    """Ingestor for plain text data."""

    def __init__(self, mime_type: str = "text/plain"):
        self.mime_type = mime_type

    async def load_and_preprocess(self, source: str) -> Document:
        try:
            logger.info(f"Processing text source: {source[:50]}...")
            doc_id = str(uuid.uuid4())
            return Document(
                id=doc_id,
                source_uri=source,
                text=source,
                mime_type=self.mime_type,
                routing_tags=["embeddable"],
            )
        except Exception as error:
            logger.error(f"Error processing text: {error}")
            raise


class PDFIngestor(IIngestor):
    """Ingestor for PDF files."""

    async def load_and_preprocess(self, source: str) -> Document:
        try:
            logger.info(f"Processing PDF: {source}")
            # TODO: Implement actual PDF text extraction using PyPDF2 or pypdf
            # For now, return placeholder
            extracted_text = f"Extracted text from PDF: {source}"
            doc_id = hashlib.md5(source.encode()).hexdigest()
            return Document(
                id=doc_id,
                source_uri=source,
                text=extracted_text,
                mime_type="application/pdf",
                routing_tags=["embeddable"],
                metadata={"file_path": source},
            )
        except Exception as error:
            logger.error(f"Error processing PDF {source}: {error}")
            raise


class DocumentIngestor(IIngestor):
    """Ingestor for Word documents."""

    async def load_and_preprocess(self, source: str) -> Document:
        try:
            logger.info(f"Processing document: {source}")
            # TODO: Implement document text extraction using python-docx
            extracted_text = f"Extracted text from document: {source}"
            doc_id = hashlib.md5(source.encode()).hexdigest()
            return Document(
                id=doc_id,
                source_uri=source,
                text=extracted_text,
                mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                routing_tags=["embeddable"],
                metadata={"file_path": source},
            )
        except Exception as error:
            logger.error(f"Error processing document {source}: {error}")
            raise


class CSVIngestor(IIngestor):
    """Ingestor for CSV files - converts to text for embedding or marks as SQL-able."""

    def __init__(self, embeddable: bool = True):
        self.embeddable = embeddable

    async def load_and_preprocess(self, source: str) -> Document:
        try:
            logger.info(f"Processing CSV: {source}")
            # TODO: Implement CSV processing using pandas
            processed_text = f"Processed CSV data from: {source}"
            doc_id = hashlib.md5(source.encode()).hexdigest()
            routing_tags = ["sqlable"]
            if self.embeddable:
                routing_tags.append("embeddable")
            return Document(
                id=doc_id,
                source_uri=source,
                text=processed_text,
                mime_type="text/csv",
                routing_tags=routing_tags,
                metadata={"file_path": source},
            )
        except Exception as error:
            logger.error(f"Error processing CSV {source}: {error}")
            raise


class ExcelIngestor(IIngestor):
    """Ingestor for Excel files."""

    def __init__(self, embeddable: bool = True):
        self.embeddable = embeddable

    async def load_and_preprocess(self, source: str) -> Document:
        try:
            logger.info(f"Processing Excel: {source}")
            # TODO: Implement Excel processing using openpyxl or pandas
            processed_text = f"Processed Excel data from: {source}"
            doc_id = hashlib.md5(source.encode()).hexdigest()
            routing_tags = ["sqlable"]
            if self.embeddable:
                routing_tags.append("embeddable")
            return Document(
                id=doc_id,
                source_uri=source,
                text=processed_text,
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                routing_tags=routing_tags,
                metadata={"file_path": source},
            )
        except Exception as error:
            logger.error(f"Error processing Excel {source}: {error}")
            raise


class ImageIngestor(IIngestor):
    """Ingestor for images using OCR."""

    async def load_and_preprocess(self, source: str) -> Document:
        try:
            logger.info(f"Processing image with OCR: {source}")
            # TODO: Implement OCR using pytesseract
            extracted_text = f"OCR extracted text from image: {source}"
            doc_id = hashlib.md5(source.encode()).hexdigest()
            return Document(
                id=doc_id,
                source_uri=source,
                text=extracted_text,
                mime_type="image/jpeg",  # Could be dynamic based on file
                routing_tags=["embeddable", "vision-processed"],
                metadata={"file_path": source},
            )
        except Exception as error:
            logger.error(f"Error processing image {source}: {error}")
            raise


class VideoIngestor(IIngestor):
    """Ingestor for videos using frame extraction + OCR."""

    async def load_and_preprocess(self, source: str) -> Document:
        try:
            logger.info(f"Processing video: {source}")
            # TODO: Implement video processing - extract frames and OCR
            extracted_text = f"OCR extracted text from video: {source}"
            doc_id = hashlib.md5(source.encode()).hexdigest()
            return Document(
                id=doc_id,
                source_uri=source,
                text=extracted_text,
                mime_type="video/mp4",
                routing_tags=["embeddable", "vision-processed"],
                metadata={"file_path": source},
            )
        except Exception as error:
            logger.error(f"Error processing video {source}: {error}")
            raise


# --- Orchestration Layer -----------------------------------------------------


class IngestPipeline:
    """
    Orchestrates ingestion, embedding, and indexing using abstractions.
    """

    def __init__(self, embedding: IEmbeddingService, indexer: IIndexer):
        self.embedding = embedding
        self.indexer = indexer

    async def ingest(self, ingestor: IIngestor, source: str):
        doc = await ingestor.load_and_preprocess(source)

        if "embeddable" in doc.routing_tags:
            doc.embedding = await self.embedding.embed_text(doc.text)

        await self.indexer.upsert(doc)

    async def ingest_batch(self, ingestor: IIngestor, sources: List[str]):
        docs = []
        for source in sources:
            doc = await ingestor.load_and_preprocess(source)
            if "embeddable" in doc.routing_tags:
                doc.embedding = await self.embedding.embed_text(doc.text)
            docs.append(doc)

        await self.indexer.upsert_batch(docs)


# --- Registry + Factory ------------------------------------------------------


class IngestorRegistry:
    """Registry for ingestor factories."""

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, ingestor_class: type):
        cls._registry[name] = ingestor_class

    @classmethod
    def get(cls, name: str, **kwargs) -> IIngestor:
        if name not in cls._registry:
            raise ValueError(f"No ingestor registered for type: {name}")
        return cls._registry[name](**kwargs)


# Register default ingestors
IngestorRegistry.register("text", TextIngestor)
IngestorRegistry.register("pdf", PDFIngestor)
IngestorRegistry.register("document", DocumentIngestor)
IngestorRegistry.register("csv", CSVIngestor)
IngestorRegistry.register("excel", ExcelIngestor)
IngestorRegistry.register("image", ImageIngestor)
IngestorRegistry.register("video", VideoIngestor)


# --- Factory Functions -------------------------------------------------------


def create_data_ingestion_service() -> IngestorRegistry:
    """Factory for ingestor registry."""
    return IngestorRegistry()


def create_ingest_pipeline(
    embedding: IEmbeddingService, indexer: IIndexer
) -> IngestPipeline:
    """Factory for ingest pipeline."""
    return IngestPipeline(embedding, indexer)


# --- Module-level convenience functions ---------------------------------------


async def process_data_for_embedding(
    source: str,
    data_type: str,
    embedding_service: Optional[IEmbeddingService] = None,
    **kwargs,
) -> Document:
    """Convenience function to process data and optionally embed it."""
    ingestor = IngestorRegistry.get(data_type, **kwargs)
    doc = await ingestor.load_and_preprocess(source)

    if embedding_service and "embeddable" in doc.routing_tags:
        doc.embedding = await embedding_service.embed_text(doc.text)

    return doc
