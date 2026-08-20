import logging
import asyncio
import io
import base64
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Path,
    File,
    UploadFile,
    Form,
)
from pydantic import BaseModel
import aiohttp
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils import rag
from open_webui.internal.mongo_db import mongodb_client
from open_webui.config import MONGODB_CONFIG, KIMI_CONFIG, OPENAI_API_KEY
import os as _os

_DB_TYPE = _os.getenv("DB_TYPE", "mongodb").lower()
_DATABASE_URL = _os.getenv("DATABASE_URL", "")


async def _delete_rag_by(field: str, value: str) -> int:
    """Delete rows from the rag store by a single field=value filter.

    Returns the number of rows deleted.
    """
    if _DB_TYPE == "postgresql":
        import asyncpg
        conn = await asyncpg.connect(_DATABASE_URL)
        try:
            result = await conn.execute(
                f"DELETE FROM rag WHERE {field} = $1", value
            )
            return int(result.split()[-1]) if result else 0
        finally:
            await conn.close()
    else:
        result = await mongodb_client.delete_many(
            database_name=DB_NAME,
            collection_name=RAG_COLLECTION_NAME,
            query={field: value},
        )
        return result.get("deleted_count", 0)


async def _rag_exists(field: str, value: str) -> bool:
    """Return True if any rag row matches field=value."""
    if _DB_TYPE == "postgresql":
        import asyncpg
        conn = await asyncpg.connect(_DATABASE_URL)
        try:
            row = await conn.fetchrow(
                f"SELECT 1 FROM rag WHERE {field} = $1 LIMIT 1", value
            )
            return row is not None
        finally:
            await conn.close()
    else:
        docs = await mongodb_client.query(
            database_name=DB_NAME,
            collection_name=RAG_COLLECTION_NAME,
            query={field: value},
        )
        return bool(docs)
from open_webui.routers.ingestion import ingestion_service
from open_webui.retrieval.tabular.duckdb import DuckDBClient
from open_webui.models.files import Files
from open_webui.models.ai_agents.parquets import create as create_parquet_record
from open_webui.models.ai_agents.board_embedding_job import (
    soft_delete_by_board_id as soft_delete_board_embedding_jobs,
)
from open_webui.utils.tabular_description import generate_tabular_description_batch

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("DATABOARD_EMBEDDING", logging.INFO))

router = APIRouter()

# Configuration
DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")
RAG_COLLECTION_NAME = "rag"


# Pydantic models
class EmbeddingData(BaseModel):
    id: str = None
    board_id: str
    organization_id: str
    folder_id: str = ""
    content: str  # Direct content field instead of complex field mapping

    class Config:
        # Allow field alias for _id
        allow_population_by_field_name = True

    # Add a property to handle _id access
    @property
    def _id(self) -> str:
        return self.id

    def __init__(self, **data):
        # Handle _id field mapping
        if "_id" in data:
            data["id"] = data.pop("_id")
        super().__init__(**data)


class EmbeddingRequest(BaseModel):
    data: List[EmbeddingData]


class EmbeddingResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None


class DeleteResponse(BaseModel):
    success: bool
    message: str
    deletedCount: int


class ErrorResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None


# Validation functions
def validate_file_id_input(file_id: str) -> str:
    """Validate file ID input."""
    if not file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file ID"
        )
    return file_id


async def perform_ocr_with_kimi(image_bytes: bytes, filename: str) -> str:
    """Perform OCR and description on image using Kimi vision model."""
    try:
        # Initialize Kimi ChatOpenAI client
        kimi_host = KIMI_CONFIG.get("host", "http://localhost:9000/v1")
        kimi_model = KIMI_CONFIG.get(
            "model", "moonshotai/Kimi-VL-A3B-Thinking-2506"
        )

        llm = ChatOpenAI(
            model=kimi_model,
            api_key=OPENAI_API_KEY,
            base_url=kimi_host,
            temperature=0.1,
            max_tokens=2000,
        )

        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Create message with image
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please extract all visible text from this image and also provide a detailed description of what you see in the image. Include both the extracted text and the description in your response.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        # Get OCR and description result
        response = await llm.ainvoke([message])

        extracted_content = response.content.strip()
        log.info(
            f"OCR and description successful for {filename}: extracted {len(extracted_content)} characters"
        )
        log.info(
            f"OCR extracted content preview: {extracted_content[:200]}{'...' if len(extracted_content) > 200 else ''}"
        )

        return extracted_content

    except Exception as error:
        log.error(f"OCR failed for {filename}: {error}")
        raise


async def process_content_embedding(content: str, metadata: Dict[str, Any]) -> bool:
    """Process content through ingestion service before embedding."""
    try:
        # Convert content to bytes for ingestion
        content_bytes = content.encode("utf-8")

        # Use ingestion service to process the content
        ingested = ingestion_service.ingest_file(
            file_bytes=content_bytes, mime_type="text/plain", filename="content.txt"
        )

        if not ingested or not ingested.content:
            return False

        # Prepare uploaded_file from ingested content
        ingested_content = ingested.content.encode("utf-8")
        uploaded_file = {
            "buffer": ingested_content,
            "mimetype": "text/plain",
            "originalname": "ingested_content.txt",
            "size": len(ingested_content),
        }

        # Upload to RAG with metadata
        await rag.upload(
            organization_id=metadata.get("organization_id", ""),
            assistant_id="",
            uploaded_file=uploaded_file,
            boarditem_id=metadata.get("boarditem_id", ""),
            board_id=metadata.get("board_id", ""),
            folder_id=metadata.get("folder_id", ""),
        )

        return True

    except Exception as error:
        log.error(f"Error processing content embedding: {error}")
        return False


@router.post("", response_model=EmbeddingResponse)
async def embedding(
    body: List[EmbeddingData],
):
    """
    Process embedding data by directly embedding content.
    Simplified version using the ingestion service approach.
    """
    try:
        if not isinstance(body, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body must be an array of data.",
            )

        # Delete existing documents for each boarditem_id
        for item in body:
            if await _rag_exists("boarditem_id", item._id):
                log.info(f"Documents with boarditem_id: {item._id} exist. Deleting them.")
                await _delete_rag_by("boarditem_id", item._id)
                log.info(f"All documents with boarditem_id: {item._id} deleted.")

        # Process each item for embedding
        processed_data = []
        success_count = 0
        fail_count = 0

        for item in body:
            try:
                # Prepare metadata for embedding
                metadata = {
                    "boarditem_id": item._id,
                    "board_id": item.board_id,
                    "organization_id": item.organization_id,
                    "folder_id": item.folder_id,
                }

                # Process content embedding
                success = await process_content_embedding(item.content, metadata)
                if success:
                    success_count += 1
                    processed_data.append(
                        {
                            "boarditem_id": item._id,
                            "board_id": item.board_id,
                            "organization_id": item.organization_id,
                            "folder_id": item.folder_id,
                            "content_length": len(item.content),
                        }
                    )
                else:
                    fail_count += 1

            except Exception as item_error:
                log.error(f"Error processing item {item._id}: {item_error}")
                fail_count += 1

        log.info(f"Successfully embedded {success_count} items, failed {fail_count}")

        return EmbeddingResponse(
            success=True,
            message=f"Embedded {success_count} items successfully, {fail_count} failed",
            data=processed_data,
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error in embedding process: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data embedding failed",
        )


@router.post("/upload", response_model=EmbeddingResponse)
async def upload_and_embed_file(
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None, description="URL to download file from instead of uploading"),
    organization_id: str = Form(..., description="Required organization ID"),
    folder_id: str = Form(..., description="Required folder ID for filtering"),
    board_id: Optional[str] = Form(None, description="Optional board ID"),
    boarditem_id: Optional[str] = Form(None, description="Optional board item ID"),
    file_id: Optional[str] = Form(None, description="Optional file ID to use instead of generating a new UUID"),
):
    """
    Upload a file, extract its content using the ingestion service, and embed it.
    Supports: text, CSV, Excel, PDF, DOCX, images (OCR), videos (metadata)
    Accepts either a file upload or a file_url to download from.
    """
    try:
        if not file and not file_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either file or file_url must be provided",
            )

        # Download from URL if file_url provided
        if file_url:
            async with aiohttp.ClientSession() as session:
                async with session.get(file_url) as resp:
                    if resp.status != 200:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Failed to download file from URL: {resp.status}",
                        )
                    file_content = await resp.read()
            filename = file_url.split("?")[0].split("/")[-1] or "document"
        else:
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must have a filename",
                )
            file_content = await file.read()
            filename = file.filename

        # Determine MIME type
        import mimetypes

        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            # Fallback: try to determine from file content
            if file_content.startswith(b"%PDF"):
                mime_type = "application/pdf"
            elif file_content.startswith(
                b"PK\x03\x04"
            ):  # ZIP signature (used by DOCX, XLSX)
                if filename.lower().endswith(".docx"):
                    mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif filename.lower().endswith(".xlsx"):
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    mime_type = "application/zip"
            else:
                mime_type = "application/octet-stream"

        # Additional fallback: check file extension for CSV/XLSX if MIME type is not detected
        if mime_type not in [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]:
            if filename.lower().endswith(".csv"):
                mime_type = "text/csv"
            elif filename.lower().endswith(".xlsx"):
                mime_type = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif filename.lower().endswith(".xls"):
                mime_type = "application/vnd.ms-excel"

        # Check if this is an image file for OCR processing
        is_image = mime_type and mime_type.startswith("image/")

        # Check if this is a tabular file (CSV or Excel)
        is_tabular = mime_type in [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]

        log.info(
            f"File: {filename}, MIME type: {mime_type}, Is image: {is_image}, Is tabular: {is_tabular}"
        )

        # Process tabular files through DuckDB
        tabular_data = None
        if is_tabular:
            try:
                log.info(f"Processing tabular file: {mime_type}")

                # Read file into DataFrame
                if mime_type == "text/csv":
                    df = pd.read_csv(io.BytesIO(file_content))
                elif mime_type in [
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ]:
                    df = pd.read_excel(io.BytesIO(file_content), engine="openpyxl")
                else:
                    raise ValueError(f"Unsupported tabular file type: {mime_type}")

                # Validate DataFrame
                if df.empty:
                    raise ValueError("The file is empty")
                if len(df) < 1:
                    raise ValueError("File must contain at least one row of data")
                if len(df.columns) < 1:
                    raise ValueError("File must contain at least one column")

                # Generate a file ID for tabular storage
                import uuid

                tabular_file_id = str(uuid.uuid4())

                # Generate AI description for the tabular data
                log.info(f"Generating AI description for tabular data: {filename}")
                description = await generate_tabular_description_batch(
                    df, filename
                )

                # Store tabular data in DuckDB and create Parquet file
                duckdb_client = DuckDBClient(in_memory=True)
                try:
                    s3_path = duckdb_client.tabular_uri(tabular_file_id)
                    upload_result = duckdb_client.write_file_to_s3(
                        tabular_file_id,
                        df,
                        f"tabular/{tabular_file_id}.parquet",
                        "parquet",
                        filename,
                    )

                    if upload_result:
                        log.info(
                            f"Successfully stored tabular data in DuckDB and S3: {s3_path}"
                        )

                        # Store tabular metadata for later use
                        tabular_data = {
                            "file_id": tabular_file_id,
                            "s3_path": s3_path,
                            "row_count": len(df),
                            "column_count": len(df.columns),
                            "columns": df.columns.tolist(),
                        }

                        # Save to parquets collection for future RAG queries
                        parquet_record = {
                            "id": tabular_file_id,
                            "file_name": filename,
                            "organization_id": organization_id,
                            "folder_id": folder_id,
                            "board_id": board_id,
                            "boarditem_id": boarditem_id,
                            "s3_path": s3_path,
                            "description": description,
                            "columns": df.columns.tolist(),
                            "row_count": len(df),
                            "column_count": len(df.columns),
                            "file_type": mime_type,
                            "file_size": len(file_content),
                        }

                        await create_parquet_record(parquet_record)
                        log.info(
                            f"Saved parquet metadata to database: {tabular_file_id}"
                        )

                    else:
                        log.warning(
                            "Failed to upload tabular data to S3, continuing with text processing"
                        )

                except Exception as duckdb_error:
                    log.warning(
                        f"DuckDB processing failed: {duckdb_error}, continuing with text processing"
                    )

            except Exception as tabular_error:
                log.warning(
                    f"Tabular processing failed: {tabular_error}, falling back to text processing"
                )
                is_tabular = False

        # Delete existing documents for this boarditem_id (only if provided)
        if boarditem_id:
            if await _rag_exists("boarditem_id", boarditem_id):
                log.info(f"Documents with boarditem_id: {boarditem_id} exist. Deleting them.")
                await _delete_rag_by("boarditem_id", boarditem_id)
                log.info(f"All documents with boarditem_id: {boarditem_id} deleted.")

        # Process image files with OCR using Kimi
        if is_image:
            log.info(f"Processing image file with OCR: {filename}")

            # Perform OCR and get description using Kimi
            ocr_content = await perform_ocr_with_kimi(file_content, filename)

            if not ocr_content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to extract content from image using OCR",
                )

            # Prepare uploaded_file from OCR content
            ocr_content_bytes = ocr_content.encode("utf-8")
            uploaded_file = {
                "buffer": ocr_content_bytes,
                "mimetype": "text/plain",
                "originalname": filename,  # Use original image filename
                "size": len(ocr_content_bytes),
            }

            log.info(
                f"Content to be embedded for image {filename}: {ocr_content[:500]}{'...' if len(ocr_content) > 500 else ''}"
            )

            # Upload to RAG with no_split=True for single record embedding
            await rag.upload(
                organization_id=organization_id,
                assistant_id="",
                uploaded_file=uploaded_file,
                boarditem_id=boarditem_id,
                board_id=board_id,
                folder_id=folder_id,
                no_split=True,
            )

            # Prepare response data for image
            response_data = {
                "boarditem_id": boarditem_id or None,
                "board_id": board_id or None,
                "organization_id": organization_id,
                "folder_id": folder_id,
                "file_name": filename,
                "file_type": mime_type,
                "content_length": len(ocr_content),
                "metadata": {
                    "extraction_method": "kimi_ocr_vision",
                    "is_image": True,
                },
            }

        elif mime_type == "application/pdf":
            # Pass PDF directly to RAG so it runs the PDF extractor
            # with table/text separation and isTable metadata
            uploaded_file = {
                "buffer": file_content,
                "mimetype": "application/pdf",
                "originalname": filename,
                "size": len(file_content),
            }

            rag_result = await rag.upload(
                organization_id=organization_id,
                assistant_id="",
                uploaded_file=uploaded_file,
                boarditem_id=boarditem_id,
                board_id=board_id,
                folder_id=folder_id,
                file_id=file_id or "",
            )
            response_data = {
                "boarditem_id": boarditem_id or None,
                "board_id": board_id or None,
                "organization_id": organization_id,
                "folder_id": folder_id,
                "file_name": filename,
                "file_type": "pdf",
                "content_length": len(file_content),
                "metadata": {"extraction_method": "pdf_with_table_support"},
            }

        else:
            # Process non-image, non-PDF files through ingestion service
            ingested = ingestion_service.ingest_file(
                file_bytes=file_content, mime_type=mime_type, filename=filename
            )

            if not ingested or not ingested.content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to extract content from file",
                )

            # Prepare uploaded_file from ingested content
            ingested_content = ingested.content.encode("utf-8")
            uploaded_file = {
                "buffer": ingested_content,
                "mimetype": "text/plain",
                "originalname": f"ingested_{filename}.txt",
                "size": len(ingested_content),
            }

            # Upload to RAG with metadata
            rag_result = await rag.upload(
                organization_id=organization_id,
                assistant_id="",
                uploaded_file=uploaded_file,
                boarditem_id=boarditem_id,
                board_id=board_id,
                folder_id=folder_id,
                file_id=file_id or "",
            )
            # Prepare response data for non-image files
            response_data = {
                "boarditem_id": boarditem_id or None,
                "board_id": board_id or None,
                "organization_id": organization_id,
                "folder_id": folder_id,
                "file_name": filename,
                "file_type": ingested.file_type,
                "content_length": len(ingested.content),
                "metadata": ingested.metadata,
            }

        # Add tabular data information if processed
        if tabular_data:
            response_data["tabular_data"] = tabular_data
            log.info(f"Including tabular data in response: {tabular_data}")
        else:
            log.info("No tabular data to include in response")

        return EmbeddingResponse(
            success=True,
            message=f"Successfully embedded file: {filename}",
            data=[response_data],
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error processing file upload and embedding: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(error)}",
        )


@router.delete("/board/{board_id}", response_model=DeleteResponse)
async def delete_embedding_by_board_id(
    board_id: str = Path(..., description="The board ID to delete"),
):
    """
    Delete embedding data by board ID.
    """
    try:
        # Validate the board ID
        validate_file_id_input(board_id)

        deleted_count = await _delete_rag_by("board_id", board_id)
        jobs_soft_deleted = await soft_delete_board_embedding_jobs(board_id)

        return DeleteResponse(
            success=True,
            message=f"Deleted {deleted_count} rag records and soft-deleted {jobs_soft_deleted} embedding job(s) for board_id = {board_id}",
            deletedCount=deleted_count,
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error deleting embeddings: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting embeddings",
        )


@router.delete("/file/{file_id}", response_model=DeleteResponse)
async def delete_embedding_by_file_id(
    file_id: str = Path(..., description="The file ID to delete"),
):
    """
    Delete embedding data by file ID.

    Used by data-board when a knowledge-hub file is deleted so the rag rows
    backing hybrid search are removed from the actual vector store
    (Postgres or Mongo, whichever this service is configured for).
    """
    try:
        # Validate the file ID
        validate_file_id_input(file_id)

        deleted_count = await _delete_rag_by("file_id", file_id)

        return DeleteResponse(
            success=True,
            message=f"Deleted {deleted_count} records with file_id = {file_id}",
            deletedCount=deleted_count,
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error deleting embeddings: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting embeddings",
        )


@router.delete("/{item_id}", response_model=DeleteResponse)
async def delete_embedding(
    item_id: str = Path(..., description="The boarditem ID to delete"),
):
    """
    Delete embedding data by boarditem ID.
    """
    try:
        # Validate the ID
        validate_file_id_input(item_id)

        deleted_count = await _delete_rag_by("boarditem_id", item_id)

        return DeleteResponse(
            success=True,
            message=f"Deleted {deleted_count} records with boarditem_id = {item_id}",
            deletedCount=deleted_count,
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error deleting embeddings: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting embeddings",
        )


# Export the router
__all__ = ["router"]
