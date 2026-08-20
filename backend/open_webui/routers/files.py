import asyncio
import logging
import os
import uuid
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional
from urllib.parse import quote
import io
import httpx
import pandas as pd

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
    Query,
)
from fastapi.responses import FileResponse, StreamingResponse
from open_webui.utils.file import upload_assistant_file
from open_webui.utils.pdf_tabular_data_extractor import embedPDF
from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.files import (
    FileForm,
    FileModel,
    FileModelResponse,
    Files,
)
from open_webui.models.knowledge import Knowledges
from open_webui.retrieval.tabular.duckdb import DuckDBClient
from open_webui.config import STORAGE_PROVIDER

from open_webui.routers.knowledge import get_knowledge, get_knowledge_list
from open_webui.routers.retrieval import ProcessFileForm, process_file
from open_webui.routers.audio import transcribe
from open_webui.storage.provider import Storage, S3StorageProvider
from open_webui.utils.auth import get_admin_user, get_verified_user
from pydantic import BaseModel
from fastapi import Form

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])


# Utility functions for tabular data processing
def describe_dataframe(df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Returns a list of tuples where each tuple contains:
    - Column name as a string.
    - Column data type as a string.
    """
    return [(col, str(dtype)) for col, dtype in df.dtypes.items()]


def summarize_dataframe(df: pd.DataFrame) -> dict:
    """
    Returns a dictionary containing statistical summaries for each column.
    - Numeric columns will show count, mean, std, min, max, and percentiles.
    - Non-numeric columns will show count, unique, top (most frequent), and freq (frequency).
    """
    # Initialize the summary dictionary
    summary = {}

    # Using pandas `describe` method to get statistics for numeric columns
    try:
        numeric_summary = df.describe(include=[float, int]).fillna("")
        if not numeric_summary.empty:
            summary.update(numeric_summary.to_dict())
    except Exception as e:
        log.warning(f"Error generating numeric summary: {e}")

    # Using pandas `describe` method to get statistics for non-numeric (object) columns
    try:
        object_summary = df.describe(include=[object]).fillna("")
        if not object_summary.empty:
            summary.update(object_summary.to_dict())
    except Exception as e:
        log.warning(f"Error generating object summary: {e}")

    return summary


def top_dataframe(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Returns the top n rows of the DataFrame."""
    if df is not None and not df.empty:
        return df.head(n).to_dict(orient="records")
    return []


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validates a dataframe before saving to S3.
    Returns a tuple of (is_valid, error_message).

    Parameters:
    - df: The pandas DataFrame to validate
    - max_rows: Maximum number of rows allowed (default: 100,000)
    - max_columns: Maximum number of columns allowed (default: 500)
    """
    # Check if dataframe is empty
    if df.empty:
        return False, "The file is empty"

    # Check for minimum number of rows
    if len(df) < 1:
        return False, "File must contain at least one row of data"

    # Check for minimum number of columns
    if len(df.columns) < 1:
        return False, "File must contain at least one column"

    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        return False, "File contains duplicate column names"

    # Check for invalid column names (e.g., None, empty strings)
    for col in df.columns:
        if col is None or (isinstance(col, str) and col.strip() == ""):
            return False, "File contains empty or invalid column names"
    # All checks passed
    return True, ""


router = APIRouter()


############################
# Check if the current user has access to a file through any knowledge bases the user may be in.
############################


def has_access_to_file(
    file_id: Optional[str], access_type: str, user=Depends(get_verified_user)
) -> bool:
    file = Files.get_file_by_id(file_id)
    log.debug(f"Checking if user has {access_type} access to file")

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    has_access = False
    knowledge_base_id = file.meta.get("collection_name") if file.meta else None

    if knowledge_base_id:
        knowledge_bases = Knowledges.get_knowledge_bases_by_user_id(
            user.id, access_type
        )
        for knowledge_base in knowledge_bases:
            if knowledge_base.id == knowledge_base_id:
                has_access = True
                break

    return has_access


############################
# Upload File
############################


def process_tabular_file(
    file_content: bytes, content_type: str, file_name: str, file_id: str
):
    """Process CSV or Excel file using the tabular interface and store in DuckDB."""
    try:
        if content_type == "text/csv":
            df = pd.read_csv(io.BytesIO(file_content))
        elif content_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]:
            df = pd.read_excel(io.BytesIO(file_content), engine="openpyxl")
        else:
            raise ValueError(f"Unsupported file type: {content_type}")

        # Validate the dataframe before proceeding
        is_valid, error_message = validate_dataframe(df)
        if not is_valid:
            raise ValueError(error_message)

        description = describe_dataframe(df)
        summary = summarize_dataframe(df)
        preview = top_dataframe(df, 5)

        # Convert DataFrame to JSON format for AI agent consumption
        # This provides structured data that can be easily parsed by LLMs
        content = df.fillna("").to_dict(orient="records")

        Files.update_file_data_by_id(file_id, {"content": content})

        try:
            duckdb_client = DuckDBClient(in_memory=True)

            try:
                s3_path = f"tabular/{file_id}.parquet"
                upload_result = duckdb_client.write_file_to_s3(
                    file_id, df, s3_path, "parquet"
                )

                if upload_result:
                    log.info(f"Successfully uploaded tabular data to S3: {s3_path}")
                    summary["s3_path"] = s3_path
            except Exception as s3_err:
                log.warning(f"Could not upload to S3: {s3_err}")

        except Exception as e:
            log.exception(f"Error storing tabular data in DuckDB: {e}")

        return {
            "description": description,
            "summary": summary,
            "top_n": preview,
            "content": content,
        }
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing tabular file: {str(e)}",
        )


@router.post("/agent")
async def upload_agent_file(
    request: Request,
    agent_id: str = Form(...),
    file: UploadFile = File(...),
    is_rag: bool = Form(True),
):
    try:
        result = await upload_assistant_file(request, agent_id, file, is_rag)
        return {"message": "File uploaded successfully", "result": result}
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error uploading file: {str(e)}",
        )


@router.post("/extract")
async def extract_file(
    file: UploadFile = File(None),
    url: str = Form(None),
    method: str = Form("custom"),
):
    if not file and not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either file or url must be provided",
        )
    if file and url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either file or url, not both",
        )

    valid_methods = {"custom", "langchain"}
    if method not in valid_methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid method '{method}'. Must be one of: {', '.join(sorted(valid_methods))}",
        )

    try:
        if method == "langchain":
            from open_webui.utils.pdf_tabular_data_extractor import extract_pdf_with_langchain_loader
            extractor = extract_pdf_with_langchain_loader
        else:
            extractor = embedPDF

        if url:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=60.0)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"URL does not point to a PDF (content-type: {content_type})",
                    )
                pdf_file = io.BytesIO(response.content)
            result = await asyncio.to_thread(extractor, pdf_file)
        else:
            result = await asyncio.to_thread(extractor, file.file)

        return {"message": "File extracted successfully", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error extracting file: {str(e)}",
        )


@router.post("/", response_model=FileModelResponse)
def upload_file(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(get_verified_user),
    file_metadata: dict = {},
    agent_file_id: Optional[str] = None,
    process: bool = Query(True),
):
    log.info(f"file: {file}")
    try:
        unsanitized_filename = file.filename
        filename = os.path.basename(unsanitized_filename)

        # replace filename with uuid
        id = str(uuid.uuid4())
        name = filename
        filename = f"{id}_{filename}"

        # Check if this is a tabular file
        tabular_file_types = [
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]

        if file.content_type in tabular_file_types:
            # For tabular files, process in memory without saving to disk
            contents = file.file.read()
            if not contents:
                raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)

            # Create a virtual path for database record, but don't actually save the file
            file_path = f"virtual_tabular/{filename}"

            # Create file record
            file_item = Files.insert_new_file(
                user.id,
                FileForm(
                    **{
                        "id": id,
                        "filename": name,
                        "path": file_path,
                        "meta": {
                            "name": name,
                            "content_type": file.content_type,
                            "size": len(contents),
                            "data": file_metadata,
                            "virtual": True,  # Mark as virtual (not stored on disk)
                        },
                    }
                ),
            )

            if process:
                # Process tabular file
                log.info(f"Processing tabular file: {file.content_type}")
                result = process_tabular_file(
                    contents, file.content_type, file.filename, id
                )

                # Update file metadata with tabular processing results but exclude content
                Files.update_file_metadata_by_id(
                    id,
                    {
                        "description": result["description"],
                        "summary": result["summary"],
                        "top_n": result["top_n"],
                    },
                )

                # Return file item with content directly in response
                file_item = Files.get_file_by_id(id=id)
                file_item.data = {"content": result["content"]}
                return file_item
        else:
            # For non-tabular files, use the regular upload process
            contents, file_path = Storage.upload_file(file.file, filename)

            file_item = Files.insert_new_file(
                user.id,
                FileForm(
                    **{
                        "id": id,
                        "filename": name,
                        "path": file_path,
                        "meta": {
                            "name": name,
                            "content_type": file.content_type,
                            "size": len(contents),
                            "file_agent_id": agent_file_id,
                            "data": file_metadata,
                        },
                    }
                ),
            )

            if process:
                try:
                    # Process audio files
                    if file.content_type in [
                        "audio/mpeg",
                        "audio/wav",
                        "audio/ogg",
                        "audio/x-m4a",
                    ]:
                        file_path = Storage.get_file(file_path)
                        result = transcribe(request, file_path)

                        process_file(
                            request,
                            ProcessFileForm(file_id=id, content=result.get("text", "")),
                            user=user,
                        )
                    # Process other non-image files
                    elif file.content_type not in [
                        "image/png",
                        "image/jpeg",
                        "image/gif",
                    ]:
                        process_file(request, ProcessFileForm(file_id=id), user=user)

                    file_item = Files.get_file_by_id(id=id)
                except Exception as e:
                    log.exception(e)
                    log.error(f"Error processing file: {file_item.id}")
                    file_item = FileModelResponse(
                        **{
                            **file_item.model_dump(),
                            "error": str(e.detail) if hasattr(e, "detail") else str(e),
                        }
                    )

        if file_item:
            return file_item
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error uploading file"),
            )

    except Exception as e:
        log.exception(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


############################
# List Files
############################


@router.get("/", response_model=list[FileModelResponse])
async def list_files(user=Depends(get_verified_user), content: bool = Query(True)):
    if user.role == "admin":
        files = Files.get_files()
    else:
        files = Files.get_files_by_user_id(user.id)

    if not content:
        for file in files:
            del file.data["content"]

    return files


############################
# Search Files
############################


@router.get("/search", response_model=list[FileModelResponse])
async def search_files(
    filename: str = Query(
        ...,
        description="Filename pattern to search for. Supports wildcards such as '*.txt'",
    ),
    content: bool = Query(True),
    user=Depends(get_verified_user),
):
    """
    Search for files by filename with support for wildcard patterns.
    """
    # Get files according to user role
    if user.role == "admin":
        files = Files.get_files()
    else:
        files = Files.get_files_by_user_id(user.id)

    # Get matching files
    matching_files = [
        file for file in files if fnmatch(file.filename.lower(), filename.lower())
    ]

    if not matching_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No files found matching the pattern.",
        )

    if not content:
        for file in matching_files:
            del file.data["content"]

    return matching_files


############################
# Delete All Files
############################


@router.delete("/all")
async def delete_all_files(user=Depends(get_admin_user)):
    result = Files.delete_all_files()
    if result:
        try:
            Storage.delete_all_files()
        except Exception as e:
            log.exception(e)
            log.error("Error deleting files")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
            )
        return {"message": "All files deleted successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
        )


############################
# Get File By Id
############################


@router.get("/{id}", response_model=Optional[FileModel])
async def get_file_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user)
    ):
        return file
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Get File Data Content By Id
############################


@router.get("/{id}/data/content")
async def get_file_data_content_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user)
    ):
        return {"content": file.data.get("content", "")}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Update File Data Content By Id
############################


class ContentForm(BaseModel):
    content: str


@router.post("/{id}/data/content/update")
async def update_file_data_content_by_id(
    request: Request, id: str, form_data: ContentForm, user=Depends(get_verified_user)
):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "write", user)
    ):
        try:
            process_file(
                request,
                ProcessFileForm(file_id=id, content=form_data.content),
                user=user,
            )
            file = Files.get_file_by_id(id=id)
        except Exception as e:
            log.exception(e)
            log.error(f"Error processing file: {file.id}")

        return {"content": file.data.get("content", "")}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Get File Content By Id
############################


@router.get("/{id}/content")
async def get_file_content_by_id(
    id: str, user=Depends(get_verified_user), attachment: bool = Query(False)
):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user)
    ):
        try:
            file_path = Storage.get_file(file.path)

            # For tabular files that might be in S3
            meta = file.meta
            if meta and meta.get("content_type") in [
                "text/csv",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ]:
                try:
                    duckdb_client = DuckDBClient(in_memory=True)
                    s3_path = f"tabular/{file.id}.parquet"
                    file_data = duckdb_client.read_file_from_s3(
                        s3_path, file_type="parquet"
                    )
                    log.info(f"Retrieved tabular data from S3")

                    # Get filename and content type from metadata
                    filename = file.meta.get("name", file.filename)
                    encoded_filename = quote(filename)  # RFC5987 encoding
                    content_type = file.meta.get("content_type")

                    # Create in-memory file buffer
                    output = io.BytesIO()

                    # Determine export format based on content type and filename
                    if content_type == "text/csv" or filename.lower().endswith(".csv"):
                        media_type = "text/csv"
                        file_data.to_csv(output, index=False)
                        if not filename.lower().endswith(".csv"):
                            filename += ".csv"
                    else:
                        # Default to Excel for all other tabular types
                        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        file_data.to_excel(output, index=False)
                        if not any(
                            filename.lower().endswith(ext) for ext in [".xls", ".xlsx"]
                        ):
                            filename += ".xlsx"

                    # Prepare for download
                    output.seek(0)
                    headers = {
                        "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
                    }

                    return StreamingResponse(
                        iter([output.getvalue()]),
                        media_type="text/csv",
                        headers=headers,
                    )
                except Exception as e:
                    log.exception(f"Error processing tabular data: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Error processing tabular data: {str(e)}",
                    )
            else:
                # Handle local files
                file_path = Storage.get_file(file.path)
                path_obj = Path(file_path)
                if not path_obj.is_file():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=ERROR_MESSAGES.NOT_FOUND,
                    )

                # Handle Unicode filenames
                filename = file.meta.get("name", file.filename)
                encoded_filename = quote(filename)  # RFC5987 encoding

                content_type = file.meta.get("content_type")
                headers = {}

                if attachment:
                    headers["Content-Disposition"] = (
                        f"attachment; filename*=UTF-8''{encoded_filename}"
                    )
                else:
                    if content_type == "application/pdf" or filename.lower().endswith(
                        ".pdf"
                    ):
                        headers["Content-Disposition"] = (
                            f"inline; filename*=UTF-8''{encoded_filename}"
                        )
                        content_type = "application/pdf"
                    elif content_type != "text/plain":
                        headers["Content-Disposition"] = (
                            f"attachment; filename*=UTF-8''{encoded_filename}"
                        )

                return FileResponse(file_path, headers=headers, media_type=content_type)

        except Exception as e:
            log.exception(e)
            log.error("Error getting file content")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error getting file content"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.get("/{id}/content/html")
async def get_html_file_content_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user)
    ):
        try:
            file_path = Storage.get_file(file.path)
            file_path = Path(file_path)

            # Check if the file already exists in the cache
            if file_path.is_file():
                log.info(f"file_path: {file_path}")
                return FileResponse(file_path)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ERROR_MESSAGES.NOT_FOUND,
                )
        except Exception as e:
            log.exception(e)
            log.error("Error getting file content")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error getting file content"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.get("/{id}/content/{file_name}")
async def get_file_content_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "read", user)
    ):
        file_path = file.path

        # Handle Unicode filenames
        filename = file.meta.get("name", file.filename)
        encoded_filename = quote(filename)  # RFC5987 encoding
        headers = {
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
        }

        if file_path:
            file_path = Storage.get_file(file_path)
            file_path = Path(file_path)

            # Check if the file already exists in the cache
            if file_path.is_file():
                return FileResponse(file_path, headers=headers)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=ERROR_MESSAGES.NOT_FOUND,
                )
        else:
            # File path doesn't exist, return the content as .txt if possible
            file_content = file.content.get("content", "")
            file_name = file.filename

            # Create a generator that encodes the file content
            def generator():
                yield file_content.encode("utf-8")

            return StreamingResponse(
                generator(),
                media_type="text/plain",
                headers=headers,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# Delete File By Id
############################


@router.delete("/{id}")
async def delete_file_by_id(id: str, user=Depends(get_verified_user)):
    file = Files.get_file_by_id(id)

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        file.user_id == user.id
        or user.role == "admin"
        or has_access_to_file(id, "write", user)
    ):
        # Check if this is a virtual tabular file
        is_virtual = file.meta and file.meta.get("virtual") == True

        # Delete from database
        result = Files.delete_file_by_id(id)
        if result:
            try:
                # For virtual files, we don't need to delete from local storage
                if not is_virtual:
                    Storage.delete_file(file.path)

                # For tabular files (both virtual and normal), try to delete from S3 if applicable
                if file.meta and file.meta.get("content_type") in [
                    "text/csv",
                    "application/vnd.ms-excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ]:
                    # Attempt to delete the parquet file from S3 if it exists
                    try:
                        s3_path = f"tabular/{id}.parquet"
                        # Use the appropriate storage provider based on configuration
                        if hasattr(Storage, "s3_client"):
                            log.info(
                                f"Attempting to delete tabular file from S3: {s3_path}"
                            )
                            Storage.s3_client.delete_object(
                                Bucket=Storage.bucket_name, Key=s3_path
                            )
                    except Exception as s3_error:
                        log.warning(
                            f"Error deleting S3 tabular file (non-critical): {s3_error}"
                        )
            except Exception as e:
                log.exception(e)
                log.error("Error deleting files")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT("Error deleting files"),
                )
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error deleting file"),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
