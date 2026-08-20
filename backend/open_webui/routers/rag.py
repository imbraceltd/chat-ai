import logging
import json
import asyncio
import functools
from typing import Optional, Dict, Any, List
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
    Form,
    Query,
)
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from urllib.parse import quote

from open_webui.constants import ERROR_MESSAGES
from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.auth import auth_imbrace
from open_webui.utils.rag import (
    FileUploadError,
    remove as rag_remove,
    upload as rag_upload,
    getById as rag_get_by_id,
    getFileContent as rag_get_file_content,
    getByAssistantId as rag_get_by_assistant_id,
)
from open_webui.llm.main import AnswerQuestionInputs
from open_webui.llm.agent import llm_agent
from open_webui.llm.utils.echart import get_echarts_by_thread_id
from open_webui.llm.utils.databoard import create_databoard_service

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAG", logging.INFO))

router = APIRouter()


# Pydantic models for request/response
class AnswerQuestionRequest(BaseModel):
    text: str
    file_content: str = ""
    thread_id: Optional[str] = ""
    file_ids: Optional[List[str]] = None
    assistant_id: Optional[str] = ""
    provider_id: Optional[str] = None  # Optional provider ID for custom LLM config
    meta_data: Optional[Dict] = None  # Optional metadata (e.g. tool_server)
    # Legacy fields for backward compatibility - accepted but not used
    role: Optional[str] = None
    secret: Optional[str] = None
    instructions: Optional[str] = ""
    knowledge_hubs: Optional[List[str]] = None
    workflow_function_call: Optional[List] = None
    preload_information: Optional[str] = ""
    model: str = ""
    board_ids: Optional[List[str]] = None
    tools: Optional[List[Dict]] = None
    show_thinking_process: Optional[bool] = False
    streaming: Optional[bool] = False
    guardrail_id: Optional[str] = ""
    message_id: Optional[str] = ""
    message_created_at: Optional[str] = ""
    conversation_id: Optional[str] = ""
    created_at: Optional[str] = ""
    visualization: Optional[bool] = False
    tool_server: Optional[Dict] = None
    response_mode: Optional[str] = "standard"  # "standard" or "advanced" (Vercel AI SDK)


class HybridSearchRequest(BaseModel):
    vector_search_query: str
    full_text_search_query: str
    assistant_id: Optional[str] = ""
    file_ids: Optional[List[str]] = None
    folder_ids: Optional[List[str]] = None
    knowledge_hubs: Optional[List[str]] = None
    board_ids: Optional[List[str]] = None
    organization_id: Optional[str] = ""
    # Business metadata constraints applied as AND (narrowing) on top of the
    # scope $or — e.g. {"doc_type": "call_report", "doc_year": "2026"}. Each maps
    # to a data->>'<key>' = <value> condition (engine: _rag_filter_to_where).
    metadata_filters: Optional[Dict[str, str]] = None


class UploadRequest(BaseModel):
    assistant_id: str = ""
    json_input: Optional[str] = None
    text_input: Optional[str] = None
    is_rag: str = "true"
    board_id: str = ""


class UploadResponse(BaseModel):
    id: str
    bytes: int
    filename: str
    assistant_id: str
    created_at: str
    board_id: str
    boarditem_id: str = ""
    extraction_method: Optional[str] = None
    tables_extracted: Optional[int] = None
    text_chunks_extracted: Optional[int] = None
    url: Optional[str] = None


class ErrorResponse(BaseModel):
    message: str


def validate_answer_question_input(
    body: AnswerQuestionRequest, validate_assistant: bool = True
) -> Dict[str, Any]:
    """Validate answer question input similar to JavaScript validator."""

    # Validate file_ids if provided
    if body.file_ids is not None and not isinstance(body.file_ids, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid assistant file IDs - must be an array",
        )

    # Validate text
    if not isinstance(body.text, str):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid thread message - must be a string",
        )

    # Validate streaming
    if body.streaming is not None and not isinstance(body.streaming, bool):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid streaming parameter - must be boolean",
        )

    # Validate assistant_id if required
    if not body.assistant_id and validate_assistant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid assistant ID - required",
        )

    # Validate tools
    if body.tools is not None and not isinstance(body.tools, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid tools - must be an array",
        )

    # Basic tool validation (could be enhanced with more specific validation)
    if body.tools:
        for tool in body.tools:
            if not isinstance(tool, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid tools format - each tool must be an object",
                )

    return {
        "text": body.text,
        "file_content": body.file_content,
        "thread_id": body.thread_id,
        "file_ids": body.file_ids,
        "assistant_id": body.assistant_id,
        "provider_id": body.provider_id,
        # Always return a dict for meta_data to simplify downstream `.get` usage
        "meta_data": body.meta_data or {},
        "instructions": body.instructions,
        "knowledge_hubs": body.knowledge_hubs,
        "workflow_function_call": body.workflow_function_call,
        "model": body.model,
        "board_ids": body.board_ids,
        "tools": body.tools,
        "show_thinking_process": body.show_thinking_process,
        "streaming": body.streaming,
        "preload_information": body.preload_information,
        "guardrail_id": body.guardrail_id,
        "message_id": body.message_id,
        "message_created_at": body.message_created_at,
        "conversation_id": body.conversation_id,
        "created_at": body.created_at,
        "visualization": body.visualization,
        "response_mode": body.response_mode or "standard",
    }


def validate_upload_file_input(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate file input similar to JavaScript validator."""
    required_fields = ["buffer", "mimetype", "originalname"]

    for field in required_fields:
        if field not in file_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required field: {field}",
            )

    # Add size if not present
    if "size" not in file_data and "buffer" in file_data:
        file_data["size"] = len(file_data["buffer"])

    return file_data


def handle_controller_error(error: Exception) -> tuple[str, int]:
    """Handle controller errors with proper sanitization to prevent information leakage."""

    # Log the full error details for debugging (server-side only)
    log.error(f"Controller error: {type(error).__name__}: {str(error)}", exc_info=True)

    # Return sanitized error messages to clients
    if isinstance(error, HTTPException):
        # HTTPExceptions are already intended for clients, but still sanitize
        return error.detail, error.status_code

    elif isinstance(error, ValueError):
        # Value errors are usually safe to expose (validation errors)
        return str(error), status.HTTP_400_BAD_REQUEST

    elif isinstance(error, FileNotFoundError):
        # Generic file not found message - don't expose file paths
        return "Requested file not found", status.HTTP_404_NOT_FOUND

    elif isinstance(error, FileUploadError):
        # File upload errors are usually safe for users
        return str(error), status.HTTP_400_BAD_REQUEST

    elif isinstance(error, PermissionError):
        # Don't expose permission details
        return "Access denied", status.HTTP_403_FORBIDDEN

    elif isinstance(error, ConnectionError):
        # Don't expose connection details
        return "Service temporarily unavailable", status.HTTP_503_SERVICE_UNAVAILABLE

    elif isinstance(error, TimeoutError):
        # Generic timeout message
        return "Request timeout", status.HTTP_408_REQUEST_TIMEOUT

    else:
        # For any other unexpected errors, return a generic message
        # The full error is logged server-side for debugging
        return (
            "An internal error occurred. Please try again later.",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@router.post("/files", response_model=UploadResponse)
async def upload_file(
    file: Optional[UploadFile] = File(None),
    assistant_id: str = Form(""),
    json_input: Optional[str] = Form(None),
    text_input: Optional[str] = Form(None),
    is_rag: str = Form("true"),
    board_id: str = Form(""),
    enable_ocr: bool = Form(False),
    ctx=Depends(auth_imbrace),
):
    """
    Upload file for RAG processing.

    This endpoint handles:
    - File uploads (PDF, DOCX, TXT, CSV, JSON, PPTX)
    - JSON input as text
    - Plain text input
    - RAG processing with vector embeddings
    """
    try:
        log.info(f"ctx context: {ctx}")
        org_id = ctx.get("organization_id")

        uploaded_file = None

        if file:
            # Process file upload
            contents = await file.read()
            if not contents:
                raise ValueError("Empty file content")

            uploaded_file = validate_upload_file_input(
                {
                    "buffer": contents,
                    "mimetype": file.content_type,
                    "originalname": file.filename,
                    "size": len(contents),
                }
            )

        elif json_input:
            # Convert JSON input to a Buffer
            try:
                json_data = json.loads(json_input)
                json_buffer = json.dumps(json_data).encode("utf-8")

                uploaded_file = validate_upload_file_input(
                    {
                        "buffer": json_buffer,
                        "mimetype": "application/json",
                        "originalname": "temp.json",
                        "size": len(json_buffer),
                    }
                )
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON input")

        elif text_input:
            # Convert text input to a Buffer
            text_buffer = text_input.encode("utf-8")

            uploaded_file = validate_upload_file_input(
                {
                    "buffer": text_buffer,
                    "mimetype": "text/plain",
                    "originalname": "temp.txt",
                    "size": len(text_buffer),
                }
            )

        else:
            raise ValueError("No file or input provided")

        # Process with RAG service
        log.info(
            f"Processing RAG upload for organization: {org_id}, assistant: {assistant_id}"
        )

        result = await rag_upload(
            organization_id=org_id,
            assistant_id=assistant_id,
            uploaded_file=uploaded_file,
            boarditem_id="",  # Empty string as in JavaScript
            board_id=board_id,
            enable_ocr=enable_ocr,
        )

        log.info(f"RAG upload completed successfully: {result.get('id')}")

        return UploadResponse(**result)

    except Exception as error:
        message, status_code = handle_controller_error(error)
        log.error(f"Upload error: {message}")
        raise HTTPException(status_code=status_code, detail=message)


@router.delete("/files/{file_id}")
async def remove_file(request: Request, file_id: str, ctx=Depends(auth_imbrace)):
    """
    Remove a RAG file.

    Handles both regular files (starting with 'file-') and RAG files.
    """
    try:
        org_id = ctx.get("organization_id")
        log.info(f"Removing file with ID: {file_id} for organization: {org_id}")

        # Check if fileId starts with 'file-' (regular file)
        if file_id.startswith("file-"):
            # This would integrate with your file service
            # result = await file_service.remove(org_id, None, file_id)
            # return result

            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Regular file removal not implemented yet",
            )

        # Handle RAG file removal - FIX: Use keyword arguments
        result = await rag_remove(
            file_id=file_id,  # Use keyword argument
            organization_id=org_id,  # Use keyword argument
        )

        return result

    except Exception as error:
        message, status_code = handle_controller_error(error)
        log.error(f"Remove file error: {message}")
        raise HTTPException(status_code=status_code, detail=message)


@router.get("/files")
async def get_all_files(
    assistant_id: str = Query(""),
    is_rag: str = Query("true"),
    ctx=Depends(auth_imbrace),
):
    """
    Get all RAG files for an organization and assistant.
    """
    try:
        org_id = ctx.get("organization_id")
        is_rag_bool = is_rag.lower() == "true"

        result = await rag_get_by_assistant_id(
            organization_id=org_id,
            assistant_id=assistant_id,
        )

        return JSONResponse(content=result, status_code=status.HTTP_200_OK)

    except Exception as error:
        message, status_code = handle_controller_error(error)
        raise HTTPException(status_code=status_code, detail=message)


@router.get("/files/content/{file_id}")
async def get_file_content(
    file_id: str,
    attachment: bool = Query(False),
    ctx=Depends(auth_imbrace),
):
    """
    Get the content of a specific RAG file by ID.
    """
    try:
        org_id = ctx.get("organization_id")

        file_details = await rag_get_file_content(file_id=file_id)

        log.info(f"Retrieved file content for file ID: {file_details}")

        file_path = file_details["file_path"]

        path_obj = Path(file_path)
        if not path_obj.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
        # Handle Unicode filenames
        filename = file_details.get("file_name") or file_details.get("filename")
        encoded_filename = quote(filename)  # RFC5987 encoding

        content_type = file_details["file_type"]
        headers = {}

        if attachment:
            headers["Content-Disposition"] = (
                f"attachment; filename*=UTF-8''{encoded_filename}"
            )
        else:
            if content_type == "application/pdf" or filename.lower().endswith(".pdf"):
                headers["Content-Disposition"] = (
                    f"inline; filename*=UTF-8''{encoded_filename}"
                )
                content_type = "application/pdf"
            elif content_type != "text/plain":
                headers["Content-Disposition"] = (
                    f"attachment; filename*=UTF-8''{encoded_filename}"
                )

            return FileResponse(file_path, headers=headers, media_type=content_type)

    except Exception as error:
        message, status_code = handle_controller_error(error)
        if status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status_code, detail="File not found")
        raise HTTPException(status_code=status_code, detail=message)


@router.get("/files/{file_id}")
async def get_file_by_id(
    file_id: str,
    ctx=Depends(auth_imbrace),
):
    """
    Get a specific RAG file by ID.
    """
    try:
        org_id = ctx.get("organization_id")

        result = await rag_get_by_id(file_id=file_id, organization_id=org_id)

        log.info(f"Retrieved file by ID: {file_id} for organization: {org_id}")

        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    except Exception as error:
        message, status_code = handle_controller_error(error)
        if status_code == status.HTTP_404_NOT_FOUND:
            raise HTTPException(status_code=status_code, detail="File not found")
        raise HTTPException(status_code=status_code, detail=message)


@router.post("/answer_question")
async def answer_question(body: AnswerQuestionRequest, ctx=Depends(auth_imbrace)):
    """
    Answer a question using RAG (Retrieval Augmented Generation).

    This endpoint handles both streaming and non-streaming responses
    based on the JavaScript controller logic.
    """
    try:
        org_id = ctx.get("organization_id")
        log.info(f"Processing question for organization: {org_id}")

        # Validate input (not requiring assistant for this endpoint)
        validated_data = validate_answer_question_input(body, validate_assistant=False)

        # Create AnswerQuestionInputs for the agent
        # NOTE: provider_id and tool_server will be resolved inside llm_agent.answer_question
        # from enriched_inputs (fetched from assistant details in DB).
        agent_inputs = AnswerQuestionInputs(
            question=validated_data["text"],
            file_content=validated_data["file_content"],
            assistant_id=validated_data["assistant_id"] or "",
            provider_id=validated_data["provider_id"],
            meta_data=validated_data["meta_data"],
            file_ids=validated_data["file_ids"]
            or [],  # Will be mapped to fileIds in the service
            instructions=validated_data["instructions"] or "",
            thread_id=validated_data["thread_id"] or "",
            board_ids=validated_data["board_ids"] or [],
            conversation_id=validated_data["conversation_id"] or "",
            message_id=validated_data["message_id"] or "",
            message_created_at=validated_data["message_created_at"] or "",
            created_at=validated_data["created_at"] or "",
            knowledge_hubs=validated_data["knowledge_hubs"] or [],
            workflow_function_call=validated_data["workflow_function_call"] or [],
            tools=validated_data["tools"] or [],
            show_thinking_process=validated_data["show_thinking_process"] or False,
            guardrail_id=validated_data["guardrail_id"] or "",
            preload_information=validated_data["preload_information"] or "",
            streaming=(
                validated_data["streaming"]
                if validated_data["streaming"] is not None
                else True
            ),
            visualization=validated_data.get("visualization", False),
            sub_agents=None,  # No sub-agents in this context
            response_mode=validated_data.get("response_mode", "standard")
        )

        model = validated_data["model"] or ""

        # Handle non-streaming response
        if not agent_inputs.streaming:
            log.info("Processing non-streaming response")
            # This would use ragServiceV2.answerQuestion equivalent
            # For now, we'll use the agent's answer_question method
            result = None
            async for chunk in llm_agent.answer_question(
                organization_id=org_id,
                model=model,
                inputs=agent_inputs,
            ):
                result = chunk

            if result:
                return JSONResponse(
                    content=result.dict(), status_code=status.HTTP_200_OK
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No response generated",
                )

        # Handle streaming response
        log.info("Processing streaming response")

        is_advanced = validated_data.get("response_mode") == "advanced"

        async def generate_streaming_response():
            """Generate streaming response chunks"""
            try:
                async for chunk in llm_agent.answer_question(
                    organization_id=org_id,
                    model=model,
                    inputs=agent_inputs,
                ):
                    if is_advanced and isinstance(chunk, str):
                        # Advanced mode: chunks are already Vercel AI SDK SSE strings
                        yield chunk
                    else:
                        # Standard mode: serialize AnswerChunk/AgentChunk as JSON SSE
                        chunk_data = chunk.dict() if hasattr(chunk, "dict") else chunk
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                log.info("End of streaming response")

            except Exception as e:
                log.error(f"Streaming error: {e}")
                if is_advanced:
                    from open_webui.llm.utils.vercel_ai_stream import (
                        error_part,
                        finish_message,
                        stream_done,
                    )

                    yield error_part(str(e))
                    yield finish_message()
                    yield stream_done()
                else:
                    error_chunk = {
                        "error": str(e),
                        "thread_id": agent_inputs.thread_id,
                        "message": "",
                        "is_partial": False,
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
        if is_advanced:
            headers["x-vercel-ai-ui-message-stream"] = "v1"

        return StreamingResponse(
            generate_streaming_response(),
            media_type="text/event-stream",
            headers=headers,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as error:
        message, status_code = handle_controller_error(error)
        log.error(f"Answer question error: {message}")
        raise HTTPException(status_code=status_code, detail=message)


@router.get("/echarts/{thread_id}")
async def get_echarts_by_thread(thread_id: str, ctx=Depends(auth_imbrace)):
    """
    Get all echarts for a specific thread_id, ordered by created_at descending.

    Args:
        thread_id: The thread ID to filter echarts by

    Returns:
        List of echart documents ordered by created_at descending
    """
    try:
        org_id = ctx.get("organization_id")
        log.info(
            f"Retrieving echarts for thread_id: {thread_id}, organization: {org_id}"
        )

        # Only filter by thread_id; organization is not used for filtering
        echarts = await get_echarts_by_thread_id(thread_id=thread_id)

        return JSONResponse(
            content={"echarts": echarts}, status_code=status.HTTP_200_OK
        )

    except Exception as error:
        message, status_code = handle_controller_error(error)
        log.error(f"Get echarts error: {message}")
        raise HTTPException(status_code=status_code, detail=message)


@router.post("/hybrid_search")
async def execute_hybrid_search(body: HybridSearchRequest, request: Request, ctx=Depends(auth_imbrace)):
    """
    Execute hybrid search using both vector and full-text search.

    This endpoint performs hybrid retrieval by combining:
    - Vector similarity search (semantic search)
    - Full-text search (keyword-based search)

    Args:
        body: HybridSearchRequest containing search queries and filters

    Returns:
        Dictionary containing retrieved context and sources
    """
    try:
        org_id = ctx.get("organization_id") or body.organization_id
        log.info(f"Executing hybrid search for organization: {org_id}")

        # Validate required fields
        if not body.vector_search_query and not body.full_text_search_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one of vector_search_query or full_text_search_query must be provided",
            )

        # Create AnswerQuestionInputs object for the hybrid search function
        inputs = AnswerQuestionInputs(
            question=body.vector_search_query or body.full_text_search_query,
            assistant_id=body.assistant_id or "",
            file_ids=body.file_ids or [],
            folder_ids=body.folder_ids or [],
            knowledge_hubs=body.knowledge_hubs or [],
            board_ids=body.board_ids or [],
            organization_id=org_id,
        )

        # Build filters for MongoDB queries
        from open_webui.llm.utils.assistant import build_filters

        filters = build_filters(
            {
                "assistant_id": body.assistant_id,
                "file_ids": body.file_ids,
                "folder_ids": body.folder_ids,
                "knowledge_hubs": body.knowledge_hubs,
                "board_ids": body.board_ids,
            },
            org_id,
        )

        # Apply business metadata constraints as AND (narrowing) on top of the scope
        # $or. Top-level keys in `filters` are AND-joined by _rag_filter_to_where, and
        # any key not in _RAG_DIRECT_COLUMNS resolves to data->>'<key>'. So this turns
        # {"doc_type": "call_report"} into `AND data->>'doc_type' = 'call_report'`.
        if body.metadata_filters:
            for _mkey, _mval in body.metadata_filters.items():
                if _mval is None or _mval == "":
                    continue
                filters[_mkey] = {"$eq": str(_mval)}
            log.info(f"Hybrid search: applied metadata_filters {body.metadata_filters}")

        # Expand folder_ids to include all nested subfolder IDs
        folder_ids = body.folder_ids or []
        folder_info = None  # Will hold the folder structure from the API
        if folder_ids:
            try:
                databoard_svc = create_databoard_service()
                # Forward the caller's identity so data-board can resolve team
                # scope (from x-access-token via platform-service) and drop the
                # requested ids the caller's teams can't access. Without these
                # headers data-board logs "team scope unresolved" and fails
                # open to the whole org.
                identity_headers = {
                    k: v
                    for k, v in {
                        "x-organization-id": request.headers.get("x-organization-id"),
                        "x-user-id": request.headers.get("x-user-id"),
                        "x-access-token": request.headers.get("x-access-token"),
                        "x-api-key": request.headers.get("x-api-key"),
                    }.items()
                    if v
                }
                subfolder_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    functools.partial(
                        databoard_svc.get_folder_subfolders,
                        folder_ids,
                        True,   # recursive
                        True,   # ignore_assistant
                        identity_headers=identity_headers,
                    ),
                )
                if subfolder_response and subfolder_response.get("success"):
                    folder_info = subfolder_response.get("data", {})
                    # Seed ONLY from data-board's response — it is the access
                    # authority. It already dropped ids outside the caller's
                    # team scope; unioning the raw body ids back in would undo
                    # that filtering and leak other teams' folders into the
                    # search scope.
                    all_folder_ids = set()
                    for folder in subfolder_response.get("data", {}).get("folders", []):
                        all_folder_ids.add(folder.get("id", ""))
                        for subfolder in folder.get("subfolders", []):
                            all_folder_ids.add(subfolder.get("id", ""))
                    all_folder_ids.discard("")
                    expanded_folder_ids = list(all_folder_ids)
                    log.info(
                        f"Hybrid search: expanded folder IDs from {len(folder_ids)} to "
                        f"{len(expanded_folder_ids)} (including subfolders)"
                    )

                    # Update inputs.folder_ids with the expanded set
                    inputs.folder_ids = expanded_folder_ids

                    # Update filters to use expanded folder IDs
                    if filters.get("$or"):
                        for i, condition in enumerate(filters["$or"]):
                            if "folder_id" in condition:
                                filters["$or"][i] = {
                                    "folder_id": {"$in": expanded_folder_ids}
                                }
                                break
                        else:
                            # No existing folder_id filter found, add one
                            filters["$or"].append(
                                {"folder_id": {"$in": expanded_folder_ids}}
                            )
                    else:
                        filters.setdefault("$or", []).append(
                            {"folder_id": {"$in": expanded_folder_ids}}
                        )
                else:
                    log.warning(
                        "Hybrid search: subfolder expansion API call failed, "
                        "using original folder_ids"
                    )
            except Exception as expand_error:
                log.warning(
                    f"Hybrid search: error expanding folder IDs: {expand_error}, "
                    "using original folder_ids"
                )

        # Set up the collection name
        collection_name = "rag"  # Default collection name

        # Initialize vector store and collection
        from open_webui.internal.mongo_db import mongodb_client
        from open_webui.llm.agent import get_llm_provider, EMBEDDING_PROVIDER, CONFIG

        # rag_collection is only needed for MongoDB FTS; None in PostgreSQL mode
        from open_webui.internal.document_store import DB_TYPE as _DOC_DB_TYPE
        if _DOC_DB_TYPE == "postgresql":
            rag_collection = None
        else:
            db_client = await mongodb_client.get_client()
            db = db_client[llm_agent.db_name]
            rag_collection = db[collection_name]

        # Create embedding model using the default embedding provider (openai)
        _emb_provider_name = EMBEDDING_PROVIDER if EMBEDDING_PROVIDER else "openai"
        _emb_provider = get_llm_provider(_emb_provider_name, CONFIG)
        embedding_model = _emb_provider.create_embedding_model()
        vector_store = await llm_agent.create_vector_store(
            embedding_model, collection_name
        )

        # Resolve reranker from app state (applied by default when model is configured)
        reranking_function = getattr(request.app.state, "rf", None)
        top_k_reranker = getattr(
            getattr(request.app.state, "config", None), "TOP_K_RERANKER", 3
        )
        relevance_threshold = getattr(
            getattr(request.app.state, "config", None), "RELEVANCE_THRESHOLD", 0.0
        )

        # Execute hybrid search
        log.info(
            f"Executing hybrid search with vector_query='{body.vector_search_query}', full_text_query='{body.full_text_search_query}'"
        )
        context_result, sources = await llm_agent._execute_hybrid_retrieve_context(
            {
                "vector_search_query": body.vector_search_query,
                "full_text_search_query": body.full_text_search_query,
            },
            inputs,
            filters,
            vector_store,
            rag_collection,
            reranking_function=reranking_function,
            top_k_reranker=top_k_reranker,
            relevance_threshold=relevance_threshold,
        )

        log.info(
            f"Hybrid search completed successfully. Context length: {len(context_result) if context_result else 0}, Sources: {len(sources) if sources else 0}"
        )

        return JSONResponse(
            content={
                "context": context_result,
                "sources": sources,
                "folder_info": folder_info,
            },
            status_code=status.HTTP_200_OK,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as error:
        message, status_code = handle_controller_error(error)
        log.error(f"Hybrid search error: {message}")
        raise HTTPException(status_code=status_code, detail=message)


# Export the router
__all__ = ["router"]
