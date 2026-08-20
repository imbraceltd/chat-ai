import logging
import uuid
from typing import List, Dict, Any, Optional
import pandas as pd
from open_webui.config import VLLM_CONFIG
from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.ai_agents.board_embedding_job import (
    create as create_job,
    get_by_filters as get_jobs_by_filters,
    get_pending_jobs,
    mark_job_started,
    mark_job_completed,
    mark_job_failed,
    JobStatus,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BOARD_EMBEDDING_JOB", logging.INFO))


async def create_board_embedding_jobs_for_assistant(
    organization_id: str, assistant_id: str, board_ids: List[str]
) -> List[str]:
    """
    Create board embedding jobs for an assistant's board_ids.
    First marks each board with is_embed=true, then creates embedding jobs.
    Checks for existing jobs and only creates new ones if they don't exist.

    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID
        board_ids: List of board IDs

    Returns:
        List of created job IDs
    """
    created_job_ids = []

    # Board content is served via SPARQL (query_board_data / get_board_rdf), NOT vector
    # search. Embedding boards into the `rag` table only pollutes document retrieval
    # (board/news chunks outrank sparse document chunks) and wastes storage + embedding
    # cost. Disabled by default; set ENABLE_BOARD_EMBEDDING=true to restore.
    import os
    if os.getenv("ENABLE_BOARD_EMBEDDING", "false").lower() != "true":
        log.info(
            "Board embedding disabled (ENABLE_BOARD_EMBEDDING != true) — "
            f"skipping {len(board_ids)} board(s) for assistant {assistant_id}"
        )
        return created_job_ids

    for board_id in board_ids:
        try:
            # Step 1: Mark board with is_embed=true (without changing its type)
            await _mark_board_as_embed(board_id)

            # Step 2: Check if a job already exists for this board_id and assistant_id
            existing_jobs = await get_jobs_by_filters(
                organization_id=organization_id,
                board_id=board_id,
                assistant_id=assistant_id,
            )

            if existing_jobs:
                log.info(
                    f"Job already exists for board_id {board_id} and assistant_id {assistant_id}"
                )
                continue

            # Step 3: Create new job
            job_details = {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "assistant_id": assistant_id,
                "board_id": board_id,
                "status": JobStatus.PENDING.value,
                "retry_count": 0,
                "max_retries": 3,
            }

            created_job = await create_job(job_details)
            if created_job:
                created_job_ids.append(job_details["id"])
                log.info(
                    f"Created board embedding job {job_details['id']} for board {board_id}"
                )
            else:
                log.error(f"Failed to create job for board {board_id}")

        except Exception as e:
            log.error(f"Error creating job for board {board_id}: {e}")

    return created_job_ids


async def process_pending_board_embedding_job() -> Optional[Dict[str, Any]]:
    """
    Process the next pending board embedding job.
    Only processes one job at a time as requested.

    Returns:
        Job details if processed, None if no jobs available
    """
    try:
        log.info("🔍 Getting pending jobs (limit=1)...")
        # Get one pending job
        pending_jobs = await get_pending_jobs(limit=1)
        log.info(f"📊 Found {len(pending_jobs) if pending_jobs else 0} pending jobs")

        if not pending_jobs:
            log.info("⚠️ No pending board embedding jobs found")
            return None

        job = pending_jobs[0]
        job_id = job["id"]
        log.info(f"🎯 Processing job {job_id} for board {job.get('board_id')}")

        # Mark job as started
        log.info(f"📝 Marking job {job_id} as started...")
        if not await mark_job_started(job_id):
            log.error(f"❌ Failed to mark job {job_id} as started")
            return None

        log.info(
            f"✅ Job {job_id} marked as started, now processing board embedding job {job_id} for board {job.get('board_id')}"
        )

        # Process the job
        success, error_message = await _process_board_embedding_job(job)

        if success:
            await mark_job_completed(job_id)
            log.info(f"Successfully completed board embedding job {job_id}")
        else:
            await mark_job_failed(job_id, error_message or "Job processing failed")
            log.error(
                f"Failed to process board embedding job {job_id}: {error_message}"
            )

        return job

    except Exception as e:
        log.error(f"Error processing board embedding job: {e}")
        return None


async def _process_board_embedding_job(
    job: Dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """
    Process a single board embedding job.

    Steps:
    1. Delete existing RAG records with the same board_id
    2. Call board export API to get parquet file S3 URL
    3. Read parquet data and create embeddings
    4. Save parquet metadata to parquet collection

    Args:
        job: Job details

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        organization_id = job["organization_id"]
        assistant_id = job["assistant_id"]
        board_id = job["board_id"]

        log.info(
            f"Processing board embedding for board_id: {board_id}, assistant_id: {assistant_id}"
        )

        # Step 1: Delete existing RAG records with the same board_id
        await _delete_existing_rag_records(board_id)

        # Step 2: Call board export API to get parquet file S3 URL
        parquet_s3_url, api_error = await _get_board_parquet_s3_url(board_id)

        if not parquet_s3_url:
            error_msg = (
                api_error or f"Failed to get parquet S3 URL for board {board_id}"
            )
            log.error(error_msg)
            return False, error_msg

        # Step 3: Read parquet data and create embeddings
        embedding_results = await _process_parquet_and_create_embeddings(
            parquet_s3_url, board_id, assistant_id, organization_id
        )

        if not embedding_results:
            error_msg = (
                f"Failed to process parquet and create embeddings for board {board_id}"
            )
            log.error(error_msg)
            return False, error_msg

        # Step 4: Save parquet metadata to parquet collection
        # Convert HTTPS URL to S3 path format for storage
        if parquet_s3_url.startswith("https://"):
            # Extract bucket and path from HTTPS URL
            # Example: https://your-bucket.s3.ap-east-1.amazonaws.com/board/org_imbrace/embedding_data/file.parquet
            # Should become: s3://your-bucket/board/org_imbrace/embedding_data/file.parquet
            url_parts = parquet_s3_url.replace("https://", "").split("/")
            bucket = url_parts[0].split(".")[
                0
            ]  # your-bucket from your-bucket.s3.ap-east-1.amazonaws.com
            s3_path = "/".join(
                url_parts[1:]
            )  # board/org_imbrace/embedding_data/file.parquet
            s3_storage_path = f"s3://{bucket}/{s3_path}"
            log.info(f"Converted HTTPS URL to S3 path for storage: {s3_storage_path}")
        else:
            s3_storage_path = parquet_s3_url

        await _save_board_to_collection(
            {"board_name": "Board from API"},
            board_id,
            assistant_id,
            organization_id,
            embedding_results,
            s3_storage_path,  # Pass the converted S3 path
        )

        log.info(f"Successfully processed board embedding job for board {board_id}")
        return True, None

    except Exception as e:
        error_msg = f"Error processing board embedding job: {str(e)}"
        log.error(error_msg)
        return False, error_msg


async def _delete_existing_rag_records(board_id: str) -> None:
    """
    Delete existing RAG records with the same board_id.

    Args:
        board_id: Board ID to delete records for
    """
    import os
    db_type = os.getenv("DB_TYPE", "mongodb").lower()

    try:
        if db_type == "postgresql":
            import asyncpg
            database_url = os.getenv("DATABASE_URL", "")
            conn = await asyncpg.connect(database_url)
            try:
                result = await conn.execute("DELETE FROM rag WHERE board_id = $1", board_id)
                deleted = int(result.split()[-1]) if result else 0
                log.info(f"Deleted {deleted} existing RAG records for board {board_id}")
            finally:
                await conn.close()
        else:
            from open_webui.internal.mongo_db import mongodb_client
            from open_webui.config import MONGODB_CONFIG

            db_name = MONGODB_CONFIG.get("openai_db_name", "openai_db")
            collection_name = "rag"

            query = {"board_id": board_id}
            result = await mongodb_client.delete_many(db_name, collection_name, query)
            log.info(
                f"Deleted {result.get('deleted_count', 0)} existing RAG records for board {board_id}"
            )

    except Exception as e:
        log.error(f"Error deleting existing RAG records for board {board_id}: {e}")
        raise


async def _get_board_parquet_s3_url(
    board_id: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Call the board export API to get parquet file S3 URL for a board.

    API: POST /v1/board/:boardId/export_embedding_data
    Response: {
        "message": "Embedding data export completed successfully",
        "file_name": "embedding_data_...",
        "download_url": "https://...",
        "record_count": 92,
        "board_id": "brd_...",
        "board_name": "..."
    }

    Args:
        board_id: Board ID

    Returns:
        Tuple of (S3 URL of the parquet file, error_message)
        If successful: (url, None)
        If failed: (None, error_message)
    """
    try:
        log.info(f"🔍 Starting board export API call for board {board_id}")

        # Import required modules
        import os
        import aiohttp
        import json

        # Get databoard host from environment
        backend_host = os.getenv("DATABOARD_HOST", "http://localhost:8081")
        log.info(f"📡 DATABOARD_HOST: {backend_host}")

        if not backend_host:
            error_msg = "DATABOARD_HOST environment variable not set"
            log.error(f"❌ {error_msg}")
            return None, error_msg

        # Build API URL
        api_url = f"{backend_host}/api/boards/{board_id}/export-embedding"
        log.info(f"🌐 API URL: {api_url}")

        # Make API call with proper headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-organization-id": "org_imbrace",  # Add organization header
        }
        log.info(f"📋 Request headers: {headers}")
        log.info("📤 Sending POST request with empty JSON body: {}")

        async with aiohttp.ClientSession(headers=headers) as session:
            log.info(f"⏳ Making HTTP request to {api_url}")
            async with session.post(api_url, json={}) as response:
                log.info(f"📊 Response status: {response.status}")
                log.info(f"📊 Response headers: {dict(response.headers)}")

                # Get response text regardless of status
                response_text = await response.text()
                log.info(f"📄 Raw response text: {response_text}")

                if response.status != 200:
                    error_msg = f"API call failed with status {response.status}: {response_text}"
                    log.error(f"❌ {error_msg}")

                    # Try to parse as JSON for better error details
                    try:
                        error_json = json.loads(response_text)
                        if "message" in error_json:
                            detailed_error = (
                                f"API Error {response.status}: {error_json['message']}"
                            )
                            log.error(f"🔍 Detailed error: {detailed_error}")
                            return None, detailed_error
                    except json.JSONDecodeError:
                        pass  # Response is not JSON, use raw text

                    return None, error_msg

                # Parse successful response
                try:
                    response_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse JSON response: {e}, raw response: {response_text}"
                    log.error(f"❌ {error_msg}")
                    return None, error_msg

                log.info(
                    f"✅ API response parsed successfully: {json.dumps(response_data, indent=2)}"
                )

                # Extract download URL
                download_url = response_data.get("download_url")
                if not download_url:
                    error_msg = "No download_url in API response"
                    log.error(f"❌ {error_msg}")
                    log.error(f"📋 Full response data: {response_data}")
                    return None, error_msg

                record_count = response_data.get("record_count", 0)
                board_name = response_data.get("board_name", "")
                file_name = response_data.get("file_name", "")

                log.info(f"🎯 Successfully got parquet URL for board {board_id}")
                log.info(f"📁 Board name: '{board_name}'")
                log.info(f"📊 Record count: {record_count}")
                log.info(f"📄 File name: '{file_name}'")
                log.info(f"🔗 Download URL: {download_url}")

                return download_url, None

    except aiohttp.ClientError as e:
        error_msg = (
            f"HTTP client error calling board export API for board {board_id}: {str(e)}"
        )
        log.error(f"❌ {error_msg}")
        return None, error_msg
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error for board {board_id}: {str(e)}"
        log.error(f"❌ {error_msg}")
        return None, error_msg
    except Exception as e:
        error_msg = (
            f"Unexpected error calling board export API for board {board_id}: {str(e)}"
        )
        log.error(f"❌ {error_msg}")
        import traceback

        traceback.print_exc()
        return None, error_msg


async def _process_board_items_and_create_embeddings(
    board_data: Dict[str, Any], board_id: str, assistant_id: str, organization_id: str
) -> Optional[Dict[str, Any]]:
    """
    Process board items and create embeddings using the RAG system.

    Args:
        board_data: Board data from MongoDB with items
        board_id: Board ID
        assistant_id: Assistant ID
        organization_id: Organization ID

    Returns:
        Embedding results or None if failed
    """
    try:
        log.info(f"Processing board items for board {board_id}")

        items = board_data.get("items", [])
        total_items = len(items)

        log.info(f"Processing {total_items} board items for embedding")

        if total_items == 0:
            log.warning(f"No items found in board {board_id}")
            return {
                "total_items": 0,
                "embedding_chunks_created": 0,
                "board_name": board_data.get("board_name", ""),
            }

        # Convert board items to text format for embedding
        log.info("Converting board items to text format for embedding...")

        # Create text representation of the board items
        text_documents = []

        # Process items in chunks to avoid memory issues
        chunk_size = 20  # Process 20 items at a time
        embedding_chunks_created = 0

        for i in range(0, total_items, chunk_size):
            chunk_items = items[i : i + chunk_size]
            chunk_text = f"Board: {board_data.get('board_name', board_id)}\n\n"

            # Convert each item to text
            for idx, item in enumerate(chunk_items):
                item_text = f"Item {i + idx + 1}: "

                # Extract field values from the item
                if "fields" in item and isinstance(item["fields"], list):
                    field_values = []
                    for field in item["fields"]:
                        field_name = field.get(
                            "board_field_name", field.get("name", "Unknown")
                        )
                        field_value = field.get("value", "")
                        if field_value:
                            field_values.append(f"{field_name}={field_value}")
                    item_text += ", ".join(field_values)
                else:
                    # Fallback: convert the entire item to string representation
                    item_text += str(item)

                chunk_text += item_text + "\n"

            # Create document for this chunk
            from langchain.schema import Document
            import uuid
            from datetime import datetime

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "id": str(uuid.uuid4()),
                    "org_id": organization_id,
                    "assistant_id": assistant_id,
                    "board_id": board_id,
                    "file_id": f"board-{board_id}-{i//chunk_size}",  # Unique file ID for this chunk
                    "original_name": f"board-{board_id}-chunk-{i//chunk_size}.json",
                    "bytes": len(chunk_text.encode("utf-8")),
                    "created_at": datetime.utcnow().isoformat(),
                    "boarditem_id": "",
                    "folder_id": "",
                    "url": f"mongodb://board/{board_id}",
                    "key": f"board-{board_id}-{i//chunk_size}",
                    "source": "board_embedding_job",
                    "chunk_index": i // chunk_size,
                    "total_items_in_chunk": len(chunk_items),
                },
            )

            text_documents.append(doc)
            embedding_chunks_created += 1

        log.info(f"Created {len(text_documents)} text documents for embedding")

        # Create embeddings using the RAG system
        if text_documents:
            log.info("Creating embeddings for board items...")

            from open_webui.internal.vector_store import afrom_documents
            from open_webui.llm.agent import get_llm_provider, EMBEDDING_PROVIDER, CONFIG

            # Follow LLM_EMBEDDING_PROVIDER so board embeddings match the RAG
            # ingest/query paths (unified provider factory; no RAG_EMBEDDING_TYPE).
            embeddings = get_llm_provider(
                EMBEDDING_PROVIDER or "openai", CONFIG
            ).create_embedding_model()

            log.info(f"Storing {len(text_documents)} document chunks in vector database...")
            await afrom_documents(text_documents, embeddings)
            log.info(
                f"Successfully created embeddings for {total_items} items in {embedding_chunks_created} chunks"
            )

        return {
            "total_items": total_items,
            "embedding_chunks_created": embedding_chunks_created,
            "board_name": board_data.get("board_name", ""),
            "documents_embedded": len(text_documents),
        }

    except Exception as e:
        log.error(f"Error processing board items and creating embeddings: {e}")
        import traceback

        traceback.print_exc()
        return None


async def _process_parquet_and_create_embeddings(
    parquet_s3_url: str, board_id: str, assistant_id: str, organization_id: str
) -> Optional[Dict[str, Any]]:
    """
    Read parquet data from S3 or local backend and create embeddings using the RAG system.

    Handles two storage cases:
    1. S3 Storage: Direct S3 URLs (https://bucket.s3.region.amazonaws.com/...)
    2. Local Storage: Backend API URLs (https://domain/api/v1/backend/files/...)

    Args:
        parquet_s3_url: URL of the parquet file (S3 or backend API)
        board_id: Board ID
        assistant_id: Assistant ID
        organization_id: Organization ID

    Returns:
        Embedding results or None if failed
    """
    try:
        log.info(f"Processing parquet file from {parquet_s3_url}")

        # Detect storage type: S3 direct URL vs Backend API URL
        is_backend_api_url = (
            "/api/v1/backend/files/" in parquet_s3_url
            or "/api/v1/files/" in parquet_s3_url
        )

        if is_backend_api_url:
            # Case 2: Local storage - Download via HTTP first
            log.info(f"Detected backend API URL, downloading parquet file via HTTP...")

            import aiohttp
            import tempfile
            import os

            # Download the parquet file
            async with aiohttp.ClientSession() as session:
                async with session.get(parquet_s3_url) as response:
                    if response.status != 200:
                        error_msg = (
                            f"Failed to download parquet file: HTTP {response.status}"
                        )
                        log.error(error_msg)
                        return None

                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".parquet"
                    ) as tmp_file:
                        tmp_file.write(await response.read())
                        tmp_file_path = tmp_file.name
                        log.info(
                            f"Downloaded parquet file to temporary location: {tmp_file_path}"
                        )

            try:
                # Read parquet file using pandas
                log.info(f"Reading parquet data from temporary file...")
                df = pd.read_parquet(tmp_file_path)
                log.info(f"Successfully loaded {len(df)} records from parquet file")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                    log.info("Cleaned up temporary parquet file")
                except Exception as cleanup_error:
                    log.warning(f"Failed to cleanup temporary file: {cleanup_error}")
        else:
            # Case 1: S3 storage - Use DuckDB with S3 credentials
            log.info(f"Detected S3 URL, using DuckDB S3 client...")

            # Initialize DuckDB client for parquet processing
            from open_webui.retrieval.tabular.duckdb import DuckDBClient

            duckdb_client = DuckDBClient(in_memory=True)
            try:
                # Convert HTTPS URL to S3 path for DuckDB
                # URL format: https://bucket.s3.region.amazonaws.com/path/to/file
                # S3 path format: s3://bucket/path/to/file
                if parquet_s3_url.startswith("https://"):
                    # Extract bucket and path from HTTPS URL
                    # Example: https://your-bucket.s3.ap-east-1.amazonaws.com/board/org_imbrace/embedding_data/file.parquet
                    # Should become: s3://your-bucket/board/org_imbrace/embedding_data/file.parquet
                    url_parts = parquet_s3_url.replace("https://", "").split("/")
                    bucket = url_parts[0].split(".")[
                        0
                    ]  # your-bucket from your-bucket.s3.ap-east-1.amazonaws.com
                    s3_path = "/".join(
                        url_parts[1:]
                    )  # board/org_imbrace/embedding_data/file.parquet
                    duckdb_s3_path = f"s3://{bucket}/{s3_path}"
                    log.info(f"Converted HTTPS URL to S3 path: {duckdb_s3_path}")
                else:
                    duckdb_s3_path = parquet_s3_url

                # Read parquet file from S3
                log.info(f"Reading parquet data from {duckdb_s3_path}")
                df = duckdb_client.read_file_from_s3(
                    duckdb_s3_path, file_type="parquet"
                )
            finally:
                # Clean up DuckDB connection
                if hasattr(duckdb_client, "conn"):
                    duckdb_client.conn.close()

        # Common processing for both S3 and local storage
        total_records = len(df)
        log.info(f"Loaded {total_records} records from parquet file")

        if total_records == 0:
            log.warning(f"No records found in parquet file for board {board_id}")
            return {
                "total_records": 0,
                "embedding_chunks_created": 0,
                "s3_url": parquet_s3_url,
            }

        # Convert DataFrame to text format for embedding
        log.info("Converting tabular data to text format for embedding...")

        # Create text representation of the data
        text_documents = []

        # Define metadata columns that shouldn't be embedded
        metadata_columns = {
            "_id",
            "board_id",
            "organization_id",
            "created_at",
            "updated_at",
        }
        embeddable_columns = [col for col in df.columns if col not in metadata_columns]

        log.info(f"Metadata columns (excluded): {list(metadata_columns)}")
        log.info(f"Embeddable columns: {embeddable_columns}")

        # Process each record individually for better granularity
        embedding_chunks_created = 0

        for idx, row in df.iterrows():
            # Extract metadata
            boarditem_id = str(row.get("_id", ""))

            # Create text content from embeddable fields only
            row_values = []
            for col in embeddable_columns:
                value = str(row[col]) if pd.notna(row[col]) else ""
                if value.strip():  # Only include non-empty values
                    row_values.append(f"{col}: {value}")

            # Skip empty records
            if not row_values:
                log.warning(
                    f"Skipping empty record {idx + 1} (boarditem_id: {boarditem_id})"
                )
                continue

            record_text = "\n".join(row_values)

            # Create document for this individual record
            from langchain.schema import Document
            import uuid
            from datetime import datetime

            doc = Document(
                page_content=record_text,
                metadata={
                    "id": str(uuid.uuid4()),
                    "org_id": organization_id,
                    "assistant_id": assistant_id,
                    "board_id": board_id,
                    "file_id": f"board-{board_id}-record-{idx}",  # Unique file ID for this record
                    "original_name": f"board-{board_id}-record-{idx}.parquet",
                    "bytes": len(record_text.encode("utf-8")),
                    "created_at": datetime.utcnow().isoformat(),
                    "boarditem_id": boarditem_id,  # Individual board item ID
                    "folder_id": "",
                    "url": parquet_s3_url,
                    "key": f"board-{board_id}-record-{idx}",
                    "source": "board_embedding_job",
                    "record_index": idx,
                    "total_embeddable_fields": len(row_values),
                },
            )

            text_documents.append(doc)
            embedding_chunks_created += 1

            # Log first few records for debugging
            if idx < 3:
                log.info(f"Sample record {idx + 1} content: {record_text[:200]}...")

        log.info(f"Created {len(text_documents)} text documents for embedding")

        # Create embeddings using the RAG system with batching
        if text_documents:
            log.info("Creating embeddings for board data with batch processing...")

            from open_webui.internal.vector_store import afrom_documents
            from open_webui.llm.agent import get_llm_provider, EMBEDDING_PROVIDER, CONFIG

            # Follow LLM_EMBEDDING_PROVIDER so board embeddings match the RAG
            # ingest/query paths (unified provider factory; no RAG_EMBEDDING_TYPE).
            embeddings = get_llm_provider(
                EMBEDDING_PROVIDER or "openai", CONFIG
            ).create_embedding_model()

            log.info(
                f"Storing {len(text_documents)} document chunks in vector database with batching..."
            )

            batch_size = 500
            total_processed = 0
            for i in range(0, len(text_documents), batch_size):
                batch = text_documents[i : i + batch_size]
                batch_num = i // batch_size + 1
                log.info(f"Processing embedding batch {batch_num} ({len(batch)} docs)...")
                await afrom_documents(batch, embeddings)
                total_processed += len(batch)
                log.info(f"Completed batch {batch_num}, total processed: {total_processed}")

            log.info(
                f"Successfully created embeddings for {total_records} records in {embedding_chunks_created} chunks"
            )

        return {
            "total_records": total_records,
            "embedding_chunks_created": embedding_chunks_created,
            "s3_url": parquet_s3_url,
            "columns": list(df.columns) if hasattr(df, "columns") else [],
            "documents_embedded": len(text_documents),
        }

    except Exception as e:
        log.error(f"Error processing parquet and creating embeddings: {e}")
        import traceback

        traceback.print_exc()
        return None


async def _process_jsonl_and_create_embeddings(
    jsonl_s3_url: str, board_id: str, assistant_id: str, organization_id: str
) -> Optional[Dict[str, Any]]:
    """
    Download JSONL file and create embeddings using the RAG system.

    JSONL format:
    {
        "_id": "bi_db989000-4726-4b51-a59a-690f6da9a263",
        "board_id": "brd_7f8bd42f-dcba-44c2-b166-76cda6083850",
        "organization_id": "org_imbrace",
        "fields_data": {
            "Name": "WhatsApp Outbound",
            "Category": "Journey",
            "Description": "...",
            "Link": "..."
        }
    }

    Args:
        jsonl_s3_url: S3 URL of the JSONL file
        board_id: Board ID
        assistant_id: Assistant ID
        organization_id: Organization ID

    Returns:
        Embedding results or None if failed
    """
    try:
        log.info(f"Processing JSONL file from {jsonl_s3_url}")

        # Import required modules
        import aiohttp
        import json

        # Download JSONL file from S3
        log.info(f"Downloading JSONL file from {jsonl_s3_url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(jsonl_s3_url) as response:
                if response.status != 200:
                    log.error(f"Failed to download JSONL file: HTTP {response.status}")
                    return None

                content = await response.text()
                lines = content.strip().split("\n")

        # Parse JSONL data
        board_items = []
        for line_num, line in enumerate(lines):
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    board_items.append(item)
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse JSON line {line_num}: {e}")
                    continue

        total_items = len(board_items)
        log.info(f"Loaded {total_items} items from JSONL file")

        if total_items == 0:
            log.warning(f"No items found in JSONL file for board {board_id}")
            return {
                "total_items": 0,
                "embedding_chunks_created": 0,
                "s3_url": jsonl_s3_url,
            }

        # Convert board items to text format for embedding
        log.info("Converting board items to text format for embedding...")

        # Create text representation of the board items
        text_documents = []

        # Process items in chunks to avoid memory issues
        chunk_size = 20  # Process 20 items at a time
        embedding_chunks_created = 0

        for i in range(0, total_items, chunk_size):
            chunk_items = board_items[i : i + chunk_size]
            chunk_text = f"Board Items:\n\n"

            # Convert each item to text
            for idx, item in enumerate(chunk_items):
                item_text = f"Item {i + idx + 1}: "

                # Use fields_data for the actual content to embed
                fields_data = item.get("fields_data", {})

                if fields_data:
                    field_values = []
                    for field_name, field_value in fields_data.items():
                        if field_value:
                            field_values.append(f"{field_name}={field_value}")
                    item_text += ", ".join(field_values)
                else:
                    # Fallback: convert the entire item to string representation
                    item_text += str(item)

                chunk_text += item_text + "\n"

            # Create document for this chunk
            from langchain.schema import Document
            import uuid
            from datetime import datetime

            doc = Document(
                page_content=chunk_text,
                metadata={
                    "id": str(uuid.uuid4()),
                    "org_id": organization_id,
                    "assistant_id": assistant_id,
                    "board_id": board_id,
                    "file_id": f"board-{board_id}-{i//chunk_size}",  # Unique file ID for this chunk
                    "original_name": f"board-{board_id}-chunk-{i//chunk_size}.jsonl",
                    "bytes": len(chunk_text.encode("utf-8")),
                    "created_at": datetime.utcnow().isoformat(),
                    "boarditem_id": item.get("_id", "") if chunk_items else "",
                    "folder_id": "",
                    "url": jsonl_s3_url,
                    "key": f"board-{board_id}-{i//chunk_size}",
                    "source": "board_embedding_job",
                    "chunk_index": i // chunk_size,
                    "total_items_in_chunk": len(chunk_items),
                },
            )

            text_documents.append(doc)
            embedding_chunks_created += 1

        log.info(f"Created {len(text_documents)} text documents for embedding")

        # Create embeddings using the RAG system
        if text_documents:
            log.info("Creating embeddings for board items...")

            from open_webui.internal.vector_store import afrom_documents
            from open_webui.llm.agent import get_llm_provider, EMBEDDING_PROVIDER, CONFIG

            # Follow LLM_EMBEDDING_PROVIDER so board embeddings match the RAG
            # ingest/query paths (unified provider factory; no RAG_EMBEDDING_TYPE).
            embeddings = get_llm_provider(
                EMBEDDING_PROVIDER or "openai", CONFIG
            ).create_embedding_model()

            log.info(f"Storing {len(text_documents)} document chunks in vector database...")
            await afrom_documents(text_documents, embeddings)
            log.info(
                f"Successfully created embeddings for {total_items} items in {embedding_chunks_created} chunks"
            )

        return {
            "total_items": total_items,
            "embedding_chunks_created": embedding_chunks_created,
            "s3_url": jsonl_s3_url,
            "documents_embedded": len(text_documents),
        }

    except Exception as e:
        log.error(f"Error processing JSONL and creating embeddings: {e}")
        import traceback

        traceback.print_exc()
        return None


async def _save_board_to_collection(
    board_data: Dict[str, Any],
    board_id: str,
    assistant_id: str,
    organization_id: str,
    embedding_results: Dict[str, Any],
    parquet_s3_url: str,
) -> None:
    """
    Save board embedding information to the parquet collection.

    Args:
        board_data: Board data from MongoDB
        board_id: Board ID
        assistant_id: Assistant ID
        organization_id: Organization ID
        embedding_results: Results from embedding processing
        parquet_s3_url: The actual S3 URL where the parquet file is stored
    """
    try:
        from open_webui.models.ai_agents.parquets import create as create_parquet
        import uuid
        from datetime import datetime

        # Create board embedding record
        board_details = {
            "id": str(uuid.uuid4()),
            "organization_id": organization_id,
            "assistant_id": assistant_id,
            "board_id": board_id,
            "s3_path": parquet_s3_url,  # Use the actual S3 URL of the parquet file
            "total_records": embedding_results.get("total_records", 0),
            "embedding_chunks": embedding_results.get("embedding_chunks_created", 0),
            "status": "completed",
            "processed_at": datetime.utcnow().isoformat(),
        }

        created_record = await create_parquet(board_details)
        if created_record:
            log.info(
                f"Saved board embedding record {board_details['id']} for board {board_id}"
            )
            log.info(f"Parquet S3 URL: {parquet_s3_url}")
        else:
            log.error(f"Failed to save board embedding record for board {board_id}")

    except Exception as e:
        log.error(f"Error saving board embedding to collection: {e}")
        raise


async def _mark_board_as_embed(board_id: str) -> None:
    """
    Mark a board with is_embed=true so the embedding pipeline knows to process it.
    Does NOT change the board's type — the original type is preserved.

    API: PUT /v1/board/:boardId
    Body: {"is_embed": true}
    """
    try:
        log.info(f"🔄 Marking board {board_id} as is_embed=true...")

        import os
        import aiohttp
        import json

        # Get databoard host from environment
        backend_host = os.getenv("DATABOARD_HOST", "http://localhost:8081")
        if not backend_host:
            error_msg = "DATABOARD_HOST environment variable not set"
            log.error(f"❌ {error_msg}")
            raise Exception(error_msg)

        # Build API URL
        api_url = f"{backend_host}/api/boards/{board_id}"
        log.info(f"🌐 Board update API URL: {api_url}")

        request_data = {"is_embed": True}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-organization-id": "org_imbrace",
        }

        log.info(f"📋 Request headers: {headers}")
        log.info(f"📤 Request body: {json.dumps(request_data)}")

        async with aiohttp.ClientSession(headers=headers) as session:
            log.info(f"⏳ Making PUT request to {api_url}")
            async with session.put(api_url, json=request_data) as response:
                log.info(f"📊 Response status: {response.status}")
                log.info(f"📊 Response headers: {dict(response.headers)}")

                response_text = await response.text()
                log.info(f"📄 Raw response text: {response_text}")

                if response.status not in [200, 204]:
                    error_msg = f"Failed to mark board as is_embed: HTTP {response.status}: {response_text}"
                    log.error(f"❌ {error_msg}")

                    try:
                        error_json = json.loads(response_text)
                        if "message" in error_json:
                            detailed_error = (
                                f"API Error {response.status}: {error_json['message']}"
                            )
                            log.error(f"🔍 Detailed error: {detailed_error}")
                            raise Exception(detailed_error)
                    except json.JSONDecodeError:
                        pass

                    raise Exception(error_msg)

                log.info(f"✅ Successfully marked board {board_id} as is_embed=true")

    except aiohttp.ClientError as e:
        error_msg = f"HTTP client error marking board {board_id} as is_embed: {str(e)}"
        log.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error marking board {board_id} as is_embed: {str(e)}"
        log.error(f"❌ {error_msg}")
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error marking board {board_id} as is_embed: {str(e)}"
        log.error(f"❌ {error_msg}")
        raise


# Export functions
__all__ = [
    "create_board_embedding_jobs_for_assistant",
    "process_pending_board_embedding_job",
]
