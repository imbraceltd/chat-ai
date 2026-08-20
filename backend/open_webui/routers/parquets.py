import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, status, Query, Path
from pydantic import BaseModel

from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.ai_agents.parquets import (
    get_by_filters,
    get_by_id,
    update,
    delete,
    get_all_active,
)
from open_webui.repository.assistant import AssistantRepository
from open_webui.repository.parquet import parquet_repo

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("PARQUETS", logging.INFO))

router = APIRouter()

assistant_repo = AssistantRepository()


# Pydantic models
class ParquetResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[Dict[str, Any]]] = None


class ParquetSingleResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class DeleteResponse(BaseModel):
    success: bool
    message: str
    deleted_count: int


class UpdateParquetRequest(BaseModel):
    description: Optional[str] = None
    s3_path: Optional[str] = None
    # Add other updatable fields as needed


@router.get("", response_model=ParquetResponse)
async def get_parquets(
    organization_id: Optional[str] = Query(
        None, description="Filter by organization ID"
    ),
    assistant_id: Optional[str] = Query(None, description="Filter by assistant ID"),
    folder_ids: Optional[List[str]] = Query(None, description="Filter by folder IDs"),
    board_ids: Optional[List[str]] = Query(None, description="Filter by board IDs"),
    boarditem_id: Optional[str] = Query(None, description="Filter by board item ID"),
):
    """
    Get parquet files based on filters.
    All filters are optional and will be combined with AND logic.
    If assistant_id is provided along with organization_id, folder_ids and board_ids
    will be automatically populated from the assistant's configuration if not already specified.
    """
    try:
        # Enrich filters with assistant details if assistant_id and organization_id are provided
        if assistant_id and organization_id:
            try:
                assistant_details = (
                    await assistant_repo.get_by_assistant_id_and_organization_id(
                        organization_id, assistant_id
                    )
                )

                if assistant_details:
                    # Set folder_ids if not already provided
                    if (
                        assistant_details.get("folder_ids")
                        and len(assistant_details["folder_ids"]) > 0
                        and (not folder_ids or len(folder_ids) == 0)
                    ):
                        folder_ids = assistant_details["folder_ids"]

                    # Set board_ids if not already provided
                    if (
                        assistant_details.get("board_ids")
                        and len(assistant_details["board_ids"]) > 0
                        and (not board_ids or len(board_ids) == 0)
                    ):
                        board_ids = assistant_details["board_ids"]

            except Exception as enrich_error:
                log.warning(
                    f"Error enriching filters with assistant details: {enrich_error}"
                )

        result = await get_by_filters(
            organization_id=organization_id,
            folder_ids=folder_ids,
            board_ids=board_ids,
            boarditem_id=boarditem_id,
        )

        return ParquetResponse(
            success=True,
            message=f"Retrieved {len(result)} parquet records",
            data=result,
        )

    except Exception as error:
        log.error(f"Error getting parquet records: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving parquet records",
        )


@router.get("/{parquet_id}", response_model=ParquetSingleResponse)
async def get_parquet(
    parquet_id: str = Path(..., description="The parquet record ID"),
):
    """
    Get a single parquet record by ID.
    """
    try:
        result = await get_by_id(parquet_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parquet record not found",
            )

        return ParquetSingleResponse(
            success=True,
            message="Parquet record retrieved successfully",
            data=result,
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error getting parquet record {parquet_id}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving parquet record",
        )


@router.put("/{parquet_id}", response_model=ParquetSingleResponse)
async def update_parquet(
    update_data: UpdateParquetRequest,
    parquet_id: str = Path(..., description="The parquet record ID"),
    delete_old_s3_file: bool = Query(
        False, description="Whether to delete the old S3 file when s3_path is updated"
    ),
):
    """
    Update a parquet record by parquet_id.
    """
    try:
        # Convert Pydantic model to dict, excluding None values
        update_dict = update_data.dict(exclude_unset=True, exclude_none=True)

        if not update_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields provided for update",
            )

        # Convert HTTPS S3 URLs to S3 format for consistency
        if update_dict.get("s3_path") and update_dict["s3_path"].startswith("https://"):
            https_url = update_dict["s3_path"]
            # Extract bucket and path from HTTPS URL
            # Example: https://your-bucket.s3.ap-east-1.amazonaws.com/board/org_imbrace/embedding_data/file.parquet
            # Should become: s3://your-bucket/board/org_imbrace/embedding_data/file.parquet
            url_parts = https_url.replace("https://", "").split("/")
            if len(url_parts) >= 2:
                bucket = url_parts[0].split(".")[
                    0
                ]  # your-bucket from your-bucket.s3.ap-east-1.amazonaws.com
                s3_path = "/".join(
                    url_parts[1:]
                )  # board/org_imbrace/embedding_data/file.parquet
                update_dict["s3_path"] = f"s3://{bucket}/{s3_path}"
                log.info(
                    f"Converted HTTPS S3 URL to S3 format: {https_url} -> {update_dict['s3_path']}"
                )

        # Check if s3_path is being updated and delete_old_s3_file is requested
        if update_dict.get("s3_path") and delete_old_s3_file:
            # Get current record to get the old S3 path
            current_record = await get_by_id(parquet_id)
            if not current_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Parquet record not found",
                )

            old_s3_path = current_record.get("s3_path")
            if old_s3_path and old_s3_path != update_dict["s3_path"]:
                try:
                    from open_webui.storage.provider import Storage

                    log.info(f"Deleting old S3 file: {old_s3_path}")
                    Storage.delete_file(old_s3_path)
                    log.info(f"Successfully deleted old S3 file: {old_s3_path}")
                except Exception as s3_error:
                    log.warning(
                        f"Failed to delete old S3 file {old_s3_path}: {s3_error}"
                    )
                    # Continue with update even if S3 deletion fails

        result = await update(parquet_id, update_dict)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parquet record not found",
            )

        # Get updated record
        updated_record = await get_by_id(parquet_id)

        message = "Parquet record updated successfully"
        if update_dict.get("s3_path") and delete_old_s3_file:
            message += " (old S3 file deleted)"

        return ParquetSingleResponse(
            success=True,
            message=message,
            data=updated_record,
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error updating parquet record {parquet_id}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating parquet record",
        )


@router.put("/by-board/{board_id}", response_model=ParquetSingleResponse)
async def update_parquet_by_board(
    update_data: UpdateParquetRequest,
    board_id: str = Path(..., description="The board ID"),
    organization_id: str = Query(
        ..., description="Organization ID (required when using board_id)"
    ),
    delete_old_s3_file: bool = Query(
        False, description="Whether to delete the old S3 file when s3_path is updated"
    ),
):
    """
    Update a parquet record by board_id (one board can only have one parquet record).
    """
    try:
        # Convert Pydantic model to dict, excluding None values
        update_dict = update_data.dict(exclude_unset=True, exclude_none=True)

        if not update_dict:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid fields provided for update",
            )

        # Convert HTTPS S3 URLs to S3 format for consistency
        if update_dict.get("s3_path") and update_dict["s3_path"].startswith("https://"):
            https_url = update_dict["s3_path"]
            # Extract bucket and path from HTTPS URL
            # Example: https://your-bucket.s3.ap-east-1.amazonaws.com/board/org_imbrace/embedding_data/file.parquet
            # Should become: s3://your-bucket/board/org_imbrace/embedding_data/file.parquet
            url_parts = https_url.replace("https://", "").split("/")
            if len(url_parts) >= 2:
                bucket = url_parts[0].split(".")[
                    0
                ]  # your-bucket from your-bucket.s3.ap-east-1.amazonaws.com
                s3_path = "/".join(
                    url_parts[1:]
                )  # board/org_imbrace/embedding_data/file.parquet
                update_dict["s3_path"] = f"s3://{bucket}/{s3_path}"
                log.info(
                    f"Converted HTTPS S3 URL to S3 format: {https_url} -> {update_dict['s3_path']}"
                )

        # Find the parquet record for this board
        parquet_records = await get_by_filters(
            organization_id=organization_id,
            board_ids=[board_id],
        )

        if not parquet_records:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No parquet record found for board {board_id}",
            )

        if len(parquet_records) > 1:
            log.warning(
                f"Multiple parquet records found for board {board_id}, using the first one"
            )

        parquet_record = parquet_records[0]
        parquet_id = parquet_record["id"]

        # Check if s3_path is being updated and delete_old_s3_file is requested
        if update_dict.get("s3_path") and delete_old_s3_file:
            old_s3_path = parquet_record.get("s3_path")
            if old_s3_path and old_s3_path != update_dict["s3_path"]:
                try:
                    from open_webui.storage.provider import Storage

                    log.info(f"Deleting old S3 file: {old_s3_path}")
                    Storage.delete_file(old_s3_path)
                    log.info(f"Successfully deleted old S3 file: {old_s3_path}")
                except Exception as s3_error:
                    log.warning(
                        f"Failed to delete old S3 file {old_s3_path}: {s3_error}"
                    )
                    # Continue with update even if S3 deletion fails

        result = await update(parquet_id, update_dict)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parquet record not found",
            )

        # Get updated record
        updated_record = await get_by_id(parquet_id)

        message = "Parquet record updated successfully"
        if update_dict.get("s3_path") and delete_old_s3_file:
            message += " (old S3 file deleted)"

        return ParquetSingleResponse(
            success=True,
            message=message,
            data=updated_record,
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error updating parquet record for board {board_id}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating parquet record",
        )


@router.delete("/{parquet_id}", response_model=DeleteResponse)
async def delete_parquet(
    parquet_id: str = Path(..., description="The parquet record ID"),
    delete_s3_file: bool = Query(
        True, description="Whether to also delete the S3 file"
    ),
):
    """
    Delete a parquet record and optionally its S3 file.
    """
    try:
        result = await delete(parquet_id, delete_s3_file)

        if result.get("deleted_count", 0) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parquet record not found",
            )

        return DeleteResponse(
            success=True,
            message=f"Parquet record deleted successfully{' (including S3 file)' if delete_s3_file else ''}",
            deleted_count=result.get("deleted_count", 0),
        )

    except HTTPException:
        raise
    except Exception as error:
        log.error(f"Error deleting parquet record {parquet_id}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting parquet record",
        )


@router.get("/all/active", response_model=ParquetResponse)
async def get_all_active_parquets():
    """
    Get all active (non-deleted) parquet records.
    """
    try:
        result = await get_all_active()

        return ParquetResponse(
            success=True,
            message=f"Retrieved {len(result)} active parquet records",
            data=result,
        )

    except Exception as error:
        log.error(f"Error getting all active parquet records: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving active parquet records",
        )


class ParquetCleanupRequest(BaseModel):
    folder_id: str
    organization_id: Optional[str] = None
    # If provided, only parquets whose file_name is in this list are removed;
    # otherwise every parquet in the folder is removed.
    file_names: Optional[List[str]] = None


class ParquetCleanupResponse(BaseModel):
    success: bool
    message: str
    deletedCount: int


@router.post("/cleanup", response_model=ParquetCleanupResponse)
async def cleanup_parquets(body: ParquetCleanupRequest):
    """
    Soft-delete tabular parquet records for a folder (optionally scoped to
    specific file_names). Called by data-board when a knowledge-hub file or
    folder is deleted, so the agent's duckdb_query tool stops reading stale
    rows. Only the DB rows are soft-deleted (deleted_at); S3 files are left
    in place (harmless, not queried once deleted_at is set).
    """
    try:
        rows = await parquet_repo.get_by_filters(
            organization_id=body.organization_id,
            folder_ids=[body.folder_id],
        )
        if body.file_names:
            wanted = set(body.file_names)
            rows = [r for r in rows if r.get("file_name") in wanted]

        deleted = 0
        for r in rows:
            deleted += await parquet_repo.soft_delete(r["id"])

        return ParquetCleanupResponse(
            success=True,
            message=f"Soft-deleted {deleted} parquet record(s) for folder {body.folder_id}",
            deletedCount=deleted,
        )
    except Exception as error:
        log.error(f"Error cleaning up parquets for folder {body.folder_id}: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error cleaning up parquet records",
        )


# Export the router
__all__ = ["router"]
