import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import MONGODB_CONFIG
from open_webui.internal.mongo_db import mongodb_client

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("BOARD_EMBEDDING_JOB", logging.INFO))

# Configuration from MongoDB config
DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")
COLLECTION_NAME = "boardEmbeddingJob"


class JobStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    ERROR = "error"


class BoardEmbeddingJobModel:
    """Model for managing board embedding jobs in MongoDB."""

    def __init__(self):
        self.db_name = DB_NAME
        self.collection_name = COLLECTION_NAME

    async def create(self, job_details: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new board embedding job."""
        try:
            # Add timestamps and defaults
            now = datetime.utcnow().isoformat()
            job_details["created_at"] = now
            job_details["updated_at"] = now
            job_details["status"] = job_details.get("status", JobStatus.PENDING.value)
            job_details["retry_count"] = job_details.get("retry_count", 0)
            job_details["max_retries"] = job_details.get("max_retries", 3)
            job_details["deleted_at"] = None

            result = await mongodb_client.insert(
                self.db_name, self.collection_name, job_details
            )

            log.info(
                f"Created board embedding job with ID: {job_details.get('id', 'unknown')}"
            )
            return result

        except Exception as error:
            log.error(f"Error creating board embedding job: {error}")
            raise error

    async def get_by_filters(
        self,
        organization_id: Optional[str] = None,
        board_id: Optional[str] = None,
        assistant_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get board embedding jobs by filters."""
        try:
            query = {"deleted_at": None}

            if organization_id:
                query["organization_id"] = organization_id
            if board_id:
                query["board_id"] = board_id
            if assistant_id:
                query["assistant_id"] = assistant_id
            if status:
                query["status"] = status

            result = await mongodb_client.query(
                self.db_name, self.collection_name, query
            )

            log.info(f"Retrieved {len(result)} board embedding jobs matching filters")
            return result

        except Exception as error:
            log.error(f"Error getting board embedding jobs by filters: {error}")
            raise error

    async def get_by_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a single board embedding job by ID."""
        try:
            query = {
                "id": job_id,
                "deleted_at": None,
            }

            result = await mongodb_client.query_one(
                self.db_name, self.collection_name, query
            )

            if result:
                log.info(f"Found board embedding job for ID: {job_id}")
                return result
            else:
                log.info(f"No board embedding job found for ID: {job_id}")
                return None

        except Exception as error:
            log.error(f"Error getting board embedding job by ID: {error}")
            raise error

    async def update(
        self, job_id: str, update_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Update a board embedding job."""
        try:
            query = {
                "id": job_id,
                "deleted_at": None,
            }

            # Add updated_at timestamp
            update_data["updated_at"] = datetime.utcnow().isoformat()

            result = await mongodb_client.update(
                self.db_name, self.collection_name, query, update_data
            )

            log.info(f"Updated board embedding job for ID: {job_id}")
            return result

        except Exception as error:
            log.error(f"Error updating board embedding job: {error}")
            raise error

    async def get_pending_jobs(self, limit: int = 1) -> List[Dict[str, Any]]:
        """Get pending jobs, ordered by creation time."""
        try:
            query = {
                "status": JobStatus.PENDING.value,
                "deleted_at": None,
                "retry_count": {"$lt": 3},  # Less than max retries
            }

            # Get jobs ordered by creation time (oldest first)
            result = await mongodb_client.query_with_sort_and_pagination(
                self.db_name,
                self.collection_name,
                query,
                sort_field="created_at",
                sort_order=1,  # Ascending (oldest first)
                limit=limit,
            )

            log.info(
                f"Retrieved {len(result.get('data', []))} pending board embedding jobs"
            )
            return result.get("data", [])

        except Exception as error:
            log.error(f"Error getting pending board embedding jobs: {error}")
            raise error

    async def mark_job_started(self, job_id: str) -> bool:
        """Mark a job as in progress."""
        try:
            update_data = {
                "status": JobStatus.IN_PROGRESS.value,
                "started_at": datetime.utcnow().isoformat(),
            }
            result = await self.update(job_id, update_data)
            return result is not None
        except Exception as error:
            log.error(f"Error marking job {job_id} as started: {error}")
            return False

    async def mark_job_completed(self, job_id: str) -> bool:
        """Mark a job as completed."""
        try:
            update_data = {
                "status": JobStatus.DONE.value,
                "completed_at": datetime.utcnow().isoformat(),
            }
            result = await self.update(job_id, update_data)
            return result is not None
        except Exception as error:
            log.error(f"Error marking job {job_id} as completed: {error}")
            return False

    async def mark_job_failed(self, job_id: str, error_message: str = None) -> bool:
        """Mark a job as failed and increment retry count."""
        try:
            # First get current job to check retry count
            job = await self.get_by_id(job_id)
            if not job:
                return False

            current_retries = job.get("retry_count", 0)
            new_retry_count = current_retries + 1

            update_data = {
                "status": (
                    JobStatus.PENDING.value
                    if new_retry_count < 3
                    else JobStatus.ERROR.value
                ),
                "retry_count": new_retry_count,
                "last_error": error_message,
                "failed_at": datetime.utcnow().isoformat(),
            }

            result = await self.update(job_id, update_data)
            return result is not None
        except Exception as error:
            log.error(f"Error marking job {job_id} as failed: {error}")
            return False

    async def soft_delete_by_board_id(self, board_id: str) -> int:
        """Soft-delete all jobs for a board. Returns the number of rows modified."""
        try:
            now = datetime.utcnow().isoformat()
            result = await mongodb_client.update_many(
                self.db_name,
                self.collection_name,
                {"board_id": board_id, "deleted_at": None},
                {"deleted_at": now, "updated_at": now},
            )
            modified = result.get("modified_count", 0) if result else 0
            log.info(
                f"Soft-deleted {modified} board embedding job(s) for board_id {board_id}"
            )
            return modified
        except Exception as error:
            log.error(
                f"Error soft-deleting board embedding jobs for board {board_id}: {error}"
            )
            return 0


# Create a global instance
board_embedding_job_model = BoardEmbeddingJobModel()


# Export convenience functions
async def create(job_details: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new board embedding job."""
    return await board_embedding_job_model.create(job_details)


async def get_by_filters(
    organization_id: Optional[str] = None,
    board_id: Optional[str] = None,
    assistant_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get board embedding jobs by filters."""
    return await board_embedding_job_model.get_by_filters(
        organization_id, board_id, assistant_id, status
    )


async def get_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a single board embedding job by ID."""
    return await board_embedding_job_model.get_by_id(job_id)


async def update(job_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a board embedding job."""
    return await board_embedding_job_model.update(job_id, update_data)


async def get_pending_jobs(limit: int = 1) -> List[Dict[str, Any]]:
    """Get pending jobs."""
    return await board_embedding_job_model.get_pending_jobs(limit)


async def mark_job_started(job_id: str) -> bool:
    """Mark a job as in progress."""
    return await board_embedding_job_model.mark_job_started(job_id)


async def mark_job_completed(job_id: str) -> bool:
    """Mark a job as completed."""
    return await board_embedding_job_model.mark_job_completed(job_id)


async def mark_job_failed(job_id: str, error_message: str = None) -> bool:
    """Mark a job as failed."""
    return await board_embedding_job_model.mark_job_failed(job_id, error_message)


async def soft_delete_by_board_id(board_id: str) -> int:
    """Soft-delete all jobs for a board."""
    return await board_embedding_job_model.soft_delete_by_board_id(board_id)


# Export all functions and the model class
__all__ = [
    "BoardEmbeddingJobModel",
    "board_embedding_job_model",
    "JobStatus",
    "create",
    "get_by_filters",
    "get_by_id",
    "update",
    "get_pending_jobs",
    "mark_job_started",
    "mark_job_completed",
    "mark_job_failed",
]
