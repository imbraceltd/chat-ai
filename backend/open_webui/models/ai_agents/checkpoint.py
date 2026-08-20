import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import MONGODB_CONFIG
from open_webui.internal.mongo_db import mongodb_client

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("CHECKPOINT", logging.INFO))

# Configuration from MongoDB config
DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")
COLLECTION_NAME = "checkpoints_aio"
WRITE_COLLECTION_NAME = "checkpoint_writes_aio"


def convert_timestamp_to_date(timestamp: Optional[int] = None) -> str:
    """Convert timestamp to ISO date string."""
    if timestamp is None:
        timestamp = int(datetime.utcnow().timestamp())
    
    return datetime.fromtimestamp(timestamp).isoformat()


class CheckpointWritesAioModel:
    """Python model for MongoDB checkpoint write operations with async support."""
    
    def __init__(self):
        self.db_name = DB_NAME
        self.collection_name = COLLECTION_NAME
    
    async def create(self, checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new checkpoint record."""
        try:
            created_at = convert_timestamp_to_date(checkpoint_data.get("created_at"))
            
            # Prepare inputs with checkpoint-specific fields
            inputs = {
                **checkpoint_data,
                "checkpoint_id": "",
                "thread_id": "",
                "checkpoint_ns": "",
                "checkpoint_data": {},
                "metadata": {},
                "updated_at": "",
                "created_at": "",
            }
            
            # Set the specific fields
            inputs["checkpoint_id"] = checkpoint_data.get("id", "")
            inputs["thread_id"] = checkpoint_data.get("thread_id", "")
            inputs["checkpoint_ns"] = checkpoint_data.get("checkpoint_ns", "")
            inputs["checkpoint_data"] = checkpoint_data.get("checkpoint_data", {})
            inputs["metadata"] = checkpoint_data.get("metadata", {})
            inputs["updated_at"] = created_at
            inputs["created_at"] = created_at
            
            result = await mongodb_client.insert(
                self.db_name,
                self.collection_name,
                inputs
            )
            
            log.info(f"Created checkpoint record with checkpoint_id: {inputs['checkpoint_id']}")
            return result
            
        except Exception as error:
            log.error(f"Error creating checkpoint record: {error}")
            raise error
    
    async def get_all_by_thread(self, thread_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all checkpoints for a specific thread."""
        try:
            query = {
                "thread_id": thread_id,
            }
            
            # Sort by created_at descending to get latest first
            result = await mongodb_client.query_with_sort_and_pagination(
                self.db_name,
                self.collection_name,
                query,
                sort_field="created_at",
                sort_order=-1,
                limit=limit,
                skip=0
            )
            
            checkpoints = result.get("data", []) if isinstance(result, dict) else result
            log.info(f"Retrieved {len(checkpoints)} checkpoint records for thread_id: {thread_id}")
            return checkpoints
            
        except Exception as error:
            log.error(f"Error getting checkpoint records by thread: {error}")
            raise error
    
    async def get_by_id(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a single checkpoint by checkpoint ID."""
        try:
            query = {
                "checkpoint_id": checkpoint_id,
            }
            
            result = await mongodb_client.query_one(
                self.db_name,
                self.collection_name,
                query
            )
            
            if result:
                log.info(f"Found checkpoint record for checkpoint_id: {checkpoint_id}")
            else:
                log.info(f"No checkpoint record found for checkpoint_id: {checkpoint_id}")
            
            return result
            
        except Exception as error:
            log.error(f"Error getting checkpoint record by ID: {error}")
            raise error
    
    async def get_latest_by_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a specific thread."""
        try:
            query = {
                "thread_id": thread_id,
            }
            
            # Use aggregation pipeline to get the latest checkpoint
            pipeline = [
                {"$match": query},
                {"$sort": {"created_at": -1}},
                {"$limit": 1}
            ]
            
            result = await mongodb_client.query_one_by_pipeline(
                self.db_name,
                self.collection_name,
                pipeline
            )
            
            if result:
                log.info(f"Found latest checkpoint for thread_id: {thread_id}")
            else:
                log.info(f"No checkpoint found for thread_id: {thread_id}")
            
            return result
            
        except Exception as error:
            log.error(f"Error getting latest checkpoint by thread: {error}")
            raise error
    
    async def update(self, checkpoint_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a checkpoint record."""
        try:
            query = {
                "checkpoint_id": checkpoint_id,
            }
            
            # Add updated_at timestamp
            update_data["updated_at"] = datetime.utcnow().isoformat()
            
            result = await mongodb_client.update(
                self.db_name,
                self.collection_name,
                query,
                update_data
            )
            
            log.info(f"Updated checkpoint record for checkpoint_id: {checkpoint_id}")
            return result
            
        except Exception as error:
            log.error(f"Error updating checkpoint record: {error}")
            raise error
    
    async def update_metadata(self, checkpoint_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update checkpoint metadata."""
        try:
            update_data = {
                "metadata": metadata,
                "updated_at": datetime.utcnow().isoformat(),
            }
            
            return await self.update(checkpoint_id, update_data)
            
        except Exception as error:
            log.error(f"Error updating checkpoint metadata: {error}")
            raise error
    
    async def delete_by_id(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint record permanently."""
        try:
            query = {
                "checkpoint_id": checkpoint_id,
            }
            
            result = await mongodb_client.delete_one(
                self.db_name,
                self.collection_name,
                query
            )
            
            success = result.get("deleted_count", 0) > 0 if isinstance(result, dict) else False
            if success:
                log.info(f"Deleted checkpoint record for checkpoint_id: {checkpoint_id}")
            else:
                log.warning(f"Failed to delete checkpoint record for checkpoint_id: {checkpoint_id}")
            
            return success
            
        except Exception as error:
            log.error(f"Error deleting checkpoint record: {error}")
            raise error
    
    async def delete_by_thread(self, thread_id: str) -> int:
        """Delete all checkpoints for a specific thread permanently."""
        try:
            query = {
                "thread_id": thread_id,
            }
            
            result = await mongodb_client.delete_many(
                self.db_name,
                self.collection_name,
                query
            )

            result = await mongodb_client.delete_many(
                self.db_name,
                WRITE_COLLECTION_NAME,
                query
            )
            
            deleted_count = result.get("deleted_count", 0) if isinstance(result, dict) else 0
            log.info(f"Deleted {deleted_count} checkpoint records for thread_id: {thread_id}")
            return deleted_count
            
        except Exception as error:
            log.error(f"Error deleting checkpoint records by thread: {error}")
            raise error
    
    async def get_threads_with_checkpoints(self, limit: int = 100, skip: int = 0) -> List[str]:
        """Get a list of thread IDs that have checkpoints."""
        try:
            pipeline = [
                {"$group": {"_id": "$thread_id"}},
                {"$sort": {"_id": 1}},
                {"$skip": skip},
                {"$limit": limit},
                {"$project": {"thread_id": "$_id", "_id": 0}}
            ]
            
            result = await mongodb_client.query_by_pipeline(
                self.db_name,
                self.collection_name,
                pipeline
            )
            
            thread_ids = [doc.get("thread_id") for doc in result if doc.get("thread_id")]
            log.info(f"Retrieved {len(thread_ids)} unique thread IDs with checkpoints")
            return thread_ids
            
        except Exception as error:
            log.error(f"Error getting threads with checkpoints: {error}")
            raise error
    
    async def cleanup_old_checkpoints(self, days_old: int = 30, max_per_thread: int = 10) -> int:
        """Clean up old checkpoints, keeping only the most recent ones per thread."""
        try:
            cutoff_date = datetime.utcnow().timestamp() - (days_old * 24 * 60 * 60)
            cutoff_date_str = convert_timestamp_to_date(int(cutoff_date))
            
            # Find checkpoints to delete: either older than cutoff or exceeding max_per_thread
            pipeline = [
                {"$sort": {"thread_id": 1, "created_at": -1}},
                {"$group": {
                    "_id": "$thread_id",
                    "checkpoints": {"$push": "$$ROOT"}
                }},
                {"$project": {
                    "thread_id": "$_id",
                    "to_delete": {
                        "$filter": {
                            "input": "$checkpoints",
                            "cond": {
                                "$or": [
                                    {"$lt": ["$$this.created_at", cutoff_date_str]},
                                    {"$gte": [{"$indexOfArray": ["$checkpoints", "$$this"]}, max_per_thread]}
                                ]
                            }
                        }
                    }
                }},
                {"$unwind": "$to_delete"},
                {"$project": {"checkpoint_id": "$to_delete.checkpoint_id"}}
            ]
            
            checkpoints_to_delete = await mongodb_client.query_by_pipeline(
                self.db_name,
                self.collection_name,
                pipeline
            )
            
            deleted_count = 0
            for checkpoint in checkpoints_to_delete:
                checkpoint_id = checkpoint.get("checkpoint_id")
                if checkpoint_id:
                    success = await self.delete_by_id(checkpoint_id)
                    if success:
                        deleted_count += 1
            
            log.info(f"Cleaned up {deleted_count} old checkpoint records")
            return deleted_count
            
        except Exception as error:
            log.error(f"Error cleaning up old checkpoints: {error}")
            raise error


# Global instance
CheckpointWritesAio = CheckpointWritesAioModel()