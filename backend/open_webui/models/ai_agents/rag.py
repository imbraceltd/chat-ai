import logging
from typing import List, Dict, Any, Optional
from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import MONGODB_CONFIG
from open_webui.internal.mongo_db import mongodb_client

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAG", logging.INFO))

# Configuration from MongoDB config
DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")
COLLECTION_NAME = "rag"
NEW_COLLECTION_NAME = "new_rag"


class RAGModel:
    """Python equivalent of the JavaScript RAG model for MongoDB operations."""
    
    def __init__(self):
        self.db_name = DB_NAME
        self.collection_name = COLLECTION_NAME
        self.new_collection_name = NEW_COLLECTION_NAME
    
    async def remove(self, file_id: str = "", organization_id: str = "") -> Dict[str, Any]:
        """Remove RAG documents by file_id and organization_id."""
        try:
            query = {
                "file_id": file_id,
                "org_id": organization_id,
            }
            
            result = await mongodb_client.delete_many(
                self.db_name,
                self.collection_name,
                query
            )
            
            log.info(f"Removed {result.get('deleted_count', 0)} RAG documents for file_id: {file_id}")
            return result
            
        except Exception as error:
            log.error(f"Error removing RAG documents: {error}")
            raise error
    
    async def remove_new_rag(self, file_id: str = "", organization_id: str = "") -> Dict[str, Any]:
        """Remove new RAG documents by file_id and organization_id."""
        try:
            query = {
                "file_id": file_id,
                "org_id": organization_id,
            }
            
            result = await mongodb_client.delete_many(
                self.db_name,
                self.new_collection_name,
                query
            )
            
            log.info(f"Removed {result.get('deleted_count', 0)} new RAG documents for file_id: {file_id}")
            return result
            
        except Exception as error:
            log.error(f"Error removing new RAG documents: {error}")
            raise error
    
    async def remove_by_assistant_id(self, organization_id: str = "", assistant_id: str = "") -> Dict[str, Any]:
        """Remove RAG documents by organization_id and assistant_id."""
        try:
            query = {
                "org_id": organization_id,
                "assistant_id": assistant_id,
            }
            
            result = await mongodb_client.delete_many(
                self.db_name,
                self.collection_name,
                query
            )
            
            log.info(f"Removed {result.get('deleted_count', 0)} RAG documents for assistant_id: {assistant_id}")
            return result
            
        except Exception as error:
            log.error(f"Error removing RAG documents by assistant ID: {error}")
            raise error
    
    async def get_all(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all RAG documents with projection and deduplication."""
        try:
            projection = {
                "id": "$file_id",
                "file_id": 1,
                "org_id": 1,
                "assistant_id": 1,
                "bytes": 1,
                "created_at": 1,
                "filename": "$original_name",
                "blobType": 1,
                "source": 1,
                "url": 1,
            }
            
            data = await mongodb_client.query_with_projection(
                self.db_name,
                self.collection_name,
                query,
                projection
            )
            
            # Remove duplicates based on file_id (equivalent to JavaScript reduce logic)
            unique_data = []
            seen_file_ids = set()
            
            for item in data:
                file_id = item.get("file_id")
                if file_id not in seen_file_ids:
                    unique_data.append(item)
                    seen_file_ids.add(file_id)
            
            log.info(f"Retrieved {len(unique_data)} unique RAG documents")
            return unique_data
            
        except Exception as error:
            log.error(f"Error getting all RAG documents: {error}")
            raise error
    
    async def get_all_new_rag(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all new RAG documents with projection and deduplication."""
        try:
            projection = {
                "id": "$file_id",
                "file_id": 1,
                "org_id": 1,
                "assistant_id": 1,
                "bytes": 1,
                "created_at": 1,
                "filename": "$original_name",
                "blobType": 1,
                "source": 1,
                "url": 1,
            }
            
            data = await mongodb_client.query_with_projection(
                self.db_name,
                self.new_collection_name,
                query,
                projection
            )
            
            # Remove duplicates based on file_id
            unique_data = []
            seen_file_ids = set()
            
            for item in data:
                file_id = item.get("file_id")
                if file_id not in seen_file_ids:
                    unique_data.append(item)
                    seen_file_ids.add(file_id)
            
            log.info(f"Retrieved {len(unique_data)} unique new RAG documents")
            return unique_data
            
        except Exception as error:
            log.error(f"Error getting all new RAG documents: {error}")
            raise error
    
    async def get_by_org_id_and_file_id(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get RAG document by organization ID and file ID."""
        try:
            projection = {
                "id": "$file_id",
                "file_id": 1,
                "org_id": 1,
                "assistant_id": 1,
                "bytes": 1,
                "created_at": 1,
                "filename": "$original_name",
                "blobType": 1,
                "source": 1,
                "url": 1,
            }
            
            result = await mongodb_client.query_one_with_projection(
                self.db_name,
                self.collection_name,
                query,
                projection
            )
            
            if result:
                log.info(f"Found RAG document for query: {query}")
            else:
                log.info(f"No RAG document found for query: {query}")
            
            return result
            
        except Exception as error:
            log.error(f"Error getting RAG document by org_id and file_id: {error}")
            raise error
    
    async def get_by_org_id_and_file_id_for_new_rag(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get new RAG document by organization ID and file ID."""
        try:
            projection = {
                "id": "$file_id",
                "file_id": 1,
                "org_id": 1,
                "assistant_id": 1,
                "bytes": 1,
                "created_at": 1,
                "filename": "$original_name",
                "blobType": 1,
                "source": 1,
                "url": 1,
            }
            
            result = await mongodb_client.query_one_with_projection(
                self.db_name,
                self.new_collection_name,
                query,
                projection
            )
            
            if result:
                log.info(f"Found new RAG document for query: {query}")
            else:
                log.info(f"No new RAG document found for query: {query}")
            
            return result
            
        except Exception as error:
            log.error(f"Error getting new RAG document by org_id and file_id: {error}")
            raise error
    
    async def update_assistant_id(self, org_id: str = "", file_ids: List[str] = None, assistant_id: str = "") -> Dict[str, Any]:
        """Update assistant_id for multiple RAG documents."""
        try:
            if file_ids is None:
                file_ids = []
            
            query = {
                "org_id": org_id,
                "file_id": {"$in": file_ids},
            }
            
            update_data = {
                "assistant_id": assistant_id,
            }
            
            result = await mongodb_client.update_many(
                self.db_name,
                self.collection_name,
                query,
                update_data
            )
            
            log.info(f"Updated assistant_id for {result.get('modified_count', 0)} RAG documents")
            return result
            
        except Exception as error:
            log.error(f"Error updating assistant_id for RAG documents: {error}")
            raise error
    
    async def update_new_rag_assistant_id(self, org_id: str = "", file_ids: List[str] = None, assistant_id: str = "") -> Dict[str, Any]:
        """Update assistant_id for multiple new RAG documents."""
        try:
            if file_ids is None:
                file_ids = []
            
            query = {
                "org_id": org_id,
                "file_id": {"$in": file_ids},
            }
            
            update_data = {
                "assistant_id": assistant_id,
            }
            
            result = await mongodb_client.update_many(
                self.db_name,
                self.new_collection_name,
                query,
                update_data
            )
            
            log.info(f"Updated assistant_id for {result.get('modified_count', 0)} new RAG documents")
            return result
            
        except Exception as error:
            log.error(f"Error updating assistant_id for new RAG documents: {error}")
            raise error
    
    async def get_by_assistant_id(self, assistant_id: str = "") -> List[Dict[str, Any]]:
        """Get RAG documents by assistant ID."""
        try:
            query = {
                "assistant_id": assistant_id,
            }
            
            result = await mongodb_client.query(
                self.db_name,
                self.collection_name,
                query
            )
            
            log.info(f"Retrieved {len(result)} RAG documents for assistant_id: {assistant_id}")
            return result
            
        except Exception as error:
            log.error(f"Error getting RAG documents by assistant_id: {error}")
            raise error
    
    async def get_by_assistant_id_for_new_rag(self, assistant_id: str = "") -> List[Dict[str, Any]]:
        """Get new RAG documents by assistant ID."""
        try:
            query = {
                "assistant_id": assistant_id,
            }
            
            result = await mongodb_client.query(
                self.db_name,
                self.new_collection_name,
                query
            )
            
            log.info(f"Retrieved {len(result)} new RAG documents for assistant_id: {assistant_id}")
            return result
            
        except Exception as error:
            log.error(f"Error getting new RAG documents by assistant_id: {error}")
            raise error
    
    async def create(self, data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create new RAG documents."""
        try:
            if data is None:
                data = []
            
            if not data:
                raise ValueError("Data array cannot be empty")
            
            result = await mongodb_client.insert_many(
                self.db_name,
                self.collection_name,
                data
            )
            
            log.info(f"Created {result.get('inserted_count', 0)} RAG documents")
            return result
            
        except Exception as error:
            log.error(f"Error creating RAG documents: {error}")
            raise error
    
    async def create_for_new_rag(self, data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create new RAG documents in the new collection."""
        try:
            if data is None:
                data = []
            
            if not data:
                raise ValueError("Data array cannot be empty")
            
            result = await mongodb_client.insert_many(
                self.db_name,
                self.new_collection_name,
                data
            )
            
            log.info(f"Created {result.get('inserted_count', 0)} new RAG documents")
            return result
            
        except Exception as error:
            log.error(f"Error creating new RAG documents: {error}")
            raise error


# Create a global instance
rag_model = RAGModel()

# Export convenience functions that match the JavaScript interface
async def create(data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create new RAG documents."""
    return await rag_model.create(data)

async def remove(file_id: str = "", organization_id: str = "") -> Dict[str, Any]:
    """Remove RAG documents by file_id and organization_id."""
    return await rag_model.remove(file_id, organization_id)

async def remove_by_assistant_id(organization_id: str = "", assistant_id: str = "") -> Dict[str, Any]:
    """Remove RAG documents by organization_id and assistant_id."""
    return await rag_model.remove_by_assistant_id(organization_id, assistant_id)

async def get_all(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get all RAG documents with projection and deduplication."""
    return await rag_model.get_all(query)

async def get_all_new_rag(query: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get all new RAG documents with projection and deduplication."""
    return await rag_model.get_all_new_rag(query)

async def get_by_org_id_and_file_id(query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get RAG document by organization ID and file ID."""
    return await rag_model.get_by_org_id_and_file_id(query)

async def get_by_org_id_and_file_id_for_new_rag(query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get new RAG document by organization ID and file ID."""
    return await rag_model.get_by_org_id_and_file_id_for_new_rag(query)

async def update_assistant_id(org_id: str = "", file_ids: List[str] = None, assistant_id: str = "") -> Dict[str, Any]:
    """Update assistant_id for multiple RAG documents."""
    return await rag_model.update_assistant_id(org_id, file_ids, assistant_id)

async def update_new_rag_assistant_id(org_id: str = "", file_ids: List[str] = None, assistant_id: str = "") -> Dict[str, Any]:
    """Update assistant_id for multiple new RAG documents."""
    return await rag_model.update_new_rag_assistant_id(org_id, file_ids, assistant_id)

async def get_by_assistant_id(assistant_id: str = "") -> List[Dict[str, Any]]:
    """Get RAG documents by assistant ID."""
    return await rag_model.get_by_assistant_id(assistant_id)

async def get_by_assistant_id_for_new_rag(assistant_id: str = "") -> List[Dict[str, Any]]:
    """Get new RAG documents by assistant ID."""
    return await rag_model.get_by_assistant_id_for_new_rag(assistant_id)

async def remove_new_rag(file_id: str = "", organization_id: str = "") -> Dict[str, Any]:
    """Remove new RAG documents by file_id and organization_id."""
    return await rag_model.remove_new_rag(file_id, organization_id)

async def create_for_new_rag(data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create new RAG documents in the new collection."""
    return await rag_model.create_for_new_rag(data)

# Export all functions and the model class
__all__ = [
    "RAGModel",
    "rag_model",
    "create",
    "remove",
    "remove_by_assistant_id",
    "get_all",
    "get_all_new_rag",
    "get_by_org_id_and_file_id",
    "get_by_org_id_and_file_id_for_new_rag",
    "update_assistant_id",
    "update_new_rag_assistant_id",
    "get_by_assistant_id",
    "get_by_assistant_id_for_new_rag",
    "remove_new_rag",
    "create_for_new_rag",
]