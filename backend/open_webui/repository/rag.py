from typing import List, Dict, Any
import logging
from open_webui.internal.mongo_db import mongodb_client, get_mongodb_session, OPENAI_DB_NAME
from datetime import datetime
from open_webui.env import SRC_LOG_LEVELS


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])


RAG_COLLECTION = "rag"
NEW_RAG_COLLECTION = "new_rag"


class RAGRepository:
    def __init__(self):
        self.db_client = mongodb_client
        self.db_name = OPENAI_DB_NAME
        self.collection_name = RAG_COLLECTION
        self.new_rag_collection_name = NEW_RAG_COLLECTION

    async def updateAssistantId(self, organization_id: str, file_ids: List[str], assistant_id: str):
        try:
            query = {
                "organization_id": organization_id,
                "file_id": {"$in": file_ids}
            }
            update_data = {
                "assistant_id": assistant_id,
                "updated_at": datetime.utcnow().isoformat(),
            }

            result = await self.db_client.update_many(
                database_name=self.db_name,
                collection_name=self.new_rag_collection_name,
                query=query,
                data=update_data
            )
            return result
        except Exception as e:
            log.error(f"Error updating assistant id: {e}")
            raise

    async def remove_assistant_id(self, organization_id: str, assistant_id: str):
        try:
            query = {
                "organization_id": organization_id,
                "assistant_id": assistant_id
            }
            result = await self.db_client.delete_many(
                database_name=self.db_name,
                collection_name=self.collection_name,
                query=query
            )
            return result
        except Exception as e:
            log.error(f"Error removing assistant id: {e}")
            raise

