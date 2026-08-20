import json
import logging
import os
import asyncio
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from bson import ObjectId

from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import MONGODB_CONFIG
from open_webui.internal.base_document_client import BaseDocumentClient

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MONGODB", logging.INFO))

ENABLE_INIT_VECTOR_INDEX = (
    os.getenv("ENABLE_INIT_VECTOR_INDEX", "true").lower() == "true"
)
VECTOR_DIMENSIONS = int(MONGODB_CONFIG.get("vector_dimensions", 1536))

OPENAI_MONGODB_HOST = MONGODB_CONFIG.get("openai_host", "")
OPENAI_DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")

VECTOR_CONFIG = {
    "collection_name": MONGODB_CONFIG.get("vector_collection_name", "rag"),
    "index_name": MONGODB_CONFIG.get("vector_index_name", "vector_index"),
    "full_text_search_index_name": MONGODB_CONFIG.get(
        "full_text_search_index_name", "fulltext_search"
    ),
    "dimensions": VECTOR_DIMENSIONS,
}


def serialize_objectid(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: serialize_objectid(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_objectid(item) for item in obj]
    return obj


def serialize_document(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if document is None:
        return None
    return serialize_objectid(document)


def serialize_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [serialize_objectid(doc) for doc in documents]


class MongoDocumentClient(BaseDocumentClient):
    """MongoDB implementation of BaseDocumentClient."""

    def __init__(self):
        self.openai_client: Optional[MongoClient] = None
        self.async_openai_client: Optional[AsyncIOMotorClient] = None
        self._connected = False

    async def connect(self):
        try:
            if OPENAI_MONGODB_HOST:
                log.info(f"Connecting to MongoDB at {OPENAI_MONGODB_HOST}")
                self.openai_client = MongoClient(OPENAI_MONGODB_HOST)
                self.async_openai_client = AsyncIOMotorClient(OPENAI_MONGODB_HOST)
                log.info("Connected to MongoDB")

            self._connected = True

            if ENABLE_INIT_VECTOR_INDEX:
                await self.init_vector_search()
                await self.init_full_text_search()

        except Exception as e:
            log.error(f"Error connecting to MongoDB: {e}")
            raise

    async def reconnect(self):
        try:
            if OPENAI_MONGODB_HOST:
                self.openai_client = MongoClient(OPENAI_MONGODB_HOST)
                self.async_openai_client = AsyncIOMotorClient(OPENAI_MONGODB_HOST)
            self._connected = True
        except Exception as e:
            log.error(f"Error reconnecting to MongoDB: {e}")
            raise

    async def get_client(self) -> MongoClient:
        if not self._connected:
            await self.reconnect()
        if not self.openai_client:
            raise ConnectionError("MongoDB client not initialized")
        return self.openai_client

    async def get_async_client(self) -> AsyncIOMotorClient:
        if not self._connected:
            await self.reconnect()
        if not self.async_openai_client:
            raise ConnectionError("MongoDB async client not initialized")
        return self.async_openai_client

    async def close(self):
        try:
            if self.openai_client:
                self.openai_client.close()
            if self.async_openai_client:
                self.async_openai_client.close()
            self._connected = False
        except Exception as e:
            log.error(f"Error closing MongoDB connections: {e}")

    async def init_full_text_search(self):
        try:
            client = await self.get_async_client()
            db = client[OPENAI_DB_NAME]

            collections = await db.list_collection_names()
            if VECTOR_CONFIG["collection_name"] not in collections:
                await db.create_collection(VECTOR_CONFIG["collection_name"])

            collection = db[VECTOR_CONFIG["collection_name"]]
            full_text_index_name = VECTOR_CONFIG["full_text_search_index_name"]
            desired_index_definition = {
                "mappings": {
                    "dynamic": False,
                    "fields": {
                        "assistant_id": {"type": "token"},
                        "board_id": {"type": "token"},
                        "file_id": {"type": "token"},
                        "folder_id": {"type": "token"},
                        "org_id": {"type": "token"},
                        "text": {"type": "string"},
                        "board_ids": {"type": "token"},
                    },
                },
            }

            indexes = await collection.list_search_indexes().to_list(None)
            existing_index = next(
                (idx for idx in indexes if idx.get("name") == full_text_index_name),
                None,
            )

            if existing_index:
                existing_definition = json.dumps(
                    existing_index.get("latestDefinition", existing_index.get("definition", {})),
                    sort_keys=True,
                )
                desired_definition = json.dumps(desired_index_definition, sort_keys=True)
                if existing_definition == desired_definition:
                    return
                await collection.drop_search_index(full_text_index_name)

            await collection.create_search_index(
                {"name": full_text_index_name, "definition": desired_index_definition}
            )
            await self._wait_for_index_ready(collection, full_text_index_name)

        except Exception as e:
            log.error(f"Error managing full-text search index: {e}")

    async def init_vector_search(self):
        try:
            client = await self.get_async_client()
            db = client[OPENAI_DB_NAME]

            collections = await db.list_collection_names()
            if VECTOR_CONFIG["collection_name"] not in collections:
                await db.create_collection(VECTOR_CONFIG["collection_name"])

            collection = db[VECTOR_CONFIG["collection_name"]]
            indexes = await collection.list_search_indexes().to_list(None)

            existing_index = next(
                (idx for idx in indexes if idx.get("name") == VECTOR_CONFIG["index_name"]),
                None,
            )

            if existing_index:
                latest_def = existing_index.get("latestDefinition", {})
                fields = latest_def.get("fields", [])
                vector_field = next(
                    (f for f in fields if f.get("type") == "vector" and f.get("path") == "embedding"),
                    None,
                )
                if vector_field and vector_field.get("numDimensions") != VECTOR_DIMENSIONS:
                    await collection.drop_search_index(VECTOR_CONFIG["index_name"])
                else:
                    return

            vector_index_definition = SearchIndexModel(
                name=VECTOR_CONFIG["index_name"],
                definition={
                    "fields": [
                        {"type": "vector", "path": "embedding", "numDimensions": VECTOR_DIMENSIONS, "similarity": "dotProduct"},
                        {"type": "filter", "path": "org_id"},
                        {"type": "filter", "path": "assistant_id"},
                        {"type": "filter", "path": "file_id"},
                        {"type": "filter", "path": "folder_id"},
                        {"type": "filter", "path": "board_id"},
                        {"type": "filter", "path": "board_ids"},
                    ],
                },
                type="vectorSearch",
            )

            await collection.create_search_index(model=vector_index_definition)
            await self._wait_for_index_ready(collection, VECTOR_CONFIG["index_name"])

        except Exception as e:
            log.error(f"Error creating vector search index: {e}")

    async def _wait_for_index_ready(self, collection, index_name: str):
        while True:
            try:
                indexes = await collection.list_search_indexes().to_list(None)
                target = next((idx for idx in indexes if idx.get("name") == index_name), None)
                if target and target.get("queryable"):
                    break
                await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Error checking index status: {e}")
                await asyncio.sleep(5)

    # ── Document operations ──────────────────────────────────────────────────

    async def query(self, database_name="", collection_name="", query=None) -> List[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            cursor = client[database_name][collection_name].find(query or {}).sort("updated_at", -1)
            return serialize_documents(await cursor.to_list(None))
        except Exception as e:
            log.error(f"Error querying MongoDB: {e}")
            return []

    async def query_one(self, database_name="", collection_name="", query=None) -> Optional[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            doc = await client[database_name][collection_name].find_one(query or {}, sort=[("updated_at", -1)])
            return serialize_document(doc)
        except Exception as e:
            log.error(f"Error querying MongoDB: {e}")
            return None

    async def query_one_with_projection(self, database_name="", collection_name="", query=None, projection=None) -> Optional[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            doc = await client[database_name][collection_name].find_one(
                query or {}, projection=projection, sort=[("updated_at", -1)]
            )
            return serialize_document(doc)
        except Exception as e:
            log.error(f"Error querying MongoDB: {e}")
            return None

    async def query_with_projection(self, database_name="", collection_name="", query=None, projection=None) -> List[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            cursor = client[database_name][collection_name].find(query or {}, projection=projection).sort("updated_at", -1)
            return serialize_documents(await cursor.to_list(None))
        except Exception as e:
            log.error(f"Error querying MongoDB: {e}")
            return []

    async def query_with_sort_and_pagination(self, database_name="", collection_name="", query=None, sort_field="updated_at", sort_order=-1, limit=10, skip=0) -> Dict[str, Any]:
        try:
            client = await self.get_async_client()
            collection = client[database_name][collection_name]
            total = await collection.count_documents(query or {})
            cursor = collection.find(query or {}).sort(sort_field, sort_order)
            documents = serialize_documents(await cursor.to_list(None))
            count = len(documents)
            return {
                "object": "list",
                "count": count,
                "total": total,
                "has_more": skip + count < total,
                "data": documents,
                "documents": documents,
            }
        except Exception as e:
            log.error(f"Error querying MongoDB: {e}")
            return {"count": 0, "total": 0, "has_more": False, "data": [], "documents": []}

    async def query_one_by_pipeline(self, database_name="", collection_name="", pipeline=None) -> Optional[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            cursor = client[database_name][collection_name].aggregate(pipeline or [])
            results = await cursor.to_list(None)
            return serialize_document(results[0]) if results else None
        except Exception as e:
            log.error(f"Error in query_one_by_pipeline: {e}")
            raise Exception(f"Error query_one_by_pipeline: {e}")

    async def query_by_pipeline(self, database_name="", collection_name="", pipeline=None) -> List[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            cursor = client[database_name][collection_name].aggregate(pipeline or [])
            return serialize_documents(await cursor.to_list(None))
        except Exception as e:
            log.error(f"Error in query_by_pipeline: {e}")
            return []

    async def insert(self, database_name="", collection_name="", data=None) -> Optional[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            collection = client[database_name][collection_name]
            result = await collection.insert_one(data or {})
            doc = await collection.find_one({"_id": result.inserted_id})
            return serialize_document(doc)
        except Exception as e:
            log.error(f"Error inserting to MongoDB: {e}")
            return None

    async def insert_many(self, database_name="", collection_name="", data_array=None) -> Dict[str, Any]:
        try:
            if not data_array or not isinstance(data_array, list):
                raise ValueError("Input must be a non-empty list of objects.")
            client = await self.get_async_client()
            result = await client[database_name][collection_name].insert_many(data_array)
            return {
                "success": True,
                "inserted_count": len(result.inserted_ids),
                "inserted_ids": [str(i) for i in result.inserted_ids],
            }
        except Exception as e:
            log.error(f"Error inserting many to MongoDB: {e}")
            raise

    async def update(self, database_name="", collection_name="", query=None, data=None) -> Optional[Dict[str, Any]]:
        try:
            client = await self.get_async_client()
            result = await client[database_name][collection_name].find_one_and_update(
                query or {}, {"$set": data or {}}, return_document=True
            )
            return serialize_document(result)
        except Exception as e:
            log.error(f"Error updating MongoDB: {e}")
            return None

    async def update_many(self, database_name="", collection_name="", query=None, data=None) -> Dict[str, Any]:
        try:
            client = await self.get_async_client()
            result = await client[database_name][collection_name].update_many(query or {}, {"$set": data or {}})
            return {"matched_count": result.matched_count, "modified_count": result.modified_count}
        except Exception as e:
            log.error(f"Error updating many in MongoDB: {e}")
            return {"matched_count": 0, "modified_count": 0}

    async def delete_one(self, database_name="", collection_name="", query=None) -> Dict[str, Any]:
        try:
            client = await self.get_async_client()
            result = await client[database_name][collection_name].delete_one(query or {})
            return {"deleted_count": result.deleted_count}
        except Exception as e:
            log.error(f"Error deleting from MongoDB: {e}")
            return {"deleted_count": 0}

    async def delete_many(self, database_name="", collection_name="", query=None) -> Dict[str, Any]:
        try:
            client = await self.get_async_client()
            result = await client[database_name][collection_name].delete_many(query or {})
            return {"deleted_count": result.deleted_count}
        except Exception as e:
            log.error(f"Error deleting many from MongoDB: {e}")
            return {"deleted_count": 0}
