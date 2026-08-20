"""Backward-compatibility shim.

All existing imports of `mongodb_client` and the module-level helper
functions (query, insert, update, …) continue to work unchanged.

The actual implementation is selected by DB_TYPE in document_store.py:
    DB_TYPE=mongodb      → MongoDocumentClient
    DB_TYPE=postgresql   → PgDocumentClient
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from bson import ObjectId

from open_webui.env import SRC_LOG_LEVELS
from open_webui.config import MONGODB_CONFIG
from open_webui.internal.document_store import document_client

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MONGODB", logging.INFO))

# ── Backward-compat constants ────────────────────────────────────────────────
# OPENAI_DB_NAME is used by several repository/util files that pass it as
# `database_name` to the client methods.  The PgDocumentClient ignores
# database_name (it uses a single PostgreSQL database), so exporting it
# here keeps existing imports working without any caller changes.
OPENAI_DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")

# ── Backward-compat alias ───────────────────────────────────────────────────
mongodb_client = document_client


# ── Serialization helpers (kept here; used by several callers) ───────────────
def serialize_objectid(obj):
    """Convert ObjectId and other BSON types to JSON-serializable formats."""
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


# ── Context manager ──────────────────────────────────────────────────────────
@asynccontextmanager
async def get_mongodb_session():
    if not document_client._connected:
        await document_client.connect()
    try:
        yield document_client
    except Exception as e:
        log.error(f"Error in document client session: {e}")
        raise


# ── Module-level convenience functions (unchanged API) ───────────────────────
async def connect():
    return await document_client.connect()


async def query(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    return await document_client.query(database_name, collection_name, query_dict)


async def query_one(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
) -> Optional[Dict[str, Any]]:
    return await document_client.query_one(database_name, collection_name, query_dict)


async def query_one_with_projection(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
    projection: Dict[str, Any] = None,
) -> Optional[Dict[str, Any]]:
    return await document_client.query_one_with_projection(
        database_name, collection_name, query_dict, projection
    )


async def query_with_projection(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
    projection: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    return await document_client.query_with_projection(
        database_name, collection_name, query_dict, projection
    )


async def query_with_sort_and_pagination(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
    sort_field: str = "updated_at",
    sort_order: int = -1,
    limit: int = 10,
    skip: int = 0,
) -> Dict[str, Any]:
    return await document_client.query_with_sort_and_pagination(
        database_name, collection_name, query_dict, sort_field, sort_order, limit, skip
    )


async def query_one_by_pipeline(
    database_name: str = "",
    collection_name: str = "",
    pipeline: List[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    return await document_client.query_one_by_pipeline(
        database_name, collection_name, pipeline
    )


async def query_by_pipeline(
    database_name: str = "",
    collection_name: str = "",
    pipeline: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return await document_client.query_by_pipeline(
        database_name, collection_name, pipeline
    )


async def insert(
    database_name: str = "",
    collection_name: str = "",
    data: Dict[str, Any] = None,
) -> Optional[Dict[str, Any]]:
    return await document_client.insert(database_name, collection_name, data)


async def insert_many(
    database_name: str = "",
    collection_name: str = "",
    data_array: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await document_client.insert_many(database_name, collection_name, data_array)


async def update(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
    data: Dict[str, Any] = None,
) -> Optional[Dict[str, Any]]:
    return await document_client.update(database_name, collection_name, query_dict, data)


async def update_many(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
    data: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return await document_client.update_many(
        database_name, collection_name, query_dict, data
    )


async def delete_one(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return await document_client.delete_one(database_name, collection_name, query_dict)


async def delete_many(
    database_name: str = "",
    collection_name: str = "",
    query_dict: Dict[str, Any] = None,
) -> Dict[str, Any]:
    return await document_client.delete_many(database_name, collection_name, query_dict)


__all__ = [
    "mongodb_client",
    "document_client",
    "OPENAI_DB_NAME",
    "get_mongodb_session",
    "serialize_objectid",
    "serialize_document",
    "serialize_documents",
    "connect",
    "query",
    "query_one",
    "query_one_with_projection",
    "query_with_projection",
    "query_with_sort_and_pagination",
    "query_one_by_pipeline",
    "query_by_pipeline",
    "insert",
    "insert_many",
    "update",
    "update_many",
    "delete_one",
    "delete_many",
]
