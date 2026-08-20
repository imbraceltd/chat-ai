"""Vector store factory.

Returns the appropriate LangChain vector store based on DB_TYPE:

    DB_TYPE=mongodb      → MongoDBAtlasVectorSearch
    DB_TYPE=postgresql   → PGVector (langchain-postgres)

Usage in callers:
    from open_webui.internal.vector_store import get_vector_store, get_rag_collection

    # Get a configured vector store instance
    vector_store = get_vector_store(embeddings, filter={"org_id": org_id})

    # For raw MongoDB collection access (MongoDB mode only)
    rag_collection = get_rag_collection()
"""

import logging
import os
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

DB_TYPE = os.getenv("DB_TYPE", "mongodb").lower()

# ── Shared config ────────────────────────────────────────────────────────────
VECTOR_COLLECTION = os.getenv("MONGODB_VECTOR_COLLECTION", "rag")
VECTOR_INDEX_NAME = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")
VECTOR_DIMENSIONS = int(os.getenv("MONGODB_VECTOR_DIMENSIONS", "1536"))
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Blue-green embedding migration: which rag column ingest writes into.
# Whitelisted because the value is interpolated directly into SQL (column
# names can't be bound as parameters).
_ALLOWED_EMBEDDING_COLUMNS = {"embedding", "embedding_v2"}
EMBEDDING_WRITE_COLUMN = os.getenv("RAG_EMBEDDING_WRITE_COLUMN", "embedding")
if EMBEDDING_WRITE_COLUMN not in _ALLOWED_EMBEDDING_COLUMNS:
    log.warning(
        "Invalid RAG_EMBEDDING_WRITE_COLUMN=%r; falling back to 'embedding'",
        EMBEDDING_WRITE_COLUMN,
    )
    EMBEDDING_WRITE_COLUMN = "embedding"


def get_rag_collection():
    """Return the raw MongoDB collection for the rag store.

    Only valid when DB_TYPE=mongodb.  Callers that need to pass a raw
    collection to MongoDBAtlasVectorSearch should use this.
    Raises RuntimeError when running in PostgreSQL mode.
    """
    if DB_TYPE != "mongodb":
        raise RuntimeError(
            "get_rag_collection() is only available in mongodb mode. "
            "Use get_vector_store() instead."
        )
    from open_webui.internal.mongo_document_client import MongoDocumentClient
    from open_webui.internal.document_store import document_client

    if not isinstance(document_client, MongoDocumentClient):
        raise RuntimeError("document_client is not a MongoDocumentClient")

    import asyncio

    async def _get():
        client = await document_client.get_async_client()
        from open_webui.config import MONGODB_CONFIG
        db_name = MONGODB_CONFIG.get("openai_db_name", "openai_db")
        return client[db_name][VECTOR_COLLECTION]

    # Return sync client for callers that use it synchronously
    client = document_client.async_openai_client
    if client is None:
        raise RuntimeError("MongoDB async client not connected")
    from open_webui.config import MONGODB_CONFIG
    db_name = MONGODB_CONFIG.get("openai_db_name", "openai_db")
    return client[db_name][VECTOR_COLLECTION]


def get_vector_store(embeddings, pre_filter: Optional[Dict[str, Any]] = None):
    """Return a configured LangChain vector store.

    Args:
        embeddings: A LangChain embeddings instance.
        pre_filter: Optional dict of equality filters (e.g. {"org_id": "x"}).
                    Applied as pre-filter in MongoDB or as WHERE in PGVector.
    """
    if DB_TYPE == "postgresql":
        return _get_pgvector_store(embeddings, pre_filter)
    else:
        return _get_mongodb_vector_store(embeddings, pre_filter)


def _get_mongodb_vector_store(embeddings, pre_filter):
    from langchain_mongodb import MongoDBAtlasVectorSearch

    collection = get_rag_collection()
    store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=VECTOR_INDEX_NAME,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="dotProduct",
    )
    return store


def _get_pgvector_store(embeddings, pre_filter):
    from open_webui.llm.utils.pg_adapters import PgVectorStore

    return PgVectorStore(embeddings=embeddings, database_url=DATABASE_URL)


async def afrom_documents(documents, embeddings, **kwargs):
    """Async helper to ingest documents into the vector store.

    For PostgreSQL: inserts directly into the rag table via asyncpg.
    For MongoDB: delegates to MongoDBAtlasVectorSearch.afrom_documents.
    Extra kwargs (collection, index_name, etc.) are used in MongoDB mode only.
    """
    if DB_TYPE == "postgresql":
        return await _pg_ingest_documents(documents, embeddings)
    else:
        from langchain_mongodb import MongoDBAtlasVectorSearch

        return await MongoDBAtlasVectorSearch.afrom_documents(
            documents=documents,
            embedding=embeddings,
            **kwargs,
        )


async def _pg_ingest_documents(documents, embeddings):
    """Insert LangChain Document objects into the rag table."""
    import asyncpg
    import json
    import uuid

    texts = [doc.page_content for doc in documents]
    vectors = await embeddings.aembed_documents(texts)

    # Blue-green migration: the embedding column to write into is configurable
    # (default `embedding`). Validated against a whitelist because column names
    # cannot be bound as query parameters.
    upsert_sql = f"""
        INSERT INTO rag (
            id, text, {EMBEDDING_WRITE_COLUMN}, org_id, assistant_id, file_id,
            board_id, boarditem_id, folder_id, source, blob_type,
            original_name, bytes, board_ids, data
        ) VALUES (
            $1, $2, $3::vector, $4, $5, $6,
            $7, $8, $9, $10, $11,
            $12, $13, $14, $15::jsonb
        )
        ON CONFLICT (id) DO UPDATE
            SET text                     = EXCLUDED.text,
                {EMBEDDING_WRITE_COLUMN} = EXCLUDED.{EMBEDDING_WRITE_COLUMN},
                data                     = EXCLUDED.data,
                updated_at               = NOW()
    """

    rows = []
    for doc, vec in zip(documents, vectors):
        meta = doc.metadata or {}
        doc_id = meta.get("id") or str(uuid.uuid4())
        embedding_str = "[" + ",".join(str(float(v)) for v in vec) + "]"
        rows.append((
            doc_id,
            doc.page_content,
            embedding_str,
            meta.get("org_id", ""),
            meta.get("assistant_id", ""),
            meta.get("file_id", ""),
            meta.get("board_id", ""),
            meta.get("boarditem_id", ""),
            meta.get("folder_id", ""),
            meta.get("source", ""),
            meta.get("blobType", meta.get("blob_type", "")),
            meta.get("original_name", ""),
            meta.get("bytes"),
            meta.get("board_ids") or [],
            json.dumps(meta),
        ))

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        async with pool.acquire() as conn:
            await conn.executemany(upsert_sql, rows)
    finally:
        await pool.close()

    log.info(f"[pg_ingest] Inserted {len(rows)} documents into rag table")
    return rows
