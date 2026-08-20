"""PostgreSQL adapters for the agent's vector search.

PgVectorStore is a drop-in replacement for MongoDBAtlasVectorSearch that
queries the `rag` table directly with pgvector cosine-similarity search.

The filter format accepted by asimilarity_search_with_score is the same
MongoDB-style dict produced by build_filters():
    {
        "org_id": {"$eq": "org123"},
        "$or": [
            {"assistant_id": {"$eq": "asst_id"}},
            {"board_id": {"$in": ["b1", "b2"]}},
            {"file_id": {"$in": ["f1", "f2"]}},
            {"folder_id": {"$in": ["fold1"]}},
        ],
    }
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Blue-green embedding migration: which rag column vector search reads from.
# Whitelisted because the value is interpolated directly into SQL (column
# names can't be bound as parameters). During backfill keep this on the old
# `embedding` (OpenAI) column; flip to `embedding_v2` to cut over to Bedrock.
_ALLOWED_EMBEDDING_COLUMNS = {"embedding", "embedding_v2"}
EMBEDDING_READ_COLUMN = os.getenv("RAG_EMBEDDING_READ_COLUMN", "embedding")
if EMBEDDING_READ_COLUMN not in _ALLOWED_EMBEDDING_COLUMNS:
    log.warning(
        "Invalid RAG_EMBEDDING_READ_COLUMN=%r; falling back to 'embedding'",
        EMBEDDING_READ_COLUMN,
    )
    EMBEDDING_READ_COLUMN = "embedding"

# Columns that exist as plain TEXT columns in the rag table (not inside JSONB data)
_RAG_DIRECT_COLUMNS = {
    "id", "text", "org_id", "assistant_id", "file_id",
    "board_id", "boarditem_id", "folder_id", "source",
    "blob_type", "original_name",
}


def _rag_filter_to_where(filter_dict: Dict[str, Any], params: List) -> str:
    """Translate a MongoDB-style filter into a SQL WHERE fragment for the rag table.

    The rag table uses direct columns (org_id, board_id, …) rather than JSONB,
    so we map each field to the appropriate SQL expression.
    """
    if not filter_dict:
        return "TRUE"

    conditions = []

    for key, value in filter_dict.items():
        if key == "$or":
            sub = [_rag_filter_to_where(f, params) for f in value]
            conditions.append(f"({' OR '.join(sub)})")
            continue
        if key == "$and":
            sub = [_rag_filter_to_where(f, params) for f in value]
            conditions.append(f"({' AND '.join(sub)})")
            continue

        # Pick the SQL column reference
        if key in _RAG_DIRECT_COLUMNS:
            col = key
        elif key == "_id":
            col = "id"
        else:
            col = f"data->>{key!r}"

        if value is None:
            conditions.append(f"{col} IS NULL")
        elif isinstance(value, dict):
            for op, op_val in value.items():
                if op == "$eq":
                    params.append(str(op_val) if op_val is not None else op_val)
                    conditions.append(f"{col} = ${len(params)}")
                elif op == "$ne":
                    params.append(str(op_val) if op_val is not None else op_val)
                    conditions.append(f"{col} != ${len(params)}")
                elif op == "$in":
                    params.append([str(v) for v in op_val])
                    conditions.append(f"{col} = ANY(${len(params)}::text[])")
                elif op == "$nin":
                    params.append([str(v) for v in op_val])
                    conditions.append(f"NOT ({col} = ANY(${len(params)}::text[]))")
                elif op == "$exists":
                    if op_val:
                        conditions.append(f"{col} IS NOT NULL")
                    else:
                        conditions.append(f"{col} IS NULL")
                elif op in ("$gt", "$gte", "$lt", "$lte"):
                    op_map = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}
                    params.append(op_val)
                    conditions.append(f"{col} {op_map[op]} ${len(params)}")
                else:
                    log.warning(f"PgVectorStore: unsupported filter op {op} on {key}")
        else:
            params.append(str(value))
            conditions.append(f"{col} = ${len(params)}")

    return " AND ".join(conditions) if conditions else "TRUE"


class PgVectorStore:
    """Async vector store backed by the `rag` PostgreSQL table.

    Implements the subset of the LangChain VectorStore interface used by agent.py:
        - asimilarity_search_with_score(query, k, pre_filter)
    """

    def __init__(self, embeddings, database_url: str = DATABASE_URL):
        self.embeddings = embeddings
        self.database_url = database_url
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.database_url, min_size=1, max_size=4
            )
        return self._pool

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        pre_filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Return (Document, score) pairs ordered by cosine similarity (descending)."""
        from langchain_core.documents import Document

        # 1. Embed the query string
        embedding = await self.embeddings.aembed_query(query)
        embedding_str = "[" + ",".join(str(float(v)) for v in embedding) + "]"

        # 2. Translate the MongoDB-style filter to SQL
        params: List[Any] = [embedding_str]  # $1 = query embedding
        where = _rag_filter_to_where(pre_filter or {}, params)
        # params are now [$1=embedding, $2...$N=filter values]
        # k needs a separate param slot
        params.append(k)
        k_param = f"${len(params)}"

        sql = f"""
            SELECT
                id, text, org_id, assistant_id, file_id, board_id,
                boarditem_id, folder_id, source, original_name,
                bytes, board_ids,
                1 - ({EMBEDDING_READ_COLUMN} <=> $1::vector) AS score
            FROM rag
            WHERE {EMBEDDING_READ_COLUMN} IS NOT NULL
              AND ({where})
            ORDER BY {EMBEDDING_READ_COLUMN} <=> $1::vector
            LIMIT {k_param}
        """

        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except Exception as e:
            log.error(f"PgVectorStore similarity search failed: {e}")
            return []

        results = []
        for row in rows:
            metadata = {
                "org_id": row["org_id"],
                "assistant_id": row["assistant_id"],
                "file_id": row["file_id"],
                "board_id": row["board_id"],
                "boarditem_id": row["boarditem_id"],
                "folder_id": row["folder_id"],
                "source": row["source"],
                "original_name": row["original_name"],
                "bytes": row["bytes"],
                "board_ids": list(row["board_ids"]) if row["board_ids"] else [],
            }
            doc = Document(page_content=row["text"] or "", metadata=metadata)
            results.append((doc, float(row["score"])))

        return results

    async def full_text_search(
        self,
        query: str,
        k: int = 10,
        pre_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """PostgreSQL full-text search against rag.text using the GIN tsvector index.

        Returns dicts with keys: pageContent, metadata, fullTextScore.
        fullTextScore is ts_rank normalised to [0, 1] by capping at 1.
        """
        # $1 = query text (plainto_tsquery); filter values start at $2 automatically
        # because we pre-seed the params list with the query value.
        params: List[Any] = [query]
        where = _rag_filter_to_where(pre_filter or {}, params)
        params.append(k)
        k_param_idx = len(params)

        # ts_rank_cd (cover density) ranks better than ts_rank for proximity, and
        # websearch_to_tsquery understands phrases/operators ("a b", -c, or) and
        # never errors on free-text input (unlike to_tsquery). The tsvector side is
        # unchanged so the GIN index rag_text_fts_idx is still used.
        sql = f"""
            SELECT
                id, text, original_name,
                ts_rank_cd(
                    to_tsvector('english', coalesce(text, '')),
                    websearch_to_tsquery('english', $1)
                ) AS score
            FROM rag
            WHERE to_tsvector('english', coalesce(text, ''))
                  @@ websearch_to_tsquery('english', $1)
              AND ({where})
            ORDER BY score DESC
            LIMIT ${k_param_idx}
        """

        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
        except Exception as e:
            log.error(f"PgVectorStore full_text_search failed: {e}")
            return []

        results = []
        for row in rows:
            raw_score = float(row["score"])
            results.append({
                "pageContent": row["text"] or "",
                "metadata": {
                    "_id": row["id"],
                    "original_name": row["original_name"],
                },
                "fullTextScore": min(raw_score, 1.0),
            })
        return results

    async def hybrid_search(
        self,
        query: str,
        k: int = 10,
        pre_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run vector search and full-text search concurrently, merge via RRF.

        Returns dicts with keys: pageContent, metadata, vectorScore,
        fullTextScore, combined_score (RRF).
        """
        import asyncio

        vector_task = asyncio.create_task(
            self.asimilarity_search_with_score(query, k=k, pre_filter=pre_filter)
        )
        fts_task = asyncio.create_task(
            self.full_text_search(query, k=k, pre_filter=pre_filter)
        )
        vector_results, fts_results = await asyncio.gather(
            vector_task, fts_task, return_exceptions=True
        )
        if isinstance(vector_results, Exception):
            log.error(f"hybrid_search vector leg failed: {vector_results}")
            vector_results = []
        if isinstance(fts_results, Exception):
            log.error(f"hybrid_search FTS leg failed: {fts_results}")
            fts_results = []

        # Normalise into a common shape before RRF
        vec_docs = [
            {
                "pageContent": doc.page_content,
                "metadata": doc.metadata,
                "vectorScore": score,
            }
            for doc, score in vector_results
        ]
        return _rrf_merge(vec_docs, fts_results, k)


def _rrf_merge(
    vector_docs: List[Dict[str, Any]],
    fts_docs: List[Dict[str, Any]],
    k: int,
    rank_constant: int = 60,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion of two ranked lists.

    Each doc is identified by its pageContent (same chunk text = same doc).
    Final combined_score = 1/(rank_constant + rank_v) + 1/(rank_constant + rank_f).
    """
    scores: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(vector_docs):
        key = doc["pageContent"]
        entry = scores.setdefault(key, {**doc, "fullTextScore": 0.0, "combined_score": 0.0})
        entry["combined_score"] += 1.0 / (rank_constant + rank + 1)
        entry["vectorScore"] = doc.get("vectorScore", 0.0)

    for rank, doc in enumerate(fts_docs):
        key = doc["pageContent"]
        entry = scores.setdefault(key, {
            "pageContent": doc["pageContent"],
            "metadata": doc.get("metadata", {}),
            "vectorScore": 0.0,
            "fullTextScore": 0.0,
            "combined_score": 0.0,
        })
        entry["combined_score"] += 1.0 / (rank_constant + rank + 1)
        entry["fullTextScore"] = doc.get("fullTextScore", 0.0)

    merged = sorted(scores.values(), key=lambda d: d["combined_score"], reverse=True)
    return merged[:k]


# ── PgChatMessageHistory ──────────────────────────────────────────────────────

class PgChatMessageHistory:
    """Drop-in replacement for MongoDBChatMessageHistory backed by the `history` JSONB table.

    Stores messages in the same format as MongoDBChatMessageHistory so that
    migrated data is readable:
        data = {"SessionId": thread_id, "History": [{"type": ..., "data": {...}}, ...]}
    """

    def __init__(self, session_id: str, database_url: str = DATABASE_URL):
        self.session_id = session_id
        self.database_url = database_url
        self._pool = None
        self._lock = None

    async def _get_pool(self):
        if self._pool is None:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.database_url, min_size=1, max_size=4
            )
        return self._pool

    async def _load_doc(self):
        """Return the JSONB data dict for this session, or None."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data FROM history WHERE data->>'sessionId' = $1 LIMIT 1",
                self.session_id,
            )
        if row:
            import json
            data = row["data"]
            if isinstance(data, str):
                data = json.loads(data)
            return data
        return None

    async def aget_messages(self):
        """Return list of LangChain message objects for this session."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        doc = await self._load_doc()
        if not doc:
            return []

        messages = []
        # Support both "messages" (app format) and "History" (LangChain default format)
        for item in doc.get("messages", doc.get("History", [])):
            msg_type = item.get("type", "")
            content = item.get("data", {}).get("content", "")
            if msg_type == "human":
                messages.append(HumanMessage(content=content))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content))
        return messages

    def get_messages(self):
        """Synchronous wrapper — runs aget_messages in a new event loop if needed."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In async context caller should use aget_messages directly
                return []
            return loop.run_until_complete(self.aget_messages())
        except Exception:
            return []

    def _get_write_lock(self):
        """Lazily create a per-instance asyncio.Lock (must be created inside event loop)."""
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
        return self._lock

    async def _append_message(self, msg_type: str, content: str):
        """Append one message, serialised per-instance with a Lock to prevent races."""
        import json, uuid

        async with self._get_write_lock():
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, data FROM history WHERE data->>'sessionId' = $1 "
                    "ORDER BY updated_at DESC LIMIT 1",
                    self.session_id,
                )
                new_msg = {"type": msg_type, "data": {"content": content}}
                if row:
                    existing = row["data"] if isinstance(row["data"], dict) else json.loads(row["data"])
                    msgs = existing.get("messages", existing.get("History", []))
                    msgs.append(new_msg)
                    existing["messages"] = msgs
                    await conn.execute(
                        "UPDATE history SET data = $1::jsonb, updated_at = NOW() WHERE id = $2",
                        json.dumps(existing), row["id"],
                    )
                else:
                    now = __import__("datetime").datetime.utcnow().isoformat()
                    doc = {"sessionId": self.session_id, "messages": [new_msg],
                           "created_at": now, "updated_at": now}
                    await conn.execute(
                        "INSERT INTO history (id, data) VALUES ($1, $2::jsonb)",
                        str(uuid.uuid4()), json.dumps(doc),
                    )

    def add_user_message(self, content: str):
        import asyncio
        asyncio.ensure_future(self._append_message("human", content))

    def add_ai_message(self, content: str):
        import asyncio
        asyncio.ensure_future(self._append_message("ai", content))

    def clear(self):
        """Sync wrapper for aclear() — schedules the coroutine as a task."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.aclear())
            else:
                loop.run_until_complete(self.aclear())
        except Exception as e:
            log.error(f"PgChatMessageHistory.clear failed: {e}")

    async def aclear(self):
        """Clear all messages for this session (serialised with the write lock)."""
        import json
        async with self._get_write_lock():
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT id, data FROM history WHERE data->>'sessionId' = $1 LIMIT 1",
                    self.session_id,
                )
                if row:
                    existing = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
                    existing["messages"] = []
                    await conn.execute(
                        "UPDATE history SET data = $1::jsonb, updated_at = NOW() WHERE id = $2",
                        json.dumps(existing),
                        row["id"],
                    )
