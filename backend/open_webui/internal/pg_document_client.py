import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from open_webui.internal.base_document_client import BaseDocumentClient

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")


def _to_str(v: Any) -> str:
    """Convert a value to its string representation for JSONB text comparison."""
    if v is None:
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


def _filter_to_where(filter_dict: Dict[str, Any], params: List) -> str:
    """Translate a MongoDB-style filter dict into a SQL WHERE clause fragment.

    Uses positional parameters ($1, $2, …) and appends values to `params`.
    """
    if not filter_dict:
        return "TRUE"

    conditions = []

    for key, value in filter_dict.items():
        # Logical operators
        if key == "$or":
            sub = [_filter_to_where(f, params) for f in value]
            conditions.append(f"({' OR '.join(sub)})")
            continue
        if key == "$and":
            sub = [_filter_to_where(f, params) for f in value]
            conditions.append(f"({' AND '.join(sub)})")
            continue

        # _id maps to the primary key column
        if key == "_id":
            params.append(_to_str(value))
            conditions.append(f"id = ${len(params)}")
            continue

        # updated_at / created_at are dedicated columns
        if key in ("updated_at", "created_at"):
            if isinstance(value, dict):
                for op, op_val in value.items():
                    params.append(_to_str(op_val))
                    op_map = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<=", "$ne": "!="}
                    sql_op = op_map.get(op, "=")
                    conditions.append(f"{key} {sql_op} ${len(params)}")
            else:
                params.append(_to_str(value))
                conditions.append(f"{key} = ${len(params)}")
            continue

        # All other fields live inside the JSONB data column
        json_path = f"data->>{key!r}"

        if value is None:
            conditions.append(f"({json_path}) IS NULL")
        elif isinstance(value, dict):
            for op, op_val in value.items():
                if op == "$exists":
                    if op_val:
                        conditions.append(f"data ? {key!r}")
                    else:
                        conditions.append(f"NOT (data ? {key!r})")
                elif op == "$in":
                    params.append([_to_str(v) for v in op_val])
                    conditions.append(f"{json_path} = ANY(${len(params)}::text[])")
                elif op == "$nin":
                    params.append([_to_str(v) for v in op_val])
                    conditions.append(f"NOT ({json_path} = ANY(${len(params)}::text[]))")
                elif op in ("$gt", "$gte", "$lt", "$lte", "$ne"):
                    op_map = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<=", "$ne": "!="}
                    params.append(_to_str(op_val))
                    conditions.append(f"{json_path} {op_map[op]} ${len(params)}")
                elif op == "$regex":
                    params.append(op_val)
                    conditions.append(f"{json_path} ~ ${len(params)}")
                else:
                    # Unknown operator — skip
                    log.warning(f"Unsupported filter operator: {op}")
        else:
            if isinstance(value, str):
                # Use JSONB containment so the GIN index on `data` can be used.
                # Semantically equivalent to data->>'key' = 'val' for string fields.
                params.append(json.dumps({key: value}))
                conditions.append(f"data @> ${len(params)}::jsonb")
            else:
                params.append(_to_str(value))
                conditions.append(f"{json_path} = ${len(params)}")

    return " AND ".join(conditions) if conditions else "TRUE"


def _sort_clause(sort_field: str, sort_order: int) -> str:
    direction = "ASC" if sort_order == 1 else "DESC"
    if sort_field in ("updated_at", "created_at"):
        return f"{sort_field} {direction}"
    if sort_field == "_id":
        return f"id {direction}"
    return f"data->>{sort_field!r} {direction}"


def _pipeline_to_sql(collection: str, pipeline: List[Dict]) -> Tuple[str, List]:
    """Translate a MongoDB aggregation pipeline to a SQL query.

    Supports: $match, $sort, $limit, $skip, $group (simple field grouping), $project.
    Complex stages ($addFields date math, $unwind, $filter expressions) fall back to
    a plain SELECT so callers can handle results in Python.
    """
    params: List[Any] = []
    where = "TRUE"
    order_by = "updated_at DESC"
    limit_clause = ""
    skip_clause = ""
    group_field: Optional[str] = None  # the JSONB field used as GROUP BY key
    select = "data"

    for stage in pipeline:
        if "$match" in stage:
            where = _filter_to_where(stage["$match"], params)

        elif "$sort" in stage:
            parts = []
            for field, direction in stage["$sort"].items():
                parts.append(_sort_clause(field, direction))
            if parts:
                order_by = ", ".join(parts)

        elif "$limit" in stage:
            params.append(int(stage["$limit"]))
            limit_clause = f"LIMIT ${len(params)}"

        elif "$skip" in stage:
            params.append(int(stage["$skip"]))
            skip_clause = f"OFFSET ${len(params)}"

        elif "$group" in stage:
            group_spec = stage["$group"]
            group_id = group_spec.get("_id")
            if isinstance(group_id, str) and group_id.startswith("$"):
                field = group_id[1:]
                if field == "_id":
                    group_field = "id"
                    select = f"jsonb_build_object('_id', id, '{field}', id) AS data"
                else:
                    group_field = f"data->>{field!r}"
                    select = f"jsonb_build_object('_id', data->>{field!r}, '{field}', data->>{field!r}) AS data"

        elif "$project" in stage:
            # Build a jsonb_build_object for projected fields
            proj = stage["$project"]
            include = {k for k, v in proj.items() if v and k != "_id"}
            if "_id" in proj and proj["_id"] == 0:
                pass  # suppress _id
            if include:
                pairs = []
                for f in include:
                    pairs.append(f"'{f}', data->>{f!r}")
                select = f"jsonb_build_object({', '.join(pairs)}) AS data"

        else:
            # Unsupported stage — log and continue; result post-processing handles it
            stage_name = list(stage.keys())[0] if stage else "unknown"
            log.debug(f"Pipeline stage '{stage_name}' not translated; returning raw data")

    if group_field:
        sql = f"SELECT {select} FROM {collection} WHERE {where} GROUP BY {group_field} ORDER BY {order_by} {limit_clause} {skip_clause}"
    else:
        sql = f"SELECT {select} FROM {collection} WHERE {where} ORDER BY {order_by} {limit_clause} {skip_clause}"

    return sql.strip(), params


class PgDocumentClient(BaseDocumentClient):
    """PostgreSQL JSONB implementation of BaseDocumentClient.

    Each MongoDB collection maps to a PostgreSQL table with the schema:
        id          TEXT PRIMARY KEY          -- MongoDB _id (stringified)
        data        JSONB NOT NULL            -- full document
        updated_at  TIMESTAMPTZ DEFAULT NOW()
        created_at  TIMESTAMPTZ DEFAULT NOW()

    Tables are expected to exist (created via Alembic migrations).
    The client will auto-create a table on first access if it doesn't exist,
    which is useful during development / initial setup.
    """

    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
        self._connected = False
        self._ensured_tables: set = set()

    async def connect(self):
        try:
            log.info(f"Connecting to PostgreSQL at {DATABASE_URL[:30]}...")
            self._pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            self._connected = True
            log.info("Connected to PostgreSQL")
        except Exception as e:
            log.error(f"Error connecting to PostgreSQL: {e}")
            raise

    async def reconnect(self):
        await self.connect()

    async def close(self):
        try:
            if self._pool:
                await self._pool.close()
            self._connected = False
            log.info("PostgreSQL connection pool closed")
        except Exception as e:
            log.error(f"Error closing PostgreSQL pool: {e}")

    async def _get_pool(self) -> asyncpg.Pool:
        if not self._connected or self._pool is None:
            await self.connect()
        return self._pool

    async def _ensure_table(self, collection: str):
        """Create the collection table if it doesn't exist yet."""
        if collection in self._ensured_tables:
            return
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {collection} (
                    id          TEXT PRIMARY KEY,
                    data        JSONB NOT NULL,
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS {collection}_data_gin
                ON {collection} USING gin(data)
            """)
        self._ensured_tables.add(collection)

    def _doc_to_row(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """Extract (id, json_string) from a document dict."""
        doc_id = data.get("_id") or data.get("id") or str(uuid.uuid4())
        return str(doc_id), json.dumps(data, default=str)

    def _row_to_doc(self, row) -> Dict[str, Any]:
        """Convert an asyncpg Record back to a plain dict."""
        if row is None:
            return None
        data = dict(row["data"]) if isinstance(row["data"], dict) else json.loads(row["data"])
        # Ensure _id is present for callers that expect it
        if "_id" not in data:
            data["_id"] = row["id"]
        return data

    def _apply_projection(self, doc: Dict, projection: Optional[Dict]) -> Dict:
        if not projection:
            return doc
        include = {k for k, v in projection.items() if v and k != "_id"}
        exclude = {k for k, v in projection.items() if not v}
        if include:
            result = {k: doc[k] for k in include if k in doc}
            if projection.get("_id", 1):
                result["_id"] = doc.get("_id")
            return result
        if exclude:
            return {k: v for k, v in doc.items() if k not in exclude}
        return doc

    # ── Document operations ──────────────────────────────────────────────────

    async def query(self, database_name="", collection_name="", query=None) -> List[Dict[str, Any]]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            params: List[Any] = []
            where = _filter_to_where(query or {}, params)
            sql = f"SELECT id, data FROM {collection_name} WHERE {where} ORDER BY updated_at DESC"
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return [self._row_to_doc(r) for r in rows]
        except Exception as e:
            log.error(f"Error querying PostgreSQL ({collection_name}): {e}")
            return []

    async def query_one(self, database_name="", collection_name="", query=None) -> Optional[Dict[str, Any]]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            params: List[Any] = []
            where = _filter_to_where(query or {}, params)
            sql = f"SELECT id, data FROM {collection_name} WHERE {where} ORDER BY updated_at DESC LIMIT 1"
            async with pool.acquire() as conn:
                row = await conn.fetchrow(sql, *params)
            return self._row_to_doc(row)
        except Exception as e:
            log.error(f"Error querying PostgreSQL ({collection_name}): {e}")
            return None

    async def query_one_with_projection(self, database_name="", collection_name="", query=None, projection=None) -> Optional[Dict[str, Any]]:
        doc = await self.query_one(database_name, collection_name, query)
        if doc is None:
            return None
        return self._apply_projection(doc, projection)

    async def query_with_projection(self, database_name="", collection_name="", query=None, projection=None) -> List[Dict[str, Any]]:
        docs = await self.query(database_name, collection_name, query)
        return [self._apply_projection(d, projection) for d in docs]

    async def query_with_sort_and_pagination(self, database_name="", collection_name="", query=None, sort_field="updated_at", sort_order=-1, limit=10, skip=0) -> Dict[str, Any]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            params: List[Any] = []
            where = _filter_to_where(query or {}, params)
            order = _sort_clause(sort_field, sort_order)

            count_sql = f"SELECT COUNT(*) FROM {collection_name} WHERE {where}"
            data_sql = f"SELECT id, data FROM {collection_name} WHERE {where} ORDER BY {order}"

            async with pool.acquire() as conn:
                total = await conn.fetchval(count_sql, *params)
                rows = await conn.fetch(data_sql, *params)

            documents = [self._row_to_doc(r) for r in rows]
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
            log.error(f"Error querying PostgreSQL ({collection_name}): {e}")
            return {"count": 0, "total": 0, "has_more": False, "data": [], "documents": []}

    async def query_one_by_pipeline(self, database_name="", collection_name="", pipeline=None) -> Optional[Dict[str, Any]]:
        results = await self.query_by_pipeline(database_name, collection_name, pipeline)
        return results[0] if results else None

    async def query_by_pipeline(self, database_name="", collection_name="", pipeline=None) -> List[Dict[str, Any]]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            sql, params = _pipeline_to_sql(collection_name, pipeline or [])
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return [self._row_to_doc(r) for r in rows]
        except Exception as e:
            log.error(f"Error in query_by_pipeline ({collection_name}): {e}")
            return []

    async def insert(self, database_name="", collection_name="", data=None) -> Optional[Dict[str, Any]]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            doc = data or {}
            doc_id, doc_json = self._doc_to_row(doc)

            # Store id inside data too for consistency
            parsed = json.loads(doc_json)
            if "_id" not in parsed:
                parsed["_id"] = doc_id
            doc_json = json.dumps(parsed, default=str)

            now = datetime.now(timezone.utc)
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {collection_name} (id, data, updated_at, created_at)
                    VALUES ($1, $2::jsonb, $3, $3)
                    ON CONFLICT (id) DO UPDATE
                        SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at
                    """,
                    doc_id, doc_json, now,
                )
                row = await conn.fetchrow(
                    f"SELECT id, data FROM {collection_name} WHERE id = $1", doc_id
                )
            return self._row_to_doc(row)
        except Exception as e:
            log.error(f"Error inserting to PostgreSQL ({collection_name}): {e}")
            return None

    async def insert_many(self, database_name="", collection_name="", data_array=None) -> Dict[str, Any]:
        try:
            if not data_array or not isinstance(data_array, list):
                raise ValueError("Input must be a non-empty list of objects.")
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            now = datetime.now(timezone.utc)
            inserted_ids = []

            async with pool.acquire() as conn:
                async with conn.transaction():
                    for doc in data_array:
                        doc_id, doc_json = self._doc_to_row(doc)
                        parsed = json.loads(doc_json)
                        if "_id" not in parsed:
                            parsed["_id"] = doc_id
                        await conn.execute(
                            f"""
                            INSERT INTO {collection_name} (id, data, updated_at, created_at)
                            VALUES ($1, $2::jsonb, $3, $3)
                            ON CONFLICT (id) DO UPDATE
                                SET data = EXCLUDED.data, updated_at = EXCLUDED.updated_at
                            """,
                            doc_id, json.dumps(parsed, default=str), now,
                        )
                        inserted_ids.append(doc_id)

            return {"success": True, "inserted_count": len(inserted_ids), "inserted_ids": inserted_ids}
        except Exception as e:
            log.error(f"Error inserting many to PostgreSQL ({collection_name}): {e}")
            raise

    async def update(self, database_name="", collection_name="", query=None, data=None) -> Optional[Dict[str, Any]]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            params: List[Any] = []
            where = _filter_to_where(query or {}, params)

            # Merge patch: update individual JSONB keys using ||
            patch = json.dumps(data or {}, default=str)
            params.append(patch)
            patch_idx = len(params)

            now = datetime.now(timezone.utc)
            params.append(now)
            now_idx = len(params)

            sql = f"""
                UPDATE {collection_name}
                SET data = data || ${patch_idx}::jsonb,
                    updated_at = ${now_idx}
                WHERE {where}
                RETURNING id, data
            """
            async with pool.acquire() as conn:
                row = await conn.fetchrow(sql, *params)
            return self._row_to_doc(row)
        except Exception as e:
            log.error(f"Error updating PostgreSQL ({collection_name}): {e}")
            return None

    async def update_many(self, database_name="", collection_name="", query=None, data=None) -> Dict[str, Any]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            params: List[Any] = []
            where = _filter_to_where(query or {}, params)

            patch = json.dumps(data or {}, default=str)
            params.append(patch)
            patch_idx = len(params)

            now = datetime.now(timezone.utc)
            params.append(now)
            now_idx = len(params)

            sql = f"""
                UPDATE {collection_name}
                SET data = data || ${patch_idx}::jsonb,
                    updated_at = ${now_idx}
                WHERE {where}
            """
            async with pool.acquire() as conn:
                status = await conn.execute(sql, *params)
            # asyncpg returns "UPDATE n"
            modified = int(status.split()[-1]) if status else 0
            return {"matched_count": modified, "modified_count": modified}
        except Exception as e:
            log.error(f"Error updating many in PostgreSQL ({collection_name}): {e}")
            return {"matched_count": 0, "modified_count": 0}

    async def delete_one(self, database_name="", collection_name="", query=None) -> Dict[str, Any]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            params: List[Any] = []
            where = _filter_to_where(query or {}, params)
            # Delete only the first matching row
            sql = f"""
                DELETE FROM {collection_name}
                WHERE id = (
                    SELECT id FROM {collection_name} WHERE {where} LIMIT 1
                )
            """
            async with pool.acquire() as conn:
                status = await conn.execute(sql, *params)
            deleted = int(status.split()[-1]) if status else 0
            return {"deleted_count": deleted}
        except Exception as e:
            log.error(f"Error deleting from PostgreSQL ({collection_name}): {e}")
            return {"deleted_count": 0}

    async def delete_many(self, database_name="", collection_name="", query=None) -> Dict[str, Any]:
        try:
            await self._ensure_table(collection_name)
            pool = await self._get_pool()
            params: List[Any] = []
            where = _filter_to_where(query or {}, params)
            sql = f"DELETE FROM {collection_name} WHERE {where}"
            async with pool.acquire() as conn:
                status = await conn.execute(sql, *params)
            deleted = int(status.split()[-1]) if status else 0
            return {"deleted_count": deleted}
        except Exception as e:
            log.error(f"Error deleting many from PostgreSQL ({collection_name}): {e}")
            return {"deleted_count": 0}
