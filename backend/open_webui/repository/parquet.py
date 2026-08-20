import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "")

_COLS = (
    "id, organization_id, board_id, boarditem_id, folder_id, assistant_id, "
    "file_name, file_type, file_size, s3_path, description, columns, "
    "column_count, row_count, total_records, embedding_chunks, status, "
    "processed_at, deleted_at, created_at, updated_at"
)


class ParquetRepository:
    _pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, command_timeout=30)
        return self._pool

    @staticmethod
    def _row(row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        for f in ("created_at", "updated_at", "deleted_at", "processed_at"):
            if isinstance(d.get(f), datetime):
                d[f] = d[f].isoformat()
        if isinstance(d.get("columns"), str):
            try:
                d["columns"] = json.loads(d["columns"])
            except (json.JSONDecodeError, ValueError):
                pass
        return d

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        row_id = data.get("id") or data.get("_id") or str(uuid.uuid4())
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO parquets ({_COLS})
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)
                RETURNING {_COLS}
                """,
                row_id,
                data.get("organization_id", ""),
                data.get("board_id"),
                data.get("boarditem_id"),
                data.get("folder_id"),
                data.get("assistant_id"),
                data.get("file_name", ""),
                data.get("file_type"),
                data.get("file_size"),
                data.get("s3_path"),
                data.get("description", ""),
                json.dumps(data["columns"]) if isinstance(data.get("columns"), (dict, list)) else data.get("columns"),
                data.get("column_count"),
                data.get("row_count"),
                data.get("total_records"),
                data.get("embedding_chunks"),
                data.get("status"),
                None,  # processed_at
                None,  # deleted_at
                now,
                now,
            )
        return self._row(row)

    async def get_by_id(self, parquet_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_COLS} FROM parquets WHERE id = $1 AND deleted_at IS NULL",
                parquet_id,
            )
        return self._row(row)

    async def get_by_filters(
        self,
        organization_id: Optional[str] = None,
        folder_ids: Optional[List[str]] = None,
        board_ids: Optional[List[str]] = None,
        boarditem_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        conditions = ["deleted_at IS NULL"]
        params: List[Any] = []

        if organization_id:
            params.append(organization_id)
            conditions.append(f"organization_id = ${len(params)}")
        if folder_ids:
            params.append(folder_ids)
            conditions.append(f"folder_id = ANY(${len(params)})")
        if board_ids:
            params.append(board_ids)
            conditions.append(f"board_id = ANY(${len(params)})")
        if boarditem_id:
            params.append(boarditem_id)
            conditions.append(f"boarditem_id = ${len(params)}")

        where = " AND ".join(conditions)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_COLS} FROM parquets WHERE {where} ORDER BY created_at DESC",
                *params,
            )
        return [self._row(r) for r in rows]

    async def update(self, parquet_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        # Build dynamic SET clause for provided fields
        allowed = {
            "file_name", "file_type", "file_size", "s3_path", "description",
            "columns", "column_count", "row_count", "total_records",
            "embedding_chunks", "status", "processed_at", "board_id",
            "boarditem_id", "folder_id", "assistant_id",
        }
        sets = ["updated_at = $1"]
        params: List[Any] = [now]
        for key, val in data.items():
            if key in allowed:
                params.append(val)
                sets.append(f"{key} = ${len(params)}")
        params.append(parquet_id)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE parquets SET {', '.join(sets)} WHERE id = ${len(params)} AND deleted_at IS NULL RETURNING {_COLS}",
                *params,
            )
        return self._row(row)

    async def soft_delete(self, parquet_id: str) -> int:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE parquets SET deleted_at = $1, updated_at = $1 WHERE id = $2 AND deleted_at IS NULL",
                now, parquet_id,
            )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def hard_delete(self, parquet_id: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM parquets WHERE id = $1", parquet_id)
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def get_all_active(self) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT {_COLS} FROM parquets WHERE deleted_at IS NULL ORDER BY created_at DESC")
        return [self._row(r) for r in rows]


parquet_repo = ParquetRepository()
