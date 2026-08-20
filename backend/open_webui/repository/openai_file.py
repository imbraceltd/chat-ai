import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "")

_COLS = (
    "id, file_id, organization_id, assistant_id, board_id, boarditem_id, "
    "filename, file_name, file_type, file_size, url, bytes, object, "
    "purpose, status, status_details, expires_at, deleted_at, created_at, updated_at"
)


class OpenAIFileRepository:
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
        for f in ("created_at", "updated_at", "deleted_at", "expires_at"):
            if isinstance(d.get(f), datetime):
                d[f] = d[f].isoformat()
        return d

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pool = await self._get_pool()
        now_str = datetime.utcnow().isoformat()
        # Support legacy timestamp-as-int for created_at
        created_at_raw = data.get("created_at")
        if isinstance(created_at_raw, (int, float)):
            try:
                created_at = datetime.fromtimestamp(created_at_raw, tz=timezone.utc)
            except Exception:
                created_at = datetime.now(timezone.utc)
        else:
            created_at = datetime.now(timezone.utc)

        row_id = data.get("_id") or data.get("id") or str(uuid.uuid4())
        file_id = data.get("file_id") or data.get("id", "")

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO openai_files ({_COLS})
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20)
                RETURNING {_COLS}
                """,
                row_id,
                file_id,
                data.get("organization_id") or data.get("org_id"),
                data.get("assistant_id"),
                data.get("board_id"),
                data.get("boarditem_id"),
                data.get("filename", ""),
                data.get("file_name"),
                data.get("file_type"),
                data.get("file_size"),
                data.get("url"),
                data.get("bytes"),
                data.get("object"),
                data.get("purpose"),
                data.get("status"),
                data.get("status_details"),
                None,   # expires_at
                None,   # deleted_at
                created_at,
                created_at,
            )
        return self._row(row)

    async def get_all(self, file_ids: List[str]) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_COLS} FROM openai_files WHERE file_id = ANY($1) AND deleted_at IS NULL",
                file_ids,
            )
        return [self._row(r) for r in rows]

    async def get_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_COLS} FROM openai_files WHERE file_id = $1 AND deleted_at IS NULL LIMIT 1",
                file_id,
            )
        return self._row(row)

    async def update(self, file_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        allowed = {"assistant_id", "board_id", "boarditem_id", "filename", "file_name",
                   "file_type", "file_size", "url", "bytes", "object", "purpose",
                   "status", "status_details"}
        sets = ["updated_at = $1"]
        params: List[Any] = [now]
        for key, val in data.items():
            if key in allowed:
                params.append(val)
                sets.append(f"{key} = ${len(params)}")
        params.append(file_id)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE openai_files SET {', '.join(sets)} WHERE file_id = ${len(params)} AND deleted_at IS NULL RETURNING {_COLS}",
                *params,
            )
        return self._row(row)

    async def update_assistant_id(self, file_ids: List[str], assistant_id: str) -> int:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE openai_files SET assistant_id = $1, updated_at = $2 WHERE file_id = ANY($3) AND deleted_at IS NULL",
                assistant_id, now, file_ids,
            )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def soft_delete(self, file_id: str) -> int:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE openai_files SET deleted_at = $1, updated_at = $1 WHERE file_id = $2 AND deleted_at IS NULL",
                now, file_id,
            )
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def hard_delete(self, file_id: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM openai_files WHERE file_id = $1", file_id)
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0

    async def get_all_active(self) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT {_COLS} FROM openai_files WHERE deleted_at IS NULL ORDER BY created_at DESC")
        return [self._row(r) for r in rows]

    async def get_by_organization(self, org_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_COLS} FROM openai_files WHERE organization_id = $1 AND deleted_at IS NULL ORDER BY created_at DESC",
                org_id,
            )
        return [self._row(r) for r in rows]


openai_file_repo = OpenAIFileRepository()
