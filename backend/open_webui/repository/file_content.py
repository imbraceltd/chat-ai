import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import asyncpg

log = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "")


class FileContentRepository:
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
        for f in ("created_at", "updated_at"):
            if isinstance(d.get(f), datetime):
                d[f] = d[f].isoformat()
        return d

    async def get_by_thread_id(self, thread_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, thread_id, organization_id, content, created_at, updated_at "
                "FROM file_content WHERE thread_id = $1 LIMIT 1",
                thread_id,
            )
        return self._row(row)

    async def upsert(self, thread_id: str, content: str, organization_id: str = "") -> bool:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO file_content (id, thread_id, organization_id, content, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
                """,
                str(uuid.uuid4()), thread_id, organization_id, content, now, now,
            )
            # update if already exists by thread_id
            await conn.execute(
                """
                UPDATE file_content
                SET content = $1, organization_id = $2, updated_at = $3
                WHERE thread_id = $4
                """,
                content, organization_id, now, thread_id,
            )
        return True


file_content_repo = FileContentRepository()
