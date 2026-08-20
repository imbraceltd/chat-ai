import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "")


class BoardMessageRepository:
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
        if isinstance(d.get("created_at"), datetime):
            d["created_at"] = d["created_at"].isoformat()
        return d

    async def create(self, conversation_id: str, board_item_id: str, organization_id: str, created_at: str = "") -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        try:
            ts = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else datetime.now(timezone.utc)
        except ValueError:
            ts = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO board_messages (id, conversation_id, board_item_id, organization_id, created_at)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, conversation_id, board_item_id, organization_id, created_at
                """,
                str(uuid.uuid4()), conversation_id, board_item_id, organization_id, ts,
            )
        return self._row(row)

    async def get_closest_by_conversation_id(
        self, conversation_id: str, organization_id: str, created_at: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Return the board message whose created_at is closest to the given timestamp."""
        pool = await self._get_pool()
        try:
            target = datetime.fromisoformat(created_at.replace("Z", "+00:00")) if created_at else datetime.now(timezone.utc)
        except ValueError:
            target = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, conversation_id, board_item_id, organization_id, created_at
                FROM board_messages
                WHERE conversation_id = $1 AND organization_id = $2
                ORDER BY ABS(EXTRACT(EPOCH FROM (created_at - $3::timestamptz)))
                LIMIT 1
                """,
                conversation_id, organization_id, target,
            )
        return self._row(row)


board_message_repo = BoardMessageRepository()
