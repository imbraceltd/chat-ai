import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "")


class EchartRepository:
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
        if isinstance(d.get("echart_data"), str):
            try:
                d["echart_data"] = json.loads(d["echart_data"])
            except (json.JSONDecodeError, ValueError):
                pass
        return d

    async def save(self, thread_id: str, echart_data: Dict[str, Any], organization_id: str = "") -> Optional[str]:
        pool = await self._get_pool()
        echart_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO echarts (id, thread_id, organization_id, echart_data, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                echart_id,
                thread_id,
                organization_id,
                json.dumps(echart_data),
                now,
                now,
            )
        return echart_id

    async def get_by_thread_id(self, thread_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, thread_id, organization_id, echart_data, created_at, updated_at
                FROM echarts
                WHERE thread_id = $1
                ORDER BY created_at DESC
                """,
                thread_id,
            )
        return [self._row(r) for r in rows]


echart_repo = EchartRepository()
