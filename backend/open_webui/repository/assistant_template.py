import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger(__name__)
DATABASE_URL = os.getenv("DATABASE_URL", "")


class AssistantTemplateRepository:
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

    async def list_by_org(self, organization_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, organization_id, name, description, instructions, tags, created_at, updated_at "
                "FROM assistant_templates WHERE organization_id = $1 ORDER BY created_at DESC",
                organization_id,
            )
        return [self._row(r) for r in rows]

    async def get_by_id(self, template_id: str) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, organization_id, name, description, instructions, tags, created_at, updated_at "
                "FROM assistant_templates WHERE id = $1",
                template_id,
            )
        return self._row(row)

    async def create(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        row_id = data.get("_id") or data.get("id") or str(uuid.uuid4())
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO assistant_templates
                    (id, organization_id, name, description, instructions, tags, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id, organization_id, name, description, instructions, tags, created_at, updated_at
                """,
                row_id,
                data.get("organization_id", ""),
                data.get("name", ""),
                data.get("description", ""),
                data.get("instructions", ""),
                data.get("tags") or [],
                now, now,
            )
        return self._row(row)

    async def update(self, template_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE assistant_templates SET
                    name         = COALESCE($2, name),
                    description  = COALESCE($3, description),
                    instructions = COALESCE($4, instructions),
                    tags         = COALESCE($5, tags),
                    updated_at   = $6
                WHERE id = $1
                RETURNING id, organization_id, name, description, instructions, tags, created_at, updated_at
                """,
                template_id,
                data.get("name"), data.get("description"),
                data.get("instructions"), data.get("tags"),
                now,
            )
        return self._row(row)

    async def delete(self, template_id: str) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute("DELETE FROM assistant_templates WHERE id = $1", template_id)
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0


assistant_template_repo = AssistantTemplateRepository()
