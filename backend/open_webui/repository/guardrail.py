import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")


class GuardrailRepository:
    """
    Repository for the `guardrails` table — proper relational schema.

    Each row has individual typed columns (no JSONB data blob):
        id, name, model, org_id, description, instructions,
        unsafe_categories, competitor_keywords, custom_unsafe_patterns,
        guardrails_config_id, guardrail_provider_id, created_at, updated_at
    """

    _pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                DATABASE_URL, min_size=1, max_size=5, command_timeout=30
            )
        return self._pool

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        if row is None:
            return None
        d = dict(row)
        # Normalise timestamps to ISO strings for API layer
        for field in ("created_at", "updated_at"):
            v = d.get(field)
            if isinstance(v, datetime):
                d[field] = v.isoformat()
        return d

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def list_by_org(self, org_id: str) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, model, org_id, description, instructions,
                       unsafe_categories, competitor_keywords,
                       custom_unsafe_patterns, guardrails_config_id,
                       guardrail_provider_id, created_at, updated_at
                FROM guardrails
                WHERE org_id = $1
                ORDER BY created_at DESC
                """,
                org_id,
            )
        return [self._row_to_dict(r) for r in rows]

    async def list_by_model(self, model: str) -> List[Dict[str, Any]]:
        """All guardrails across every org for a given model (e.g. 'nim-nemo').

        Used by the NeMo guardrails loader at startup, which previously read
        this via the generic JSONB document client (`SELECT id, data`) — that
        breaks now that `guardrails` is a typed relational table with no `data`
        column.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, name, model, org_id, description, instructions,
                       unsafe_categories, competitor_keywords,
                       custom_unsafe_patterns, guardrails_config_id,
                       guardrail_provider_id, created_at, updated_at
                FROM guardrails
                WHERE model = $1
                ORDER BY created_at DESC
                """,
                model,
            )
        return [self._row_to_dict(r) for r in rows]

    async def get_by_config_id(
        self, guardrails_config_id: str, org_id: str
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, model, org_id, description, instructions,
                       unsafe_categories, competitor_keywords,
                       custom_unsafe_patterns, guardrails_config_id,
                       guardrail_provider_id, created_at, updated_at
                FROM guardrails
                WHERE guardrails_config_id = $1 AND org_id = $2
                LIMIT 1
                """,
                guardrails_config_id,
                org_id,
            )
        return self._row_to_dict(row)

    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        row_id = data.get("_id") or data.get("id") or str(uuid.uuid4())

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO guardrails (
                    id, name, model, org_id, description, instructions,
                    unsafe_categories, competitor_keywords,
                    custom_unsafe_patterns, guardrails_config_id,
                    guardrail_provider_id, created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6,
                    $7, $8,
                    $9, $10,
                    $11, $12, $13
                )
                RETURNING id, name, model, org_id, description, instructions,
                          unsafe_categories, competitor_keywords,
                          custom_unsafe_patterns, guardrails_config_id,
                          guardrail_provider_id, created_at, updated_at
                """,
                row_id,
                data.get("name", ""),
                data.get("model", ""),
                data.get("org_id", ""),
                data.get("description", ""),
                data.get("instructions", ""),
                data.get("unsafe_categories") or [],
                data.get("competitor_keywords") or [],
                data.get("custom_unsafe_patterns") or [],
                data.get("guardrails_config_id"),
                data.get("guardrail_provider_id"),
                now,
                now,
            )
        return self._row_to_dict(row)

    async def update(
        self, guardrails_config_id: str, org_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE guardrails SET
                    name                  = COALESCE($3, name),
                    model                 = COALESCE($4, model),
                    description           = COALESCE($5, description),
                    instructions          = COALESCE($6, instructions),
                    unsafe_categories     = COALESCE($7, unsafe_categories),
                    competitor_keywords   = COALESCE($8, competitor_keywords),
                    custom_unsafe_patterns = COALESCE($9, custom_unsafe_patterns),
                    updated_at            = $10
                WHERE guardrails_config_id = $1 AND org_id = $2
                RETURNING id, name, model, org_id, description, instructions,
                          unsafe_categories, competitor_keywords,
                          custom_unsafe_patterns, guardrails_config_id,
                          guardrail_provider_id, created_at, updated_at
                """,
                guardrails_config_id,
                org_id,
                data.get("name"),
                data.get("model"),
                data.get("description"),
                data.get("instructions"),
                data.get("unsafe_categories"),
                data.get("competitor_keywords"),
                data.get("custom_unsafe_patterns"),
                now,
            )
        return self._row_to_dict(row)

    async def delete(self, guardrails_config_id: str, org_id: str) -> int:
        """Returns number of deleted rows."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM guardrails WHERE guardrails_config_id = $1 AND org_id = $2",
                guardrails_config_id,
                org_id,
            )
        # result is e.g. "DELETE 1"
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0


guardrail_repo = GuardrailRepository()
