import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.encryption import encrypt_dict, decrypt_dict, is_encrypted

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MODELS", logging.INFO))

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Columns selected/returned for a provider row (excludes internal `id`/`deleted_at`).
_RETURN_COLS = """
    guardrail_provider_id, organization_id, name, type, source,
    is_shown, config, models_metadata, created_at, updated_at
"""


async def _init_conn(conn: asyncpg.Connection) -> None:
    """Register a JSONB codec so `models_metadata` round-trips as a dict."""
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )


# Module-level connection pool shared across all repository instances.
# Several callers (safety.py, guardrail.py) construct a new
# GuardrailProviderRepository() per request, so the pool must NOT live on the
# instance — otherwise each call would open its own pool and leak connections.
_pool: Optional[asyncpg.Pool] = None


async def _get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=5,
            command_timeout=30,
            init=_init_conn,
        )
    return _pool


class GuardrailProviderRepository:
    """
    Repository for per-organization custom guardrail provider configurations.

    Backed by the relational `guardrail_providers` table (no JSONB `data`
    blob — see migration f1a2b3c4d5e6). Columns:
        id, guardrail_provider_id, organization_id, name, type, source,
        is_shown, config, models_metadata, deleted_at, created_at, updated_at

    `config` is stored as a Fernet-encrypted string; `models_metadata` is a
    plain JSONB object keyed by model name (not encrypted).
    """

    async def _get_pool(self) -> asyncpg.Pool:
        return await _get_pool()

    # ── helpers ────────────────────────────────────────────────────────────

    def _encrypt_config(self, details: Dict[str, Any]) -> Dict[str, Any]:
        if "config" not in details:
            return details
        result = details.copy()
        config = result.get("config")
        if config and isinstance(config, dict):
            result["config"] = encrypt_dict(config)
        return result

    def _decrypt_config(
        self, provider: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not provider:
            return provider
        config = provider.get("config")
        if config and isinstance(config, str) and is_encrypted(config):
            try:
                provider["config"] = decrypt_dict(config)
            except Exception as e:
                log.error(
                    "Failed to decrypt guardrail provider config for id=%s: %s",
                    provider.get("guardrail_provider_id"),
                    e,
                )
                provider["config"] = {}
        # Normalise empty/unparsable config to a dict for the response model.
        if not isinstance(provider.get("config"), dict):
            provider["config"] = {}
        return provider

    def _row_to_dict(self, row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        if d.get("models_metadata") is None:
            d["models_metadata"] = {}
        for field in ("created_at", "updated_at"):
            v = d.get(field)
            if isinstance(v, datetime):
                d[field] = v.isoformat()
        return self._decrypt_config(d)

    # ── CRUD ─────────────────────────────────────────────────────────────────

    async def get_all_by_organization_id(
        self,
        organization_id: str,
        search: str = "",
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        params: List[Any] = [organization_id]
        sql = f"""
            SELECT {_RETURN_COLS}
            FROM guardrail_providers
            WHERE organization_id = $1 AND deleted_at IS NULL
        """
        if search:
            params.append(f"%{search}%")
            sql += f" AND name ILIKE ${len(params)}"
        sql += " ORDER BY created_at DESC"

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return [self._row_to_dict(r) for r in rows]
        except Exception as e:
            log.error(f"Error querying guardrail providers: {e}")
            raise

    async def get_by_id(
        self, organization_id: str, guardrail_provider_id: str
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT {_RETURN_COLS}
                    FROM guardrail_providers
                    WHERE guardrail_provider_id = $1
                      AND organization_id = $2
                      AND deleted_at IS NULL
                    LIMIT 1
                    """,
                    guardrail_provider_id,
                    organization_id,
                )
            return self._row_to_dict(row)
        except Exception as e:
            log.error(f"Error getting guardrail provider by id: {e}")
            raise

    async def create(
        self, organization_id: str, details: Dict[str, Any]
    ) -> Dict[str, Any]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        details = self._encrypt_config(details)

        row_id = details.get("_id") or details.get("id") or uuid.uuid4().hex
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO guardrail_providers (
                        id, guardrail_provider_id, organization_id, name, type,
                        source, is_shown, config, models_metadata,
                        created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                    )
                    RETURNING {_RETURN_COLS}
                    """,
                    row_id,
                    details["guardrail_provider_id"],
                    organization_id,
                    details.get("name", ""),
                    details.get("type", "custom"),
                    details.get("source", "custom"),
                    details.get("is_shown", True),
                    details.get("config", ""),
                    details.get("models_metadata") or {},
                    now,
                    now,
                )
            return self._row_to_dict(row)
        except Exception as e:
            log.error(f"Error creating guardrail provider: {e}")
            raise

    async def update(
        self, guardrail_provider_id: str, details: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        details = self._encrypt_config(details)
        details.pop("_id", None)

        # Only update the columns actually provided.
        column_map = {
            "name": "name",
            "type": "type",
            "config": "config",
            "is_shown": "is_shown",
            "models_metadata": "models_metadata",
        }
        set_clauses: List[str] = []
        params: List[Any] = [guardrail_provider_id]
        for key, column in column_map.items():
            if key in details:
                params.append(details[key])
                set_clauses.append(f"{column} = ${len(params)}")

        params.append(datetime.now(timezone.utc))
        set_clauses.append(f"updated_at = ${len(params)}")

        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    UPDATE guardrail_providers
                    SET {', '.join(set_clauses)}
                    WHERE guardrail_provider_id = $1 AND deleted_at IS NULL
                    RETURNING {_RETURN_COLS}
                    """,
                    *params,
                )
            return self._row_to_dict(row)
        except Exception as e:
            log.error(f"Error updating guardrail provider: {e}")
            raise

    async def remove(self, guardrail_provider_id: str) -> None:
        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE guardrail_providers
                    SET deleted_at = $2, updated_at = $2
                    WHERE guardrail_provider_id = $1 AND deleted_at IS NULL
                    """,
                    guardrail_provider_id,
                    datetime.now(timezone.utc),
                )
        except Exception as e:
            log.error(f"Error removing guardrail provider: {e}")
            raise
