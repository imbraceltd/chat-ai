import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.encryption import decrypt_dict, encrypt_dict, is_encrypted

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MODELS", logging.INFO))

DATABASE_URL = os.getenv("DATABASE_URL", "")

_COLS = (
    "id, provider_id, organization_id, name, type, source, is_shown, "
    "config, models, metadata, deleted_at, created_at, updated_at"
)


class ProviderRepository:
    _pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5, command_timeout=30)
        return self._pool

    _JSONB_COLS = {"models", "metadata"}

    @staticmethod
    def _row(row) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        d = dict(row)
        for f in ("created_at", "updated_at", "deleted_at"):
            if isinstance(d.get(f), datetime):
                d[f] = d[f].isoformat()
        for f in ProviderRepository._JSONB_COLS:
            if isinstance(d.get(f), str):
                try:
                    d[f] = json.loads(d[f])
                except (json.JSONDecodeError, ValueError):
                    pass
        return d

    def _decrypt(self, provider: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not provider:
            return provider
        config = provider.get("config")
        if config and isinstance(config, str) and is_encrypted(config):
            try:
                provider["config"] = decrypt_dict(config)
            except Exception as e:
                log.error("Failed to decrypt provider config for provider_id=%s: %s", provider.get("provider_id"), e)
                provider["config"] = {}
        elif config and isinstance(config, dict):
            pass  # legacy unencrypted
        return provider

    def _encrypt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "config" not in data:
            return data
        result = data.copy()
        config = result.get("config")
        if config and isinstance(config, dict):
            result["config"] = encrypt_dict(config)
        return result

    async def get_all_by_organization_id(
        self,
        organization_id: str,
        search: str = "",
        skip: int = 0,
        limit: int = -1,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        pool = await self._get_pool()
        conditions = ["organization_id = $1", "deleted_at IS NULL"]
        params: List[Any] = [organization_id]

        if search:
            params.append(f"%{search}%")
            conditions.append(f"name ILIKE ${len(params)}")
        if source:
            params.append(source)
            conditions.append(f"source = ${len(params)}")

        where = " AND ".join(conditions)
        count_sql = f"SELECT COUNT(*) FROM llm_providers WHERE {where}"
        data_sql = f"SELECT {_COLS} FROM llm_providers WHERE {where} ORDER BY created_at DESC"

        async with pool.acquire() as conn:
            total = await conn.fetchval(count_sql, *params)
            rows = await conn.fetch(data_sql, *params)

        documents = [self._decrypt(self._row(r)) for r in rows]
        return {
            "object": "list",
            "count": len(documents),
            "total": total,
            "has_more": skip + len(documents) < total,
            "data": documents,
            "documents": documents,
        }

    async def get_by_provider_id_and_organization_id(
        self, organization_id: str, provider_id: str
    ) -> Optional[Dict[str, Any]]:
        # Match either provider_id or id. Rows migrated from Mongo carry the
        # legacy ObjectID in `id` and a separate UUID in `provider_id`; the
        # frontend sends whichever value it has from its list view.
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_COLS} FROM llm_providers "
                f"WHERE (provider_id = $1 OR id = $1) AND organization_id = $2 AND deleted_at IS NULL LIMIT 1",
                provider_id, organization_id,
            )
        return self._decrypt(self._row(row))

    async def create(self, organization_id: str, data: Dict[str, Any] = None, provider_details: Dict[str, Any] = None) -> Dict[str, Any]:
        data = data or provider_details or {}
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        data.setdefault("source", "custom")
        data.setdefault("is_shown", True)
        data.setdefault("models", [])
        data.setdefault("metadata", {})
        data["organization_id"] = organization_id
        encrypted = self._encrypt(data)
        row_id = data.get("_id") or data.get("id") or str(uuid.uuid4())
        provider_id = data.get("provider_id") or str(uuid.uuid4())

        config_val = encrypted.get("config", "")
        if isinstance(config_val, dict):
            config_val = json.dumps(config_val)

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO llm_providers ({_COLS})
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                RETURNING {_COLS}
                """,
                row_id, provider_id, organization_id,
                encrypted.get("name", ""),
                encrypted.get("type", ""),
                encrypted.get("source", "custom"),
                bool(encrypted.get("is_shown", True)),
                config_val,
                json.dumps(encrypted.get("models", [])),
                json.dumps(encrypted.get("metadata", {})),
                None,  # deleted_at
                now, now,
            )
        return self._decrypt(self._row(row))

    async def update(self, provider_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        data.pop("_id", None)
        encrypted = self._encrypt(data)

        allowed = {"name", "type", "source", "is_shown", "config", "models", "metadata"}
        sets = ["updated_at = $1"]
        params: List[Any] = [now]
        for key, val in encrypted.items():
            if key in allowed:
                if key in ("models", "metadata") and isinstance(val, (dict, list)):
                    val = json.dumps(val)
                params.append(val)
                sets.append(f"{key} = ${len(params)}")

        params.append(provider_id)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE llm_providers SET {', '.join(sets)} "
                f"WHERE (provider_id = ${len(params)} OR id = ${len(params)}) RETURNING {_COLS}",
                *params,
            )
        return self._decrypt(self._row(row))

    async def remove(self, provider_id: str) -> None:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE llm_providers SET deleted_at = $1 WHERE provider_id = $2 OR id = $2",
                now, provider_id,
            )

    async def get_system_provider_by_type(
        self, provider_type: str, organization_id: str = ""
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_COLS} FROM llm_providers WHERE type = $1 AND source = 'system' AND organization_id = $2 AND deleted_at IS NULL LIMIT 1",
                provider_type, organization_id,
            )
        return self._decrypt(self._row(row))
