import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

DATABASE_URL = os.getenv("DATABASE_URL", "")

_SCALAR_COLS = (
    "id, assistant_id, organization_id, name, model, model_id, provider_id, "
    "description, instructions, core_task, agent_type, mode, category, channel, "
    "personality_role, response_length, tone_and_style, preload_information, "
    "preload_information_type, workflow_name, workflow_function_call, "
    "credential_name, credential_id, guardrail_id, default_folder_id, "
    "version, object, reasoning_effort, temperature, top_p, "
    "use_history, streaming, show_thinking_process, use_memory, "
    "file_ids, sub_agents, knowledge_hubs, banned_words, board_ids, folder_ids, "
    "metadata, tools, tool_resources, response_format, "
    "deleted_at, created_at, updated_at, document_ai"
)

# Fields allowed in dynamic UPDATE
_UPDATABLE = {
    "name", "model", "model_id", "provider_id", "description", "instructions",
    "core_task", "agent_type", "mode", "category", "channel", "personality_role",
    "response_length", "tone_and_style", "preload_information", "preload_information_type",
    "workflow_name", "workflow_function_call", "credential_name", "credential_id",
    "guardrail_id", "default_folder_id", "version", "object", "reasoning_effort",
    "temperature", "top_p", "use_history", "streaming", "show_thinking_process",
    "use_memory", "file_ids", "sub_agents", "knowledge_hubs", "banned_words",
    "board_ids", "folder_ids", "metadata", "tools", "tool_resources", "response_format",
    "deleted_at", "document_ai",
}
_JSONB_COLS = {"metadata", "tools", "tool_resources", "response_format", "workflow_function_call", "document_ai"}
_ARRAY_COLS = {"file_ids", "sub_agents", "knowledge_hubs", "board_ids", "folder_ids", "category"}
_NOT_NULL_TEXT_COLS = {"name", "model", "description", "instructions", "core_task", "banned_words"}
# Text columns that the frontend sometimes sends as a number (e.g. version=2).
_STRINGIFY_TEXT_COLS = {"version"}


def _to_db(key: str, val: Any) -> Any:
    if key in _NOT_NULL_TEXT_COLS:
        return val or ""
    if key in _ARRAY_COLS:
        if not val:
            return []
        if isinstance(val, str):
            return [w.strip() for w in val.split(",") if w.strip()]
        return list(val)
    if val is None:
        return None
    if key in _JSONB_COLS and isinstance(val, (dict, list)):
        return json.dumps(val)
    if key in _STRINGIFY_TEXT_COLS and not isinstance(val, str):
        return str(val)
    return val


class AssistantRepository:
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
        for f in ("created_at", "updated_at", "deleted_at"):
            if isinstance(d.get(f), datetime):
                d[f] = d[f].isoformat()
        for f in _JSONB_COLS:
            if isinstance(d.get(f), str):
                try:
                    d[f] = json.loads(d[f])
                except (json.JSONDecodeError, ValueError):
                    pass
        return d

    async def get_all_by_organization_id(
        self,
        organization_id: str,
        search: str = "",
        skip: int = 0,
        limit: int = -1,
    ) -> Dict[str, Any]:
        pool = await self._get_pool()
        conditions = ["organization_id = $1", "deleted_at IS NULL"]
        params: List[Any] = [organization_id]

        if search:
            params.append(f"%{search}%")
            conditions.append(f"name ILIKE ${len(params)}")

        where = " AND ".join(conditions)
        count_sql = f"SELECT COUNT(*) FROM openai_assistants WHERE {where}"
        data_sql = f"SELECT {_SCALAR_COLS} FROM openai_assistants WHERE {where} ORDER BY created_at DESC"

        async with pool.acquire() as conn:
            total = await conn.fetchval(count_sql, *params)
            rows = await conn.fetch(data_sql, *params)

        documents = [self._row(r) for r in rows]
        return {
            "object": "list",
            "count": len(documents),
            "total": total,
            "has_more": skip + len(documents) < total,
            "data": documents,
            "documents": documents,
        }

    async def get_all(self) -> List[Dict[str, Any]]:
        """All non-deleted assistants across every organization.

        Used by the startup RAG→openai_files migration, which previously read
        this via the generic JSONB document client (`SELECT id, data`) — that
        breaks now that `openai_assistants` is a typed relational table.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_SCALAR_COLS} FROM openai_assistants WHERE deleted_at IS NULL"
            )
        return [self._row(r) for r in rows]

    async def get_by_assistant_id_and_organization_id(
        self, organization_id: str, assistant_id: str
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_SCALAR_COLS} FROM openai_assistants "
                "WHERE assistant_id = $1 AND organization_id = $2 AND deleted_at IS NULL LIMIT 1",
                assistant_id, organization_id,
            )
        return self._row(row)

    async def get_by_assistant_name_and_organization_id(
        self, name: str, organization_id: str
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT {_SCALAR_COLS} FROM openai_assistants "
                "WHERE name = $1 AND organization_id = $2 AND deleted_at IS NULL LIMIT 1",
                name, organization_id,
            )
        return self._row(row)

    async def create(self, organization_id: str, data: Dict[str, Any] = None, assistant_details: Dict[str, Any] = None) -> Dict[str, Any]:
        data = data or assistant_details or {}
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        data["organization_id"] = organization_id
        row_id = data.get("_id") or data.get("id") or str(uuid.uuid4())
        assistant_id = data.get("assistant_id") or str(uuid.uuid4())

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                INSERT INTO openai_assistants ({_SCALAR_COLS})
                VALUES (
                    $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
                    $19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,
                    $35,$36,$37,$38,$39,$40,$41,$42,$43,$44,$45,$46,$47,$48
                )
                RETURNING {_SCALAR_COLS}
                """,
                row_id,
                assistant_id,
                organization_id,
                data.get("name") or "",
                data.get("model") or "",
                data.get("model_id"),
                data.get("provider_id"),
                data.get("description") or "",
                data.get("instructions") or "",
                data.get("core_task") or "",
                data.get("agent_type"),
                data.get("mode"),
                _to_db("category", data.get("category")) or [],
                data.get("channel"),
                data.get("personality_role"),
                data.get("response_length"),
                data.get("tone_and_style"),
                data.get("preload_information"),
                data.get("preload_information_type"),
                data.get("workflow_name"),
                json.dumps(data["workflow_function_call"]) if isinstance(data.get("workflow_function_call"), (dict, list)) else data.get("workflow_function_call"),
                data.get("credential_name"),
                data.get("credential_id"),
                data.get("guardrail_id"),
                data.get("default_folder_id"),
                _to_db("version", data.get("version")),
                data.get("object"),
                data.get("reasoning_effort"),
                data.get("temperature"),
                data.get("top_p"),
                bool(data.get("use_history", False)),
                bool(data.get("streaming", False)),
                bool(data.get("show_thinking_process", False)),
                bool(data.get("use_memory", False)),
                _to_db("file_ids", data.get("file_ids")) or [],
                _to_db("sub_agents", data.get("sub_agents")) or [],
                _to_db("knowledge_hubs", data.get("knowledge_hubs")) or [],
                _to_db("banned_words", data.get("banned_words")) or "",
                _to_db("board_ids", data.get("board_ids")) or [],
                _to_db("folder_ids", data.get("folder_ids")) or [],
                json.dumps(data["metadata"]) if isinstance(data.get("metadata"), (dict, list)) else data.get("metadata"),
                json.dumps(data["tools"]) if isinstance(data.get("tools"), (dict, list)) else data.get("tools"),
                json.dumps(data["tool_resources"]) if isinstance(data.get("tool_resources"), (dict, list)) else data.get("tool_resources"),
                json.dumps(data["response_format"]) if isinstance(data.get("response_format"), (dict, list)) else data.get("response_format"),
                None,  # deleted_at
                now,
                now,
                json.dumps(data["document_ai"]) if isinstance(data.get("document_ai"), (dict, list)) else data.get("document_ai"),
            )
        return self._row(row)

    async def update(self, assistant_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        data.pop("_id", None)

        sets = ["updated_at = $1"]
        params: List[Any] = [now]
        for key, val in data.items():
            if key in _UPDATABLE:
                params.append(_to_db(key, val))
                sets.append(f"{key} = ${len(params)}")

        params.append(assistant_id)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE openai_assistants SET {', '.join(sets)} WHERE assistant_id = ${len(params)} RETURNING {_SCALAR_COLS}",
                *params,
            )
        return self._row(row)

    async def remove(self, assistant_id: str) -> None:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE openai_assistants SET deleted_at = $1 WHERE assistant_id = $2",
                now, assistant_id,
            )

    async def get_by_assistant_ids_and_organization_id(
        self, organization_id: str, assistant_ids: List[str]
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_SCALAR_COLS} FROM openai_assistants "
                "WHERE assistant_id = ANY($1) AND organization_id = $2 AND deleted_at IS NULL",
                assistant_ids, organization_id,
            )
        return [self._row(r) for r in rows]

    async def get_agents(
        self, organization_id: str, assistant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        conditions = [
            "organization_id = $1",
            "(agent_type IS NULL OR agent_type != 'team_lead')",
            "deleted_at IS NULL",
        ]
        params: List[Any] = [organization_id]
        if assistant_id:
            params.append(assistant_id)
            conditions.append(f"assistant_id != ${len(params)}")

        where = " AND ".join(conditions)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_SCALAR_COLS} FROM openai_assistants WHERE {where}",
                *params,
            )
        return [self._row(r) for r in rows]

    async def update_assistant_instructions(
        self, assistant_id: str, instructions: str, core_task: str
    ) -> Optional[Dict[str, Any]]:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE openai_assistants SET instructions = $1, core_task = $2, updated_at = $3 "
                f"WHERE assistant_id = $4 RETURNING {_SCALAR_COLS}",
                instructions, core_task, now, assistant_id,
            )
        return self._row(row)

    async def get_by_assistant_ids(self, assistant_ids: List[str]) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_SCALAR_COLS} FROM openai_assistants WHERE assistant_id = ANY($1) AND deleted_at IS NULL",
                assistant_ids,
            )
        return [self._row(r) for r in rows]

    async def get_by_sub_agent_id_and_organization_id(
        self, organization_id: str, sub_agent_id: str
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT {_SCALAR_COLS} FROM openai_assistants "
                "WHERE $1 = ANY(sub_agents) AND organization_id = $2 AND deleted_at IS NULL",
                sub_agent_id, organization_id,
            )
        return [self._row(r) for r in rows]

    async def get_assistant_meta_by_organization_id(
        self, organization_id: str
    ) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT assistant_id, name FROM openai_assistants "
                "WHERE organization_id = $1 AND deleted_at IS NULL",
                organization_id,
            )
        return [dict(r) for r in rows]
