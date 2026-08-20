"""Replace remaining JSONB tables with proper relational schemas

Revision ID: d1e2f3a4b5c6
Revises: c1d2e3f4a5b6
Create Date: 2026-03-23 00:00:00.000000

Converts all remaining generic JSONB tables to typed relational tables:
  file_content, board_messages, echarts, assistant_templates,
  parquets, openai_files, llm_providers, openai_assistants

Also merges the legacy `guardrail` table into `guardrails` and drops it.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TEXT

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_bool(col: str) -> str:
    return f"CASE WHEN data->>{col!r} = 'true' THEN true ELSE false END"


def _safe_int(col: str) -> str:
    return f"CASE WHEN data->>{col!r} ~ '^[0-9]+$' THEN (data->>{col!r})::integer ELSE NULL END"


def _safe_bigint(col: str) -> str:
    return f"CASE WHEN data->>{col!r} ~ '^[0-9]+$' THEN (data->>{col!r})::bigint ELSE NULL END"


def _safe_float(col: str) -> str:
    return (
        f"CASE WHEN data->>{col!r} ~ '^-?[0-9]+(\\.[0-9]+)?$' "
        f"THEN (data->>{col!r})::float ELSE NULL END"
    )


def _safe_ts(col: str) -> str:
    return (
        f"CASE WHEN data->>{col!r} IS NOT NULL "
        f"AND data->>{col!r} NOT IN ('null', '') "
        f"THEN (data->>{col!r})::timestamptz ELSE NULL END"
    )


def _safe_arr(col: str) -> str:
    return (
        f"CASE "
        f"WHEN data->'{col}' IS NOT NULL AND jsonb_typeof(data->'{col}') = 'array' "
        f"THEN ARRAY(SELECT jsonb_array_elements_text(data->'{col}')) "
        f"WHEN data->'{col}' IS NOT NULL AND jsonb_typeof(data->'{col}') = 'string' "
        f"AND TRIM(data->>'{col}') <> '' "
        f"THEN ARRAY(SELECT TRIM(e) FROM unnest(string_to_array(data->>'{col}', ',')) AS e WHERE TRIM(e) <> '') "
        f"ELSE '{{}}' END"
    )


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # ── 1. file_content ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE file_content_new (
            id              TEXT PRIMARY KEY,
            thread_id       TEXT NOT NULL DEFAULT '',
            organization_id TEXT NOT NULL DEFAULT '',
            content         TEXT NOT NULL DEFAULT '',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(f"""
        INSERT INTO file_content_new
            (id, thread_id, organization_id, content, created_at, updated_at)
        SELECT
            id,
            COALESCE(data->>'thread_id', ''),
            COALESCE(data->>'organization_id', ''),
            COALESCE(data->>'content', ''),
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM file_content
    """)
    op.drop_table("file_content")
    op.rename_table("file_content_new", "file_content")
    op.create_index("file_content_thread_id_idx", "file_content", ["thread_id"])

    # ── 2. board_messages ────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE board_messages_new (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL DEFAULT '',
            board_item_id   TEXT NOT NULL DEFAULT '',
            organization_id TEXT NOT NULL DEFAULT '',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        INSERT INTO board_messages_new
            (id, conversation_id, board_item_id, organization_id, created_at)
        SELECT
            id,
            COALESCE(data->>'conversation_id', ''),
            COALESCE(data->>'board_item_id', ''),
            COALESCE(data->>'organization_id', ''),
            COALESCE(created_at, NOW())
        FROM board_messages
    """)
    op.drop_table("board_messages")
    op.rename_table("board_messages_new", "board_messages")
    op.create_index("board_messages_conversation_id_idx", "board_messages", ["conversation_id"])
    op.create_index("board_messages_org_id_idx", "board_messages", ["organization_id"])

    # ── 3. echarts ───────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE echarts_new (
            id              TEXT PRIMARY KEY,
            thread_id       TEXT NOT NULL DEFAULT '',
            organization_id TEXT NOT NULL DEFAULT '',
            echart_data     JSONB,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        INSERT INTO echarts_new
            (id, thread_id, organization_id, echart_data, created_at, updated_at)
        SELECT
            id,
            COALESCE(data->>'thread_id', ''),
            COALESCE(data->>'organization_id', ''),
            data->'echart_data',
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM echarts
    """)
    op.drop_table("echarts")
    op.rename_table("echarts_new", "echarts")
    op.create_index("echarts_thread_id_idx", "echarts", ["thread_id"])

    # ── 4. assistant_templates ───────────────────────────────────────────────
    op.execute("""
        CREATE TABLE assistant_templates_new (
            id              TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL DEFAULT '',
            name            TEXT NOT NULL DEFAULT '',
            description     TEXT NOT NULL DEFAULT '',
            instructions    TEXT NOT NULL DEFAULT '',
            tags            TEXT[] NOT NULL DEFAULT '{}',
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(f"""
        INSERT INTO assistant_templates_new
            (id, organization_id, name, description, instructions, tags, created_at, updated_at)
        SELECT
            id,
            COALESCE(data->>'organization_id', ''),
            COALESCE(data->>'name', ''),
            COALESCE(data->>'description', ''),
            COALESCE(data->>'instructions', ''),
            {_safe_arr('tags')},
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM assistant_templates
    """)
    op.drop_table("assistant_templates")
    op.rename_table("assistant_templates_new", "assistant_templates")
    op.create_index("assistant_templates_org_id_idx", "assistant_templates", ["organization_id"])

    # ── 5. parquets ──────────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE parquets_new (
            id               TEXT PRIMARY KEY,
            organization_id  TEXT NOT NULL DEFAULT '',
            board_id         TEXT,
            boarditem_id     TEXT,
            folder_id        TEXT,
            assistant_id     TEXT,
            file_name        TEXT NOT NULL DEFAULT '',
            file_type        TEXT,
            file_size        BIGINT,
            s3_path          TEXT,
            description      TEXT NOT NULL DEFAULT '',
            columns          JSONB,
            column_count     INTEGER,
            row_count        INTEGER,
            total_records    INTEGER,
            embedding_chunks INTEGER,
            status           TEXT,
            processed_at     TIMESTAMPTZ,
            deleted_at       TIMESTAMPTZ,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(f"""
        INSERT INTO parquets_new (
            id, organization_id, board_id, boarditem_id, folder_id, assistant_id,
            file_name, file_type, file_size, s3_path, description, columns,
            column_count, row_count, total_records, embedding_chunks, status,
            processed_at, deleted_at, created_at, updated_at
        )
        SELECT
            id,
            COALESCE(data->>'organization_id', ''),
            data->>'board_id',
            data->>'boarditem_id',
            data->>'folder_id',
            data->>'assistant_id',
            COALESCE(data->>'file_name', ''),
            data->>'file_type',
            {_safe_bigint('file_size')},
            data->>'s3_path',
            COALESCE(data->>'description', ''),
            data->'columns',
            {_safe_int('column_count')},
            {_safe_int('row_count')},
            {_safe_int('total_records')},
            {_safe_int('embedding_chunks')},
            data->>'status',
            {_safe_ts('processed_at')},
            {_safe_ts('deleted_at')},
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM parquets
    """)
    op.drop_table("parquets")
    op.rename_table("parquets_new", "parquets")
    op.create_index("parquets_org_id_idx", "parquets", ["organization_id"])
    op.create_index("parquets_board_id_idx", "parquets", ["board_id"])
    op.create_index("parquets_folder_id_idx", "parquets", ["folder_id"])

    # ── 6. openai_files ──────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE openai_files_new (
            id             TEXT PRIMARY KEY,
            file_id        TEXT NOT NULL DEFAULT '',
            organization_id TEXT,
            assistant_id   TEXT,
            board_id       TEXT,
            boarditem_id   TEXT,
            filename       TEXT NOT NULL DEFAULT '',
            file_name      TEXT,
            file_type      TEXT,
            file_size      BIGINT,
            url            TEXT,
            bytes          BIGINT,
            object         TEXT,
            purpose        TEXT,
            status         TEXT,
            status_details TEXT,
            expires_at     TIMESTAMPTZ,
            deleted_at     TIMESTAMPTZ,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(f"""
        INSERT INTO openai_files_new (
            id, file_id, organization_id, assistant_id, board_id, boarditem_id,
            filename, file_name, file_type, file_size, url, bytes, object,
            purpose, status, status_details, expires_at, deleted_at,
            created_at, updated_at
        )
        SELECT
            id,
            COALESCE(data->>'file_id', ''),
            data->>'organization_id',
            data->>'assistant_id',
            data->>'board_id',
            data->>'boarditem_id',
            COALESCE(data->>'filename', ''),
            data->>'file_name',
            data->>'file_type',
            {_safe_bigint('file_size')},
            data->>'url',
            {_safe_bigint('bytes')},
            data->>'object',
            data->>'purpose',
            data->>'status',
            data->>'status_details',
            {_safe_ts('expires_at')},
            {_safe_ts('deleted_at')},
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM openai_files
    """)
    op.drop_table("openai_files")
    op.rename_table("openai_files_new", "openai_files")
    op.create_index("openai_files_file_id_idx", "openai_files", ["file_id"])
    op.create_index("openai_files_org_id_idx", "openai_files", ["organization_id"])
    op.create_index("openai_files_assistant_id_idx", "openai_files", ["assistant_id"])

    # ── 7. llm_providers ─────────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE llm_providers_new (
            id              TEXT PRIMARY KEY,
            provider_id     TEXT NOT NULL DEFAULT '',
            organization_id TEXT NOT NULL DEFAULT '',
            name            TEXT NOT NULL DEFAULT '',
            type            TEXT NOT NULL DEFAULT '',
            source          TEXT NOT NULL DEFAULT 'custom',
            is_shown        BOOLEAN NOT NULL DEFAULT TRUE,
            config          TEXT NOT NULL DEFAULT '',
            models          JSONB NOT NULL DEFAULT '[]',
            metadata        JSONB NOT NULL DEFAULT '{}',
            deleted_at      TIMESTAMPTZ,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(f"""
        INSERT INTO llm_providers_new (
            id, provider_id, organization_id, name, type, source, is_shown,
            config, models, metadata, deleted_at, created_at, updated_at
        )
        SELECT
            id,
            COALESCE(data->>'provider_id', ''),
            COALESCE(data->>'organization_id', ''),
            COALESCE(data->>'name', ''),
            COALESCE(data->>'type', ''),
            COALESCE(data->>'source', 'custom'),
            {_safe_bool('is_shown')},
            COALESCE(data->>'config', ''),
            COALESCE(data->'models', '[]'::jsonb),
            COALESCE(data->'metadata', '{{}}'::jsonb),
            {_safe_ts('deleted_at')},
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM llm_providers
    """)
    op.drop_table("llm_providers")
    op.rename_table("llm_providers_new", "llm_providers")
    op.create_index("llm_providers_provider_id_idx", "llm_providers", ["provider_id"])
    op.create_index("llm_providers_org_id_idx", "llm_providers", ["organization_id"])

    # ── 8. openai_assistants ─────────────────────────────────────────────────
    op.execute("""
        CREATE TABLE openai_assistants_new (
            id                       TEXT PRIMARY KEY,
            assistant_id             TEXT NOT NULL DEFAULT '',
            organization_id          TEXT NOT NULL DEFAULT '',
            name                     TEXT NOT NULL DEFAULT '',
            model                    TEXT NOT NULL DEFAULT '',
            model_id                 TEXT,
            provider_id              TEXT,
            description              TEXT NOT NULL DEFAULT '',
            instructions             TEXT NOT NULL DEFAULT '',
            core_task                TEXT NOT NULL DEFAULT '',
            agent_type               TEXT,
            mode                     TEXT,
            category                 TEXT,
            channel                  TEXT,
            personality_role         TEXT,
            response_length          TEXT,
            tone_and_style           TEXT,
            preload_information      TEXT,
            preload_information_type TEXT,
            workflow_name            TEXT,
            workflow_function_call   TEXT,
            credential_name          TEXT,
            credential_id            TEXT,
            guardrail_id             TEXT,
            default_folder_id        TEXT,
            version                  TEXT,
            object                   TEXT,
            reasoning_effort         TEXT,
            temperature              FLOAT,
            top_p                    FLOAT,
            use_history              BOOLEAN NOT NULL DEFAULT FALSE,
            streaming                BOOLEAN NOT NULL DEFAULT FALSE,
            show_thinking_process    BOOLEAN NOT NULL DEFAULT FALSE,
            use_memory               BOOLEAN NOT NULL DEFAULT FALSE,
            file_ids                 TEXT[] NOT NULL DEFAULT '{}',
            sub_agents               TEXT[] NOT NULL DEFAULT '{}',
            knowledge_hubs           TEXT[] NOT NULL DEFAULT '{}',
            banned_words             TEXT[] NOT NULL DEFAULT '{}',
            board_ids                TEXT[] NOT NULL DEFAULT '{}',
            folder_ids               TEXT[] NOT NULL DEFAULT '{}',
            metadata                 JSONB,
            tools                    JSONB,
            tool_resources           JSONB,
            response_format          JSONB,
            deleted_at               TIMESTAMPTZ,
            created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute(f"""
        INSERT INTO openai_assistants_new (
            id, assistant_id, organization_id, name, model, model_id, provider_id,
            description, instructions, core_task, agent_type, mode, category, channel,
            personality_role, response_length, tone_and_style, preload_information,
            preload_information_type, workflow_name, workflow_function_call,
            credential_name, credential_id, guardrail_id, default_folder_id,
            version, object, reasoning_effort, temperature, top_p,
            use_history, streaming, show_thinking_process, use_memory,
            file_ids, sub_agents, knowledge_hubs, banned_words, board_ids, folder_ids,
            metadata, tools, tool_resources, response_format,
            deleted_at, created_at, updated_at
        )
        SELECT
            id,
            COALESCE(data->>'assistant_id', ''),
            COALESCE(data->>'organization_id', ''),
            COALESCE(data->>'name', ''),
            COALESCE(data->>'model', ''),
            data->>'model_id',
            data->>'provider_id',
            COALESCE(data->>'description', ''),
            COALESCE(data->>'instructions', ''),
            COALESCE(data->>'core_task', ''),
            data->>'agent_type',
            data->>'mode',
            data->>'category',
            data->>'channel',
            data->>'personality_role',
            data->>'response_length',
            data->>'tone_and_style',
            data->>'preload_information',
            data->>'preload_information_type',
            data->>'workflow_name',
            data->>'workflow_function_call',
            data->>'credential_name',
            data->>'credential_id',
            data->>'guardrail_id',
            data->>'default_folder_id',
            data->>'version',
            data->>'object',
            data->>'reasoning_effort',
            {_safe_float('temperature')},
            {_safe_float('top_p')},
            {_safe_bool('use_history')},
            {_safe_bool('streaming')},
            {_safe_bool('show_thinking_process')},
            {_safe_bool('use_memory')},
            {_safe_arr('file_ids')},
            {_safe_arr('sub_agents')},
            {_safe_arr('knowledge_hubs')},
            {_safe_arr('banned_words')},
            {_safe_arr('board_ids')},
            {_safe_arr('folder_ids')},
            data->'metadata',
            data->'tools',
            data->'tool_resources',
            data->'response_format',
            {_safe_ts('deleted_at')},
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM openai_assistants
    """)
    op.drop_table("openai_assistants")
    op.rename_table("openai_assistants_new", "openai_assistants")
    op.create_index("openai_assistants_assistant_id_idx", "openai_assistants", ["assistant_id"])
    op.create_index("openai_assistants_org_id_idx", "openai_assistants", ["organization_id"])
    op.create_index("openai_assistants_agent_type_idx", "openai_assistants", ["agent_type"])

    # ── 9. merge legacy `guardrail` → `guardrails` then drop ─────────────────
    op.execute("""
        INSERT INTO guardrails (
            id, name, model, org_id, description, instructions,
            unsafe_categories, competitor_keywords, custom_unsafe_patterns,
            guardrails_config_id, guardrail_provider_id, created_at, updated_at
        )
        SELECT
            id,
            COALESCE(data->>'name', ''),
            COALESCE(data->>'model_id', ''),
            COALESCE(data->>'organization_id', ''),
            '',
            COALESCE(data->>'instructions', ''),
            CASE WHEN data->'unsafed_keywords' IS NOT NULL AND jsonb_typeof(data->'unsafed_keywords') = 'array'
                 THEN ARRAY(SELECT jsonb_array_elements_text(data->'unsafed_keywords'))
                 ELSE '{}'::text[] END,
            CASE WHEN data->'competitor_keywords' IS NOT NULL AND jsonb_typeof(data->'competitor_keywords') = 'array'
                 THEN ARRAY(SELECT jsonb_array_elements_text(data->'competitor_keywords'))
                 ELSE '{}'::text[] END,
            CASE WHEN data->'custom_unsafe_patterns' IS NOT NULL AND jsonb_typeof(data->'custom_unsafe_patterns') = 'array'
                 THEN ARRAY(SELECT jsonb_array_elements_text(data->'custom_unsafe_patterns'))
                 ELSE '{}'::text[] END,
            NULL,
            NULL,
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM guardrail
        ON CONFLICT (id) DO NOTHING
    """)
    op.drop_table("guardrail")


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    # Re-create generic JSONB stubs (data-only, no content migration)
    for table in [
        "file_content", "board_messages", "echarts", "assistant_templates",
        "parquets", "openai_files", "llm_providers", "openai_assistants",
    ]:
        op.drop_table(table)
        op.create_table(
            table,
            sa.Column("id", sa.Text, primary_key=True),
            sa.Column("data", JSONB, nullable=False),
            sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
            sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        )
