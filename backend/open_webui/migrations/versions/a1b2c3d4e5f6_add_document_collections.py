"""Add document collections (MongoDB → PostgreSQL migration)

Revision ID: a1b2c3d4e5f6
Revises: ca81bd47c050
Create Date: 2026-03-20 00:00:00.000000

Creates JSONB tables for all active MongoDB collections, plus the
pgvector-backed `rag` table for vector search.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "3781e22d8b01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Collections that get a generic JSONB table.
# Order: anything later converted to a relational table by a JSONB→relational
# migration (d1e2f3a4b5c6, c1d2e3f4a5b6, f1a2b3c4d5e6) must appear here so the
# JSONB staging table exists when migrate_mongo_to_pg.py runs.
JSONB_COLLECTIONS = [
    "history",
    "checkpoints_aio",
    "checkpoint_writes_aio",
    "openai_assistants",
    "openai_files",
    "parquets",
    "board_embedding_job",
    "file_content",
    "llm_providers",
    "echarts",
    "board_messages",
    "migrations",
    "assistant_templates",
    "guardrail",
    "guardrails",
    # guardrail_providers added 2026-05-18: previously missing here meant
    # the JSONB staging table was created out-of-band on dev RDS and never
    # on fresh PG. f1a2b3c4d5e6 already detects the missing-source case via
    # to_regclass, so this addition is safe for both fresh and existing envs.
    "guardrail_providers",
]


def upgrade():
    # This migration is PostgreSQL-only; skip silently on SQLite (dev mode).
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # Enable pgvector extension (required for the rag table)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── JSONB document tables ────────────────────────────────────────────────
    for collection in JSONB_COLLECTIONS:
        op.create_table(
            collection,
            sa.Column("id", sa.Text, primary_key=True),
            sa.Column("data", JSONB, nullable=False),
            sa.Column(
                "updated_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
            sa.Column(
                "created_at",
                sa.TIMESTAMP(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            ),
        )
        op.create_index(
            f"{collection}_data_gin",
            collection,
            ["data"],
            postgresql_using="gin",
        )

    # ── Vector table (rag) ───────────────────────────────────────────────────
    # Uses pgvector's vector type; dimensions match the embedding model (1536).
    op.execute("""
        CREATE TABLE IF NOT EXISTS rag (
            id          TEXT PRIMARY KEY,
            text        TEXT,
            embedding   vector(1536),
            org_id      TEXT,
            assistant_id TEXT,
            file_id     TEXT,
            board_id    TEXT,
            boarditem_id TEXT,
            folder_id   TEXT,
            source      TEXT,
            blob_type   TEXT,
            line        INTEGER,
            loc         TEXT,
            original_name TEXT,
            bytes       BIGINT,
            board_ids   TEXT[],
            data        JSONB NOT NULL DEFAULT '{}',
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # ANN index for cosine / dot-product similarity search
    op.execute("""
        CREATE INDEX IF NOT EXISTS rag_embedding_idx
        ON rag USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

    # Filter indexes used in vector search queries
    op.create_index("rag_org_id_idx", "rag", ["org_id"])
    op.create_index("rag_assistant_id_idx", "rag", ["assistant_id"])
    op.create_index("rag_file_id_idx", "rag", ["file_id"])
    op.create_index("rag_board_id_idx", "rag", ["board_id"])

    # Full-text search index on rag.text
    op.execute("""
        CREATE INDEX IF NOT EXISTS rag_text_fts_idx
        ON rag USING gin(to_tsvector('english', coalesce(text, '')))
    """)


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_table("rag")
    for collection in reversed(JSONB_COLLECTIONS):
        op.drop_table(collection)
    op.execute("DROP EXTENSION IF EXISTS vector")
