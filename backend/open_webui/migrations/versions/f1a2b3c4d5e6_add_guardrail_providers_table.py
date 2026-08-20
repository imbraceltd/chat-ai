"""Migrate guardrail_providers from JSONB to proper relational schema

Revision ID: f1a2b3c4d5e6
Revises: e1f2a3b4c5d6
Create Date: 2026-03-26 00:00:00.000000

Replaces the generic JSONB guardrail_providers table with a typed relational table.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, None] = "e1f2a3b4c5d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # The earlier `a1b2c3d4e5f6_add_document_collections` migration creates
    # JSONB tables for the Mongo→PG document collections, but `guardrail_providers`
    # was never added to that list — on dev RDS the JSONB form was created
    # out-of-band. On a fresh PG (sandbox/staging cutover, local rehearsal) the
    # JSONB table never exists, so the historical conversion logic below would
    # fail with "relation guardrail_providers does not exist". Detect both
    # cases and bootstrap the relational table either way.
    legacy_jsonb_exists = bind.execute(sa.text(
        "SELECT to_regclass('public.guardrail_providers') IS NOT NULL"
    )).scalar()

    # 1. Create the typed relational table.
    op.execute("""
        CREATE TABLE guardrail_providers_new (
            id                    TEXT PRIMARY KEY,
            guardrail_provider_id TEXT NOT NULL UNIQUE,
            organization_id       TEXT NOT NULL DEFAULT '',
            name                  TEXT NOT NULL DEFAULT '',
            type                  TEXT NOT NULL DEFAULT 'custom',
            source                TEXT NOT NULL DEFAULT 'custom',
            is_shown              BOOLEAN NOT NULL DEFAULT TRUE,
            config                TEXT NOT NULL DEFAULT '',
            deleted_at            TIMESTAMPTZ,
            created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)

    # 2. Copy data from the JSONB form if it existed.
    if legacy_jsonb_exists:
        op.execute("""
            INSERT INTO guardrail_providers_new (
                id, guardrail_provider_id, organization_id, name, type,
                source, is_shown, config, deleted_at, created_at, updated_at
            )
            SELECT
                id,
                COALESCE(data->>'guardrail_provider_id', id),
                COALESCE(data->>'organization_id', ''),
                COALESCE(data->>'name', ''),
                COALESCE(data->>'type', 'custom'),
                COALESCE(data->>'source', 'custom'),
                CASE WHEN data->>'is_shown' = 'false' THEN false ELSE true END,
                COALESCE(data->>'config', ''),
                CASE WHEN data->>'deleted_at' IS NOT NULL AND data->>'deleted_at' NOT IN ('null', '')
                     THEN (data->>'deleted_at')::timestamptz
                     ELSE NULL END,
                COALESCE(created_at, NOW()),
                COALESCE(updated_at, NOW())
            FROM guardrail_providers
        """)
        op.drop_table("guardrail_providers")

    # 3. Rename the new table into place.
    op.rename_table("guardrail_providers_new", "guardrail_providers")

    # 4. Indexes.
    op.execute("""
        CREATE INDEX guardrail_providers_org_id_idx
        ON guardrail_providers (organization_id)
    """)
    op.execute("""
        CREATE INDEX guardrail_providers_provider_id_idx
        ON guardrail_providers (guardrail_provider_id)
    """)


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_table("guardrail_providers")
    op.execute("""
        CREATE TABLE guardrail_providers (
            id         TEXT PRIMARY KEY,
            data       JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE INDEX guardrail_providers_data_gin
        ON guardrail_providers USING gin (data)
    """)
