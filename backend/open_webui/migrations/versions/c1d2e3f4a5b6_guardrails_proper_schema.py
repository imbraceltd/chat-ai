"""Replace JSONB guardrails table with proper relational schema

Revision ID: c1d2e3f4a5b6
Revises: a1b2c3d4e5f6
Create Date: 2026-03-23 00:00:00.000000

Drops the generic JSONB `guardrails` table and replaces it with a
properly-typed relational table, migrating any existing data.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, TEXT

revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # 1. Create new properly-typed table
    op.create_table(
        "guardrails_new",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False, server_default=""),
        sa.Column("model", sa.Text, nullable=False, server_default=""),
        sa.Column("org_id", sa.Text, nullable=False, server_default=""),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("instructions", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "unsafe_categories",
            ARRAY(TEXT),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "competitor_keywords",
            ARRAY(TEXT),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "custom_unsafe_patterns",
            ARRAY(TEXT),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("guardrails_config_id", sa.Text, nullable=True),
        sa.Column("guardrail_provider_id", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    # 2. Migrate existing data from JSONB table (if it exists)
    op.execute("""
        INSERT INTO guardrails_new (
            id, name, model, org_id, description, instructions,
            unsafe_categories, competitor_keywords, custom_unsafe_patterns,
            guardrails_config_id, guardrail_provider_id,
            created_at, updated_at
        )
        SELECT
            id,
            COALESCE(data->>'name', ''),
            COALESCE(data->>'model', ''),
            COALESCE(data->>'org_id', ''),
            COALESCE(data->>'description', ''),
            COALESCE(data->>'instructions', ''),
            COALESCE(
                ARRAY(SELECT jsonb_array_elements_text(data->'unsafe_categories')),
                '{}'::text[]
            ),
            COALESCE(
                ARRAY(SELECT jsonb_array_elements_text(data->'competitor_keywords')),
                '{}'::text[]
            ),
            COALESCE(
                ARRAY(SELECT jsonb_array_elements_text(data->'custom_unsafe_patterns')),
                '{}'::text[]
            ),
            data->>'guardrails_config_id',
            data->>'guardrail_provider_id',
            COALESCE(created_at, NOW()),
            COALESCE(updated_at, NOW())
        FROM guardrails
    """)

    # 3. Drop old JSONB table and rename new one
    op.drop_table("guardrails")
    op.rename_table("guardrails_new", "guardrails")

    # 4. Indexes
    op.create_index("guardrails_org_id_idx", "guardrails", ["org_id"])
    op.create_index(
        "guardrails_config_id_idx", "guardrails", ["guardrails_config_id"]
    )


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_table("guardrails")

    # Re-create the generic JSONB table
    from sqlalchemy.dialects.postgresql import JSONB

    op.create_table(
        "guardrails",
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
        "guardrails_data_gin", "guardrails", ["data"], postgresql_using="gin"
    )
