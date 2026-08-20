"""Add models_metadata column to guardrail_providers

Revision ID: e2f3a4b5c6d7
Revises: d4f5a6b7c8e9
Create Date: 2026-05-21 00:00:00.000000

The relational `guardrail_providers` table (introduced in f1a2b3c4d5e6)
never carried the per-model metadata that the API layer reads/writes
(`models_metadata`, e.g. custom icon image_url keyed by model name).
In the old JSONB form it lived inside the `data` blob; the relational
rewrite dropped it. Add it back as a dedicated JSONB column.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "e2f3a4b5c6d7"
down_revision: Union[str, None] = "d4f5a6b7c8e9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.add_column(
        "guardrail_providers",
        sa.Column(
            "models_metadata",
            JSONB,
            nullable=False,
            server_default="{}",
        ),
    )


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.drop_column("guardrail_providers", "models_metadata")
