"""Add document_ai JSONB column to openai_assistants

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-05-20 00:00:00.000000

The assistant create/update repository carries a `document_ai` config object
(used by agents with agent_type='document_ai'), but the openai_assistants
table had no column for it, so the value was silently dropped at the SQL
boundary. This adds the missing JSONB column.

NOTE: this migration previously hung off 3781e22d8b01 — the *same* parent as
a1b2c3d4e5f6, which is the migration that CREATEs the openai_assistants JSONB
staging table. As siblings, alembic could run this ALTER before that CREATE,
so a fresh/blank Postgres DB crashed with `relation "openai_assistants" does
not exist` (rolling back the whole upgrade). Reparenting onto a1b2c3d4e5f6
guarantees the table exists first; this stays a merge tip of d4f5a6b7c8e9, and
the final head (e2f3a4b5c6d7) is unchanged.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(
        "ALTER TABLE openai_assistants ADD COLUMN IF NOT EXISTS document_ai JSONB"
    )


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("ALTER TABLE openai_assistants DROP COLUMN IF EXISTS document_ai")
