"""Fix openai_assistants.category column type from TEXT to TEXT[]

Revision ID: e1f2a3b4c5d6
Revises: d1e2f3a4b5c6
Create Date: 2026-03-24 00:00:00.000000

The category column was created as TEXT, causing JSON array values like "[]"
to be stored and returned as plain strings. This migration converts it to TEXT[].
"""

from typing import Sequence, Union

from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # Can't use subquery in ALTER COLUMN USING, so use a temp column approach
    op.execute("ALTER TABLE openai_assistants ADD COLUMN category_new TEXT[] NOT NULL DEFAULT '{}'")
    op.execute("""
        UPDATE openai_assistants
        SET category_new = (
            SELECT COALESCE(
                ARRAY(SELECT jsonb_array_elements_text(
                    CASE
                        WHEN category IS NULL OR category = '' THEN '[]'::jsonb
                        WHEN category ~ '^\\[' THEN category::jsonb
                        ELSE '[]'::jsonb
                    END
                )),
                '{}'::TEXT[]
            )
        )
    """)
    op.execute("ALTER TABLE openai_assistants DROP COLUMN category")
    op.execute("ALTER TABLE openai_assistants RENAME COLUMN category_new TO category")


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("ALTER TABLE openai_assistants ADD COLUMN category_old TEXT")
    op.execute("""
        UPDATE openai_assistants
        SET category_old = array_to_json(category)::TEXT
    """)
    op.execute("ALTER TABLE openai_assistants DROP COLUMN category")
    op.execute("ALTER TABLE openai_assistants RENAME COLUMN category_old TO category")
