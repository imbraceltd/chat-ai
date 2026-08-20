"""Convert banned_words column from TEXT[] to TEXT

Revision ID: a2b3c4d5e6f7
Revises: d1e2f3a4b5c6
Create Date: 2026-04-06 00:00:00.000000

Converts openai_assistants.banned_words from TEXT[] to plain TEXT.
Existing array values are joined with ', ' to preserve data.
"""

from typing import Sequence, Union
from alembic import op

revision: str = "a2b3c4d5e6f7"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("""
        ALTER TABLE openai_assistants
        ALTER COLUMN banned_words TYPE TEXT
        USING array_to_string(banned_words, ', ')
    """)

    op.execute("""
        ALTER TABLE openai_assistants
        ALTER COLUMN banned_words SET DEFAULT ''
    """)

    op.execute("""
        UPDATE openai_assistants
        SET banned_words = ''
        WHERE banned_words IS NULL
    """)

    op.execute("""
        ALTER TABLE openai_assistants
        ALTER COLUMN banned_words SET NOT NULL
    """)


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("""
        ALTER TABLE openai_assistants
        ALTER COLUMN banned_words TYPE TEXT[]
        USING CASE
            WHEN banned_words IS NULL OR TRIM(banned_words) = '' THEN '{}'::TEXT[]
            ELSE ARRAY(SELECT TRIM(e) FROM unnest(string_to_array(banned_words, ',')) AS e WHERE TRIM(e) <> '')
        END
    """)

    op.execute("""
        ALTER TABLE openai_assistants
        ALTER COLUMN banned_words SET DEFAULT '{}'
    """)
