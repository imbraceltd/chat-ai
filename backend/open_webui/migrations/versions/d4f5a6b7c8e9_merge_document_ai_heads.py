"""Merge migration heads (document_ai + banned_words + guardrail_providers)

Revision ID: d4f5a6b7c8e9
Revises: a2b3c4d5e6f7, f1a2b3c4d5e6, b2c3d4e5f6a7
Create Date: 2026-05-20 00:00:00.000000

The migration history had three divergent heads, which made
`alembic upgrade head` ambiguous and silently fail at startup, so no
pending migration (including document_ai) was ever applied. This merge
node unifies them into a single head.

It also re-asserts the `document_ai` column on openai_assistants. The
column is added by b2c3d4e5f6a7, but the d1e2f3a4b5c6 "proper_schemas_batch"
rebuild drops and recreates openai_assistants WITHOUT document_ai. Because
b2c3 and the d1e2 branch are siblings off a1b2c3d4e5f6, alembic's branch
traversal order is non-deterministic on a blank DB: when d1e2 runs after
b2c3, the rebuild silently drops the freshly-added column and it never
comes back (b2c3 won't re-run). The result is a fresh deployment whose
openai_assistants has no document_ai, so the startup RAG migration fails
with `column "document_ai" does not exist` and assistant creation 400s.

This merge is the deterministic join of both branches — it runs only after
b2c3 AND d1e2 have both applied, in either order — so an idempotent
ADD COLUMN here guarantees document_ai survives to the unified head.
"""

from alembic import op

from typing import Sequence, Union

revision: str = "d4f5a6b7c8e9"
down_revision: Union[str, Sequence[str], None] = (
    "a2b3c4d5e6f7",
    "f1a2b3c4d5e6",
    "b2c3d4e5f6a7",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Re-assert document_ai regardless of the branch order alembic picked for
    # b2c3 (adds it) vs d1e2 (rebuilds the table without it). Idempotent: a
    # no-op when the column already survived. See module docstring.
    op.execute(
        "ALTER TABLE openai_assistants ADD COLUMN IF NOT EXISTS document_ai JSONB"
    )


def downgrade():
    # document_ai is owned by b2c3d4e5f6a7; don't drop it when undoing the merge.
    pass
