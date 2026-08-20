"""Add rag.embedding_v2 column for OpenAI → Bedrock embedding migration

Revision ID: f7a8b9c0d1e2
Revises: e2f3a4b5c6d7
Create Date: 2026-06-23 00:00:00.000000

Blue-green migration of RAG embeddings from OpenAI (text-embedding-3-small)
to AWS Bedrock (amazon.titan-embed-text-v1). Both are 1536-dim, but the two
vector spaces are incompatible, so existing rows must be re-embedded from their
stored `text` column into this new `embedding_v2` column.

This migration only adds the column so that every environment gets it on
deploy. The Bedrock vectors are backfilled out-of-band by
backend/backfill_embedding_v2.py, and the ANN index on the new column is
created after the backfill completes (an IVFFlat index trains its centroids on
existing data, so building it on an empty/partial column is wasteful).
"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, None] = "e2f3a4b5c6d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # PostgreSQL-only (pgvector). Skip silently elsewhere (e.g. SQLite dev mode),
    # matching the guard used by the migration that created the rag table.
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # Same dimension as the existing `embedding` column (Titan v1 = 1536).
    # Idempotent so it is safe on environments where the column was already
    # added out-of-band by backfill_embedding_v2.py --ensure-column-only.
    op.execute("ALTER TABLE rag ADD COLUMN IF NOT EXISTS embedding_v2 vector(1536)")


def downgrade():
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    op.execute("DROP INDEX IF EXISTS rag_embedding_v2_idx")
    op.execute("ALTER TABLE rag DROP COLUMN IF EXISTS embedding_v2")
