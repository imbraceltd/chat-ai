"""Idempotent boot-time Postgres schema guard.

`alembic upgrade head` (run at import time in config.run_migrations) only
applies PENDING revisions. A DB that is already stamped at head but is missing
a column is therefore never healed — alembic considers it up to date and does
nothing.

This is exactly how `openai_assistants.document_ai` goes missing: it is added
by b2c3d4e5f6a7, but the d1e2f3a4b5c6 "proper_schemas_batch" rebuild recreates
the table WITHOUT it. Because b2c3 and the d1e2 branch are siblings off
a1b2c3d4e5f6, alembic's branch order is non-deterministic on a blank DB; when
d1e2 runs after b2c3 the freshly-added column is dropped and never comes back.
Such a DB ends up at head e2f3a4b5c6d7 with no document_ai, so the startup RAG
migration fails with `column "document_ai" does not exist` and assistant
creation 400s.

The d4f5a6b7c8e9 merge migration fixes future blank DBs, but cannot help DBs
that already migrated past it. This guard runs unconditionally on every boot
and re-asserts known-critical columns with `ADD COLUMN IF NOT EXISTS` — a no-op
when they already exist — so fresh, drifted, and already-broken DBs all converge
to the correct schema. Additive only: it never drops or retypes anything.
"""

import logging

from sqlalchemy import text

from open_webui.internal.db import engine
from open_webui.internal.document_store import DB_TYPE

log = logging.getLogger(__name__)

# (table, column, column_type) — additive, idempotent assertions only.
# Add an entry here whenever a column is known to drift on already-migrated DBs.
_REQUIRED_COLUMNS = [
    ("openai_assistants", "document_ai", "JSONB"),
]


def ensure_pg_schema() -> None:
    """Re-assert critical columns exist. Postgres-only; safe to run every boot."""
    if DB_TYPE != "postgresql":
        return

    ensured = []
    try:
        with engine.begin() as conn:
            for table, column, column_type in _REQUIRED_COLUMNS:
                # Guard on table existence so a not-yet-created table is skipped
                # rather than raising; the column clause is itself idempotent.
                conn.execute(
                    text(
                        f"""
                        DO $$
                        BEGIN
                            IF to_regclass('{table}') IS NOT NULL THEN
                                ALTER TABLE {table}
                                    ADD COLUMN IF NOT EXISTS {column} {column_type};
                            END IF;
                        END $$;
                        """
                    )
                )
                ensured.append(f"{table}.{column}")
        log.info(f"✅ Postgres schema guard: ensured {', '.join(ensured)}")
    except Exception as e:
        log.error(f"❌ Postgres schema guard failed: {e}")
