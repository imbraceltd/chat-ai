#!/usr/bin/env bash
# migrate_all.sh — one-shot Mongo → PostgreSQL migration for chat-ai.
#
# Splits alembic execution around the bulk Mongo copy so JSONB→relational
# converter migrations (c1d2e3f4a5b6, d1e2f3a4b5c6, f1a2b3c4d5e6) run
# AFTER the JSONB staging tables are populated.
#
# Required env:
#   MONGO_URL     mongodb+srv://… (source)
#   MONGO_DB      e.g. imbrace_stg
#   DATABASE_URL  postgresql://… (target; must be reachable & empty/upgradable)
#
# Optional:
#   BATCH_SIZE      default 2000 (JSONB); RAG always uses 500
#   COLLECTIONS     all|jsonb|rag (default all)
#   SKIP_SMOKE      set to skip post-migration row-count checks

set -euo pipefail

# ── Pre-flight ────────────────────────────────────────────────────────────────
say() { printf "\n[%s] %s\n" "$(date +%H:%M:%S)" "$*"; }
fail() { printf "\n[FATAL] %s\n" "$*" >&2; exit 1; }

[[ -n "${MONGO_URL:-}"    ]] || fail "MONGO_URL is required"
[[ -n "${MONGO_DB:-}"     ]] || fail "MONGO_DB is required"
[[ -n "${DATABASE_URL:-}" ]] || fail "DATABASE_URL is required"

command -v python  >/dev/null || fail "python not on PATH"
command -v alembic >/dev/null || fail "alembic not on PATH"

# Verify Mongo reachable
say "Probing source Mongo ($MONGO_DB)"
python - <<'PY'
import os, sys
from pymongo import MongoClient
try:
    c = MongoClient(os.environ["MONGO_URL"], serverSelectionTimeoutMS=10000)
    c.admin.command("ping")
    db = c[os.environ["MONGO_DB"]]
    print(f"Mongo ok — {len(db.list_collection_names())} collections in {db.name}")
except Exception as e:
    sys.stderr.write(f"Mongo unreachable: {e}\n")
    sys.exit(1)
PY

# Verify Postgres reachable
say "Probing target Postgres"
python - <<'PY'
import asyncio, os, sys
import asyncpg
async def go():
    try:
        conn = await asyncpg.connect(os.environ["DATABASE_URL"], timeout=10)
        v = await conn.fetchval("SELECT version()")
        print(f"PG ok — {v.split(',')[0]}")
        await conn.close()
    except Exception as e:
        sys.stderr.write(f"PG unreachable: {e}\n"); sys.exit(1)
asyncio.run(go())
PY

# ── Phase 1: alembic up to (and including) the JSONB-staging migration ────────
# a1b2c3d4e5f6 creates every JSONB document table including guardrail_providers.
# We stop here so the bulk copy below has tables to write into.
say "Phase 1/3: alembic upgrade a1b2c3d4e5f6 (JSONB staging tables)"
alembic -c open_webui/alembic.ini upgrade a1b2c3d4e5f6

# ── Phase 2: bulk-copy Mongo → JSONB staging tables ───────────────────────────
say "Phase 2/3: migrate_mongo_to_pg.py (bulk Mongo → JSONB)"
COLLECTIONS="${COLLECTIONS:-all}"
BATCH_SIZE="${BATCH_SIZE:-2000}"
python migrate_mongo_to_pg.py --collections "$COLLECTIONS" --batch-size "$BATCH_SIZE"

# ── Phase 3: remaining alembic migrations (JSONB → typed relational) ──────────
# c1d2e3f4a5b6 (guardrails), d1e2f3a4b5c6 (proper_schemas_batch — 9 tables),
# f1a2b3c4d5e6 (guardrail_providers), plus any trailing schema changes.
say "Phase 3/3: alembic upgrade head (JSONB → relational converters)"
alembic -c open_webui/alembic.ini upgrade head

# ── Smoke checks ──────────────────────────────────────────────────────────────
if [[ -z "${SKIP_SMOKE:-}" ]]; then
  say "Smoke: row-count parity per converted table"
  python - <<'PY'
import asyncio, os
import asyncpg
from pymongo import MongoClient

TABLES = [
    # (mongo_collection, pg_table_after_conversion)
    ("history", "history"),
    ("openai_assistants", "openai_assistants"),
    ("guardrails", "guardrails"),
    ("guardrail_providers", "guardrail_providers"),
    ("llm_providers", "llm_providers"),
    ("file_content", "file_content"),
]

async def go():
    mongo = MongoClient(os.environ["MONGO_URL"], serverSelectionTimeoutMS=10000)
    db = mongo[os.environ["MONGO_DB"]]
    pg = await asyncpg.connect(os.environ["DATABASE_URL"])
    print(f"{'collection':<28}{'mongo':>10}{'pg':>10}{'diff':>10}")
    print("─" * 58)
    bad = 0
    for col, table in TABLES:
        m = db[col].estimated_document_count()
        try:
            p = await pg.fetchval(f'SELECT count(*) FROM {table}')
        except Exception as e:
            print(f"{col:<28}{m:>10}{'ERR':>10}  {e}")
            bad += 1
            continue
        diff = m - p
        marker = " ⚠" if diff != 0 else ""
        print(f"{col:<28}{m:>10}{p:>10}{diff:>10}{marker}")
        if diff != 0:
            bad += 1
    await pg.close()
    if bad:
        print(f"\n{bad} table(s) failed parity check — investigate before declaring done")
asyncio.run(go())
PY
fi

say "✓ migration complete"
