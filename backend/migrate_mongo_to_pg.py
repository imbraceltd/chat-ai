#!/usr/bin/env python3
"""Data migration: MongoDB (imbrace_dev) → PostgreSQL (RDS chatai).

Uses asyncpg copy_records_to_table for fast bulk inserts (~10k docs/sec).

Usage:
    python migrate_mongo_to_pg.py [--collections all|rag|jsonb] [--batch-size 2000]
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from bson import ObjectId
from pymongo import MongoClient

# ── Config ───────────────────────────────────────────────────────────────────
MONGO_URL = os.getenv("MONGO_URL")
MONGO_DB = os.getenv("MONGO_DB")
DATABASE_URL = os.getenv("DATABASE_URL")
if not MONGO_URL or not MONGO_DB or not DATABASE_URL:
    sys.stderr.write(
        "MONGO_URL, MONGO_DB and DATABASE_URL must be set in env. "
        "Refusing to run with implicit defaults.\n"
    )
    sys.exit(1)
DEFAULT_BATCH_SIZE = 2000
RAG_BATCH_SIZE = 500

JSONB_COLLECTIONS = [
    # history / checkpoints already migrated — leave in list so re-runs are idempotent
    ("history",               "history"),
    ("checkpoints_aio",       "checkpoints_aio"),
    ("checkpoint_writes_aio", "checkpoint_writes_aio"),
    ("openai_assistants",     "openai_assistants"),
    ("openai_files",          "openai_files"),
    ("parquets",              "parquets"),
    ("boardEmbeddingJob",     "board_embedding_job"),
    ("file_content",          "file_content"),
    ("llm_providers",         "llm_providers"),
    ("echarts",               "echarts"),
    ("board_messages",        "board_messages"),
    ("migrations",            "migrations"),
    ("assistant_templates",   "assistant_templates"),
    ("guardrail",             "guardrail"),
    ("guardrails",            "guardrails"),
    # guardrail_providers: source collection in Mongo, JSONB staging table
    # added in a1b2c3d4e5f6 (2026-05-18). f1a2b3c4d5e6 then converts
    # JSONB → typed relational. Both pre-staging-pop and post-pop runs
    # of `alembic upgrade head` are safe because f1a2b3c4d5e6 uses
    # to_regclass to detect whether the JSONB form is present.
    ("guardrail_providers",   "guardrail_providers"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("migration.log")],
)
log = logging.getLogger(__name__)


# ── Serialisation ─────────────────────────────────────────────────────────────
def _serialize(obj: Any) -> Any:
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace").replace("\x00", "")
    if isinstance(obj, str):
        # PostgreSQL rejects \u0000 (null bytes) in JSONB
        return obj.replace("\x00", "")
    if isinstance(obj, float):
        # JSON / PostgreSQL JSONB don't support NaN or Infinity
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def _doc_id(doc: Dict) -> Optional[str]:
    v = doc.get("_id")
    return str(v) if v is not None else None


def _parse_ts(value) -> datetime:
    now = datetime.now(timezone.utc)
    if value is None:
        return now
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return now
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return now
    return now


# ── JSONB bulk migration ──────────────────────────────────────────────────────
def _strip_nullbytes(s: str) -> str:
    """Remove null bytes which PostgreSQL rejects in text/JSONB."""
    return s.replace("\x00", "")


def _doc_to_jsonb_row(doc: Dict) -> Optional[Tuple]:
    """Convert a MongoDB doc to a (id, data_json, updated_at, created_at) tuple."""
    doc_id = _doc_id(doc)
    if not doc_id:
        return None
    serialized = _serialize(doc)
    if "_id" not in serialized:
        serialized["_id"] = doc_id
    return (
        doc_id,
        _strip_nullbytes(json.dumps(serialized)),
        _parse_ts(doc.get("updated_at")),
        _parse_ts(doc.get("created_at")),
    )


async def migrate_jsonb(mongo_db, pg_pool, mongo_col, pg_table, batch_size):
    collection = mongo_db[mongo_col]
    total = collection.count_documents({})
    log.info(f"[{pg_table}] {total:,} docs")
    if total == 0:
        return 0

    migrated = 0
    skipped = 0
    batch: List[Tuple] = []
    t0 = time.time()

    upsert_sql = f"""
        INSERT INTO {pg_table} (id, data, updated_at, created_at)
        VALUES ($1, $2::jsonb, $3, $4)
        ON CONFLICT (id) DO UPDATE
            SET data = EXCLUDED.data,
                updated_at = EXCLUDED.updated_at
    """

    async def flush():
        nonlocal migrated
        if not batch:
            return
        async with pg_pool.acquire() as conn:
            await conn.executemany(upsert_sql, batch)
        migrated += len(batch)
        batch.clear()

    for doc in collection.find({}).batch_size(batch_size):
        row = _doc_to_jsonb_row(doc)
        if row is None:
            skipped += 1
            continue
        batch.append(row)
        if len(batch) >= batch_size:
            await flush()
            elapsed = time.time() - t0
            rate = migrated / elapsed if elapsed > 0 else 1
            log.info(f"[{pg_table}]  {migrated:,}/{total:,} ({rate:.0f} docs/s)")

    await flush()
    elapsed = time.time() - t0
    log.info(f"[{pg_table}] ✓ {migrated:,} migrated, {skipped} skipped in {elapsed:.1f}s")
    return migrated


# ── RAG vector bulk migration ─────────────────────────────────────────────────
def _doc_to_rag_row(doc: Dict) -> Optional[Tuple]:
    doc_id = _doc_id(doc)
    embedding = doc.get("embedding")
    if not doc_id or not embedding or not isinstance(embedding, list):
        return None
    if len(embedding) != 1536:
        return None  # skip wrong-dimension embeddings (old model — needs re-ingest)

    data_doc = _serialize({k: v for k, v in doc.items() if k != "embedding"})
    if "_id" not in data_doc:
        data_doc["_id"] = doc_id

    def _s(v) -> str:
        return (v or "").replace("\x00", "")

    embedding_str = "[" + ",".join(str(float(v)) for v in embedding) + "]"
    loc = doc.get("loc")

    return (
        doc_id,                                      # id
        _s(doc.get("text", "")),                     # text
        embedding_str,                               # embedding (cast to vector in SQL)
        _s(doc.get("org_id", "")),                   # org_id
        _s(doc.get("assistant_id", "")),             # assistant_id
        _s(doc.get("file_id", "")),                  # file_id
        _s(doc.get("board_id", "")),                 # board_id
        _s(doc.get("boarditem_id", "")),             # boarditem_id
        _s(doc.get("folder_id", "")),                # folder_id
        _s(doc.get("source", "")),                   # source
        _s(doc.get("blobType", "")),                 # blob_type
        doc.get("line"),                             # line
        _strip_nullbytes(json.dumps(loc)) if loc else None,  # loc
        _s(doc.get("original_name", "")),            # original_name
        doc.get("bytes"),                            # bytes
        doc.get("board_ids") or [],                  # board_ids
        _strip_nullbytes(json.dumps(data_doc)),      # data
        _parse_ts(doc.get("created_at")),            # updated_at / created_at
    )


async def migrate_rag(mongo_db, pg_pool, batch_size):
    collection = mongo_db["rag"]
    # estimated_document_count is instant (uses collection metadata)
    total = collection.estimated_document_count()
    log.info(f"[rag] ~{total:,} vector docs")

    migrated = 0
    skipped = 0
    batch: List[Tuple] = []
    t0 = time.time()

    # Prepare the upsert SQL (can't use copy_records for vector type directly)
    upsert_sql = """
        INSERT INTO rag (
            id, text, embedding, org_id, assistant_id, file_id,
            board_id, boarditem_id, folder_id, source, blob_type,
            line, loc, original_name, bytes, board_ids,
            data, updated_at, created_at
        ) VALUES (
            $1, $2, $3::vector, $4, $5, $6,
            $7, $8, $9, $10, $11,
            $12, $13, $14, $15, $16,
            $17::jsonb, $18, $18
        )
        ON CONFLICT (id) DO UPDATE
            SET embedding  = EXCLUDED.embedding,
                data       = EXCLUDED.data,
                updated_at = EXCLUDED.updated_at
    """

    async def flush_rag():
        nonlocal migrated, skipped
        if not batch:
            return
        async with pg_pool.acquire() as conn:
            await conn.executemany(upsert_sql, batch)
        migrated += len(batch)
        batch.clear()

    for doc in collection.find({}).batch_size(batch_size):
        row = _doc_to_rag_row(doc)
        if row is None:
            skipped += 1
            continue
        batch.append(row)
        if len(batch) >= batch_size:
            await flush_rag()
            elapsed = time.time() - t0
            rate = migrated / elapsed if elapsed > 0 else 1
            eta = (total - migrated) / rate if rate > 0 else 0
            log.info(
                f"[rag]  {migrated:,}/{total:,} "
                f"({rate:.0f} docs/s, ETA {eta/60:.1f} min)"
            )

    await flush_rag()
    elapsed = time.time() - t0
    log.info(f"[rag] ✓ {migrated:,} migrated, {skipped} skipped in {elapsed/60:.1f} min")
    return migrated


# ── Main ──────────────────────────────────────────────────────────────────────
async def main(args):
    log.info("=== MongoDB → PostgreSQL data migration ===")
    log.info(f"Mongo: {MONGO_DB}")
    log.info(f"PG:    {DATABASE_URL.split('@')[-1]}")

    mongo_client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=15000)
    mongo_db = mongo_client[MONGO_DB]
    mongo_client.admin.command("ping")
    log.info("MongoDB ✓")

    pg_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    log.info("PostgreSQL ✓\n")

    total = 0
    t_start = time.time()

    try:
        if args.collections in ("all", "jsonb"):
            log.info("── JSONB collections ─────────────────────────────────────")
            for mongo_col, pg_table in JSONB_COLLECTIONS:
                total += await migrate_jsonb(
                    mongo_db, pg_pool, mongo_col, pg_table, args.batch_size
                )

        if args.collections in ("all", "rag"):
            log.info("\n── RAG vectors ───────────────────────────────────────────")
            total += await migrate_rag(mongo_db, pg_pool, RAG_BATCH_SIZE)

    finally:
        mongo_client.close()
        await pg_pool.close()

    elapsed = time.time() - t_start
    log.info(f"\n=== Done: {total:,} docs in {elapsed/60:.1f} min ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collections", choices=["all", "jsonb", "rag"], default="all")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    asyncio.run(main(parser.parse_args()))
