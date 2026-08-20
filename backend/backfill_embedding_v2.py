#!/usr/bin/env python3
"""Backfill rag.embedding_v2 with AWS Bedrock embeddings (Titan v1).

Blue-green migration of RAG embeddings from OpenAI (text-embedding-3-small)
to AWS Bedrock (amazon.titan-embed-text-v1). Both are 1536-dim, but the two
vector spaces are incompatible, so every existing row must be re-embedded from
its stored `text` column into the new `embedding_v2` column.

The script is idempotent and resumable: it only processes rows where
`embedding_v2 IS NULL`, so it can be stopped and restarted freely, and live
ingest writing new rows (with RAG_EMBEDDING_WRITE_COLUMN=embedding_v2) won't be
clobbered.

Usage (inside the ai-service container):
    DATABASE_URL=...                       \\
    BEDROCK_REGION=us-west-2               \\
    BEDROCK_ACCESS_KEY_ID=...             \\
    BEDROCK_SECRET_ACCESS_KEY=...         \\
    python backfill_embedding_v2.py --batch-size 200 --concurrency 8

    # create the column first (idempotent) and exit:
    python backfill_embedding_v2.py --ensure-column-only
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import List, Optional, Tuple

import asyncpg
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

# ── Config ───────────────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-west-2")
BEDROCK_ACCESS_KEY_ID = os.getenv("BEDROCK_ACCESS_KEY_ID", "")
BEDROCK_SECRET_ACCESS_KEY = os.getenv("BEDROCK_SECRET_ACCESS_KEY", "")
BEDROCK_EMBEDDING_MODEL = os.getenv(
    "RAG_BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v1"
)
EXPECTED_DIM = 1536

if not DATABASE_URL:
    sys.stderr.write("DATABASE_URL must be set.\n")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("backfill_embedding_v2")


def _make_bedrock_client():
    # Adaptive retries handle Bedrock throttling with exponential backoff.
    cfg = BotoConfig(retries={"max_attempts": 8, "mode": "adaptive"})
    kwargs = {"region_name": BEDROCK_REGION, "config": cfg}
    if BEDROCK_ACCESS_KEY_ID and BEDROCK_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = BEDROCK_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = BEDROCK_SECRET_ACCESS_KEY
    return boto3.client("bedrock-runtime", **kwargs)


_bedrock = _make_bedrock_client()


def _embed_one(text: str) -> List[float]:
    """Embed a single text with Titan v1 via invoke_model (sync, boto3).

    Titan v1 caps input at 8192 tokens. We can't count tokens locally without
    the exact tokenizer, and the char/token ratio varies wildly by language
    (CJK is far denser than English), so a fixed char cap is unreliable. Start
    from a generous prefix and shrink on the ValidationException Bedrock raises
    for oversized input, until it fits. Oversized rows (rare, pre-chunking) thus
    get embedded from a truncated prefix rather than being skipped forever.
    """
    candidate = text[:40000]
    while True:
        body = json.dumps({"inputText": candidate})
        try:
            resp = _bedrock.invoke_model(modelId=BEDROCK_EMBEDDING_MODEL, body=body)
        except ClientError as e:
            if (
                "Too many input tokens" in str(e)
                and len(candidate) > 500
            ):
                # Shrink to ~60% and retry; converges in a few iterations.
                candidate = candidate[: int(len(candidate) * 0.6)]
                continue
            raise
        payload = json.loads(resp["body"].read())
        vec = payload["embedding"]
        if len(vec) != EXPECTED_DIM:
            raise ValueError(
                f"Unexpected embedding dim {len(vec)} (expected {EXPECTED_DIM})"
            )
        return vec


def _vec_to_pg(vec: List[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in vec) + "]"


async def ensure_column(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE rag ADD COLUMN IF NOT EXISTS embedding_v2 vector(1536)"
        )
    log.info("Ensured rag.embedding_v2 column exists.")


async def remaining_count(pool: asyncpg.Pool) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT count(*) FROM rag "
            "WHERE embedding_v2 IS NULL AND text IS NOT NULL AND text <> ''"
        )


async def fetch_batch(pool: asyncpg.Pool, limit: int) -> List[asyncpg.Record]:
    async with pool.acquire() as conn:
        return await conn.fetch(
            "SELECT id, text FROM rag "
            "WHERE embedding_v2 IS NULL AND text IS NOT NULL AND text <> '' "
            "ORDER BY created_at "
            "LIMIT $1",
            limit,
        )


async def embed_batch(
    rows: List[asyncpg.Record], concurrency: int
) -> List[Tuple[str, str]]:
    """Embed a batch concurrently in a thread pool. Returns (id, pgvector_str)."""
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()

    async def _one(row) -> Optional[Tuple[str, str]]:
        async with sem:
            try:
                vec = await loop.run_in_executor(None, _embed_one, row["text"])
                return (row["id"], _vec_to_pg(vec))
            except (ClientError, ValueError) as e:
                log.error("Embed failed for id=%s: %s", row["id"], e)
                return None

    results = await asyncio.gather(*[_one(r) for r in rows])
    return [r for r in results if r is not None]


async def write_batch(pool: asyncpg.Pool, updates: List[Tuple[str, str]]) -> None:
    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE rag SET embedding_v2 = $2::vector, updated_at = NOW() "
            "WHERE id = $1",
            updates,
        )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--ensure-column-only",
        action="store_true",
        help="Create the embedding_v2 column then exit (no backfill).",
    )
    args = ap.parse_args()

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=4)
    try:
        await ensure_column(pool)
        if args.ensure_column_only:
            return

        total_remaining = await remaining_count(pool)
        log.info(
            "Starting backfill: %s rows remaining (model=%s region=%s)",
            total_remaining,
            BEDROCK_EMBEDDING_MODEL,
            BEDROCK_REGION,
        )

        done = 0
        start = time.monotonic()
        while True:
            rows = await fetch_batch(pool, args.batch_size)
            if not rows:
                break
            updates = await embed_batch(rows, args.concurrency)
            if updates:
                await write_batch(pool, updates)
            done += len(updates)
            failed = len(rows) - len(updates)
            elapsed = time.monotonic() - start
            rate = done / elapsed if elapsed > 0 else 0
            log.info(
                "Progress: +%d (failed %d) | total %d/%d | %.1f rows/s",
                len(updates),
                failed,
                done,
                total_remaining,
                rate,
            )
            # If a whole batch failed (e.g. persistent throttling), back off so
            # we don't hot-loop the same NULL rows forever.
            if not updates:
                log.warning("Whole batch failed; sleeping 30s before retry.")
                await asyncio.sleep(30)

        log.info("Backfill complete. Embedded %d rows this run.", done)
        leftover = await remaining_count(pool)
        log.info("Rows still NULL (errors/empty): %d", leftover)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
