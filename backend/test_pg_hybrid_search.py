#!/usr/bin/env python3
"""Integration tests for PostgreSQL hybrid search (vector + FTS + RRF).

Run from backend/:
    python test_pg_hybrid_search.py
"""

import asyncio
import os
import sys

# ── Env must be set before any app imports ────────────────────────────────────
os.environ.setdefault("DB_TYPE", "postgresql")
if not os.environ.get("DATABASE_URL"):
    sys.exit(
        "DATABASE_URL must be set in the environment "
        "(e.g. postgresql://user:pass@host:5432/dbname) to run this test."
    )

import asyncpg


# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []


def check(name: str, condition: bool, detail: str = ""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append(condition)
    return condition


async def get_real_filter(pool) -> dict:
    """Pick a real org_id + assistant_id that has rag rows."""
    row = await pool.fetchrow(
        """
        SELECT org_id, assistant_id
        FROM rag
        WHERE org_id IS NOT NULL AND org_id <> ''
          AND assistant_id IS NOT NULL AND assistant_id <> ''
          AND embedding IS NOT NULL
        LIMIT 1
        """
    )
    if row:
        return {
            "org_id": {"$eq": row["org_id"]},
            "$or": [{"assistant_id": {"$eq": row["assistant_id"]}}],
        }
    # Fallback: no filter
    return {}


async def get_sample_text(pool, pre_filter: dict) -> str:
    """Get a short snippet of real text to query with."""
    from open_webui.llm.utils.pg_adapters import _rag_filter_to_where

    params: list = []
    where = _rag_filter_to_where(pre_filter, params)
    row = await pool.fetchrow(
        f"""
        SELECT text FROM rag
        WHERE text IS NOT NULL AND length(text) > 20
          AND ({where})
        LIMIT 1
        """,
        *params,
    )
    if row:
        # Take first 6 words
        words = (row["text"] or "").split()[:6]
        return " ".join(words)
    return "test"


# ── Test suite ────────────────────────────────────────────────────────────────

async def test_filter_builder():
    print("\n=== _rag_filter_to_where ===")
    from open_webui.llm.utils.pg_adapters import _rag_filter_to_where

    params: list = []
    where = _rag_filter_to_where(
        {"org_id": {"$eq": "org1"}, "$or": [{"assistant_id": {"$eq": "a1"}}, {"board_id": {"$in": ["b1", "b2"]}}]},
        params,
    )
    check("generates WHERE clause", "org_id" in where and "$" in where)
    check("params populated", len(params) == 3, f"len={len(params)}")
    check("no MongoDB syntax in output", "$eq" not in where and "$or" not in where)


async def test_rrf_merge():
    print("\n=== _rrf_merge ===")
    from open_webui.llm.utils.pg_adapters import _rrf_merge

    vec = [
        {"pageContent": "alpha", "metadata": {}, "vectorScore": 0.9},
        {"pageContent": "beta",  "metadata": {}, "vectorScore": 0.7},
        {"pageContent": "gamma", "metadata": {}, "vectorScore": 0.5},
    ]
    fts = [
        {"pageContent": "beta",  "metadata": {}, "fullTextScore": 0.8},
        {"pageContent": "delta", "metadata": {}, "fullTextScore": 0.6},
        {"pageContent": "alpha", "metadata": {}, "fullTextScore": 0.4},
    ]
    merged = _rrf_merge(vec, fts, k=10)

    check("returns list", isinstance(merged, list))
    check("no duplicates", len(merged) == len({d["pageContent"] for d in merged}))
    # alpha appears in both lists at rank 0 and 2 → should beat delta (only in FTS rank 1)
    top_keys = [d["pageContent"] for d in merged[:2]]
    check("alpha and beta rank highest (appear in both lists)", "alpha" in top_keys and "beta" in top_keys,
          str(top_keys))
    check("combined_score present", all("combined_score" in d for d in merged))


async def test_vector_search(pool):
    print("\n=== PgVectorStore.asimilarity_search_with_score ===")
    from open_webui.llm.utils.pg_adapters import PgVectorStore
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="text-embedding-3-small",
    )
    store = PgVectorStore(embeddings=embeddings)
    pre_filter = await get_real_filter(pool)
    query = await get_sample_text(pool, pre_filter)
    print(f"  query: '{query}', filter keys: {list(pre_filter.keys())}")

    results_vs = await store.asimilarity_search_with_score(query=query, k=5, pre_filter=pre_filter)

    check("returns list", isinstance(results_vs, list))
    check("returns ≤ 5 results", len(results_vs) <= 5, f"got {len(results_vs)}")
    if results_vs:
        doc, score = results_vs[0]
        check("score in [0, 1]", 0.0 <= score <= 1.0, f"score={score:.4f}")
        check("doc has page_content", bool(doc.page_content))
        check("metadata has org_id", "org_id" in doc.metadata)
        print(f"  top result score={score:.4f}, text='{doc.page_content[:60]}...'")


async def test_full_text_search(pool):
    print("\n=== PgVectorStore.full_text_search ===")
    from open_webui.llm.utils.pg_adapters import PgVectorStore
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="text-embedding-3-small",
    )
    store = PgVectorStore(embeddings=embeddings)
    pre_filter = await get_real_filter(pool)
    query = await get_sample_text(pool, pre_filter)
    print(f"  query: '{query}', filter keys: {list(pre_filter.keys())}")

    fts_results = await store.full_text_search(query=query, k=5, pre_filter=pre_filter)

    check("returns list", isinstance(fts_results, list))
    check("returns ≤ 5 results", len(fts_results) <= 5, f"got {len(fts_results)}")
    if fts_results:
        top = fts_results[0]
        check("has pageContent", bool(top.get("pageContent")))
        check("has fullTextScore", "fullTextScore" in top)
        check("score in [0, 1]", 0.0 <= top["fullTextScore"] <= 1.0,
              f"score={top['fullTextScore']:.4f}")
        print(f"  top result score={top['fullTextScore']:.4f}, text='{top['pageContent'][:60]}...'")


async def test_hybrid_search(pool):
    print("\n=== PgVectorStore.hybrid_search ===")
    from open_webui.llm.utils.pg_adapters import PgVectorStore
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="text-embedding-3-small",
    )
    store = PgVectorStore(embeddings=embeddings)
    pre_filter = await get_real_filter(pool)
    query = await get_sample_text(pool, pre_filter)
    print(f"  query: '{query}', filter keys: {list(pre_filter.keys())}")

    hybrid = await store.hybrid_search(query=query, k=10, pre_filter=pre_filter)

    check("returns list", isinstance(hybrid, list))
    check("returns ≤ 10 results", len(hybrid) <= 10, f"got {len(hybrid)}")
    check("no duplicates in results",
          len(hybrid) == len({d["pageContent"] for d in hybrid}),
          f"len={len(hybrid)}")
    if hybrid:
        top = hybrid[0]
        check("has combined_score", "combined_score" in top, str(list(top.keys())))
        check("has vectorScore or fullTextScore",
              top.get("vectorScore", 0) > 0 or top.get("fullTextScore", 0) > 0)
        print(f"  top combined_score={top['combined_score']:.4f}, "
              f"vec={top.get('vectorScore',0):.3f}, fts={top.get('fullTextScore',0):.3f}")
        print(f"  text='{top['pageContent'][:80]}...'")


async def test_no_filter_fallback():
    """Hybrid search still works with empty filter (no org scope)."""
    print("\n=== hybrid_search with no filter ===")
    from open_webui.llm.utils.pg_adapters import PgVectorStore
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="text-embedding-3-small",
    )
    store = PgVectorStore(embeddings=embeddings)
    hybrid = await store.hybrid_search(query="hello world", k=5, pre_filter={})
    check("works without filter", isinstance(hybrid, list))
    check("returns results", len(hybrid) > 0, f"got {len(hybrid)}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=4)

    await test_filter_builder()
    await test_rrf_merge()

    # These tests hit OpenAI embeddings API — skip if no key
    if os.environ.get("OPENAI_API_KEY"):
        await test_vector_search(pool)
        await test_full_text_search(pool)
        await test_hybrid_search(pool)
        await test_no_filter_fallback()
    else:
        print("\n[SKIP] OPENAI_API_KEY not set — skipping embedding tests")

    await pool.close()

    passed = sum(results)
    total = len(results)
    color = "\033[32m" if passed == total else "\033[31m"
    print(f"\n{color}{'='*40}\n{passed}/{total} tests passed\n{'='*40}\033[0m")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
