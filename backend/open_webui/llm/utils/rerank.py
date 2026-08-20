import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def rerank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    reranking_function: Any,
    top_n: int = 3,
    relevance_threshold: float = 0.0,
    content_key: str = "pageContent",
) -> List[Dict[str, Any]]:
    """
    Rerank documents using a Sentence Transformers CrossEncoder or ColBERT model.

    Args:
        query: The original user query to score against.
        documents: List of dicts, each containing at least a content_key field.
        reranking_function: A model with .rank() (CrossEncoder) or .predict() (ColBERT).
        top_n: Maximum number of results to return after reranking.
        relevance_threshold: Minimum rerank score to keep a document (0.0 = keep all).
        content_key: Key to extract text content from each document dict.

    Returns:
        Reranked and filtered list of document dicts with "rerank_score" added.
    """
    if not documents or not query.strip():
        return documents

    # Build passages with table-aware context for better reranking accuracy
    passages = []
    for doc in documents:
        content = doc.get(content_key, "")
        metadata = doc.get("metadata", {})
        if metadata.get("isTable"):
            content = f"[Tabular data] {content}"
        passages.append(content)

    try:
        # Prefer CrossEncoder.rank() — handles sorting internally
        ranks = reranking_function.rank(
            query, passages, top_k=top_n, return_documents=False
        )
    except (AttributeError, TypeError):
        # Fallback for models without .rank() (e.g., ColBERT)
        scores = reranking_function.predict(
            [(query, passage) for passage in passages]
        )
        ranks = sorted(
            [{"corpus_id": i, "score": float(s)} for i, s in enumerate(scores)],
            key=lambda x: x["score"],
            reverse=True,
        )[:top_n]

    result = []
    for rank in ranks:
        idx = rank["corpus_id"]
        score = rank["score"]
        if relevance_threshold > 0.0 and score < relevance_threshold:
            continue
        doc = documents[idx].copy()
        doc["rerank_score"] = score
        result.append(doc)

    logger.info(
        f"Reranked {len(documents)} documents to {len(result)} "
        f"(top_n={top_n}, threshold={relevance_threshold})"
    )
    return result
