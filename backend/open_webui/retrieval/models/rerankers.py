import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class OllamaReranker:
    """Reranker using an Ollama embedding model with cosine similarity.

    Calls the Ollama server API to generate embeddings for query and documents,
    then uses sentence_transformers.util.cos_sim to rank documents.
    Suitable for models like Qwen/Qwen3-Embedding-4B served via Ollama.
    """

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        logger.info(f"OllamaReranker initialized: model={model}, url={base_url}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from Ollama API."""
        response = requests.post(
            f"{self.base_url}/api/embed",
            headers={"Content-Type": "application/json"},
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if "embeddings" in data:
            return data["embeddings"]
        raise ValueError(f"Unexpected Ollama response: {data}")

    def rank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        from sentence_transformers import util

        all_texts = [query] + passages
        embeddings = self._get_embeddings(all_texts)
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        scores = util.cos_sim(query_emb, doc_embs)[0].tolist()

        ranked = sorted(
            [{"corpus_id": i, "score": s} for i, s in enumerate(scores)],
            key=lambda x: x["score"],
            reverse=True,
        )
        if top_k:
            ranked = ranked[:top_k]
        if return_documents:
            for item in ranked:
                item["text"] = passages[item["corpus_id"]]
        return ranked

    def predict(self, sentence_pairs: List[tuple]) -> List[float]:
        """Fallback interface compatible with CrossEncoder .predict() calls."""
        query = sentence_pairs[0][0]
        passages = [pair[1] for pair in sentence_pairs]
        ranks = self.rank(query, passages)
        score_map = {r["corpus_id"]: r["score"] for r in ranks}
        return [score_map.get(i, 0.0) for i in range(len(sentence_pairs))]


class BedrockReranker:
    """Cross-encoder reranker using the Amazon Bedrock Rerank API.

    Uses bedrock-agent-runtime `rerank` with amazon.rerank-v1:0 (a first-party AWS
    model — no AWS Marketplace subscription needed, unlike the Cohere rerank model).
    Offloads compute to Bedrock (no local model / GPU). Credentials are read from
    BEDROCK_* env vars — the same ones the embedding path uses — so it works
    wherever Titan embedding already works.
    """

    def __init__(
        self,
        model: str = "amazon.rerank-v1:0",
        region: str = "",
        access_key: str = "",
        secret_key: str = "",
    ):
        import os
        import boto3

        self.model = model or "amazon.rerank-v1:0"
        self.region = region or os.getenv("BEDROCK_REGION", "us-west-2")
        access_key = access_key or os.getenv("BEDROCK_ACCESS_KEY_ID") or ""
        secret_key = secret_key or os.getenv("BEDROCK_SECRET_ACCESS_KEY") or ""
        kwargs: Dict[str, Any] = {"region_name": self.region}
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        self.client = boto3.client("bedrock-agent-runtime", **kwargs)
        self.model_arn = (
            f"arn:aws:bedrock:{self.region}::foundation-model/{self.model}"
        )
        logger.info(
            f"BedrockReranker initialized: model={self.model}, region={self.region}"
        )

    def rank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        if not passages:
            return []
        n = min(top_k or len(passages), len(passages))
        try:
            resp = self.client.rerank(
                queries=[{"type": "TEXT", "textQuery": {"text": query}}],
                sources=[
                    {
                        "type": "INLINE",
                        "inlineDocumentSource": {
                            "type": "TEXT",
                            "textDocument": {"text": p},
                        },
                    }
                    for p in passages
                ],
                rerankingConfiguration={
                    "type": "BEDROCK_RERANKING_MODEL",
                    "bedrockRerankingConfiguration": {
                        "numberOfResults": n,
                        "modelConfiguration": {"modelArn": self.model_arn},
                    },
                },
            )
            ranked = [
                {"corpus_id": int(r["index"]), "score": float(r["relevanceScore"])}
                for r in resp.get("results", [])
            ]
        except Exception as e:
            logger.error(
                f"BedrockReranker scoring failed: {e}. Falling back to original order."
            )
            ranked = [{"corpus_id": i, "score": 0.0} for i in range(len(passages))]

        if top_k:
            ranked = ranked[:top_k]
        if return_documents:
            for item in ranked:
                item["text"] = passages[item["corpus_id"]]
        return ranked

    def predict(self, sentence_pairs: List[tuple]) -> List[float]:
        """Fallback interface compatible with CrossEncoder .predict() calls."""
        query = sentence_pairs[0][0]
        passages = [pair[1] for pair in sentence_pairs]
        ranks = self.rank(query, passages)
        score_map = {r["corpus_id"]: r["score"] for r in ranks}
        return [score_map.get(i, 0.0) for i in range(len(sentence_pairs))]


class OpenAIReranker:
    """Reranker using an OpenAI LLM to score document relevance.

    Sends a single prompt with query + all passages, asks the model to return
    a JSON array of relevance scores, then ranks by score.
    """

    SCORING_PROMPT = """You are a relevance scoring system. Given a query and a list of documents, score each document's relevance to the query on a scale of 0 to 10, where 0 means completely irrelevant and 10 means perfectly relevant.

Query: {query}

Documents:
{documents}

Return ONLY a JSON array of numbers representing the relevance score for each document in order. Example: [8.5, 3.2, 9.1]"""

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        logger.info(f"OpenAIReranker initialized: model={model}")

    def rank(
        self,
        query: str,
        passages: List[str],
        top_k: Optional[int] = None,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        documents_text = "\n".join(
            f"[{i}] {passage}" for i, passage in enumerate(passages)
        )
        prompt = self.SCORING_PROMPT.format(query=query, documents=documents_text)

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"].strip()

            # Handle markdown code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            scores = json.loads(content)

            if not isinstance(scores, list) or len(scores) != len(passages):
                logger.warning(
                    f"OpenAIReranker: expected {len(passages)} scores, got "
                    f"{len(scores) if isinstance(scores, list) else 'non-list'}. "
                    "Falling back to equal scores."
                )
                scores = [5.0] * len(passages)

        except Exception as e:
            logger.error(f"OpenAIReranker scoring failed: {e}. Falling back to equal scores.")
            scores = [5.0] * len(passages)

        ranked = sorted(
            [{"corpus_id": i, "score": float(s)} for i, s in enumerate(scores)],
            key=lambda x: x["score"],
            reverse=True,
        )
        if top_k:
            ranked = ranked[:top_k]
        if return_documents:
            for item in ranked:
                item["text"] = passages[item["corpus_id"]]
        return ranked

    def predict(self, sentence_pairs: List[tuple]) -> List[float]:
        """Fallback interface compatible with CrossEncoder .predict() calls."""
        query = sentence_pairs[0][0]
        passages = [pair[1] for pair in sentence_pairs]
        ranks = self.rank(query, passages)
        score_map = {r["corpus_id"]: r["score"] for r in ranks}
        return [score_map.get(i, 0.0) for i in range(len(sentence_pairs))]
