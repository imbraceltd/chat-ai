"""
Google AI (Gemini) Provider Implementation (Google REST)

This provider uses Google's Generative Language REST endpoints (v1) directly:
- /models/{model}:generateContent
- /models/{model}:streamGenerateContent?alt=sse
- /models/{model}:embedContent

It supports any `base_url` that ends with `/v1` (or includes it), including gateways.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Sequence, AsyncIterator, cast

import httpx
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from ..agent import LLMProvider

logger = logging.getLogger(__name__)

DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1"


def _normalize_google_model_name(name: str) -> str:
    """
    Normalize model name for URL building.
    """
    if not name:
        return name
    return name if name.startswith("models/") else f"models/{name}"


def _extract_text_from_google_candidate(candidate: dict) -> str:
    """
    Extract text from a Google generateContent response candidate.
    """
    if not isinstance(candidate, dict):
        return ""
    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        return ""
    texts: List[str] = []
    for p in parts:
        if isinstance(p, dict) and isinstance(p.get("text"), str):
            texts.append(p["text"])
    return "".join(texts)


def _messages_to_google_payload(messages: Sequence[BaseMessage]) -> Dict[str, Any]:
    """
    Convert messages to Google `generateContent` payload.

    We map:
    - SystemMessage(s) -> systemInstruction
    - HumanMessage -> contents role=user
    - AIMessage -> contents role=model
    """
    system_texts: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            if msg.content:
                system_texts.append(str(msg.content))
            continue

        role = "user"
        if isinstance(msg, AIMessage):
            role = "model"

        text = msg.content if msg.content is not None else ""
        if not isinstance(text, str):
            text = str(text)
        contents.append({"role": role, "parts": [{"text": text}]})

    payload: Dict[str, Any] = {"contents": contents}
    if system_texts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_texts)}]}
    return payload


class GoogleRESTEmbeddings(Embeddings):
    """
    Embeddings via Google Generative Language REST (v1) endpoint:
    POST /models/{model}:embedContent?key=...
    """

    def __init__(self, *, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        return headers

    def _params(self) -> Dict[str, str]:
        return {"key": self.api_key} if self.api_key else {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Sync wrapper (rarely used in our async codepaths)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        model_path = _normalize_google_model_name(self.model)
        url = f"{self.base_url}/{model_path}:embedContent"
        payload = {"content": {"parts": [{"text": text or ""}]}}
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                url, headers=self._headers(), params=self._params(), json=payload
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                # Ensure body is readable in logs/errors.
                body = resp.text
                raise RuntimeError(
                    f"Google embedContent failed: HTTP {resp.status_code}: {body}"
                ) from e
            data = resp.json()
        embedding = (data or {}).get("embedding") or {}
        values = embedding.get("values") or []
        return [float(x) for x in values] if isinstance(values, list) else []


class GoogleRESTChatModel(BaseChatModel):
    """
    Chat model via Google Generative Language REST (v1) endpoints:
    - generateContent
    - streamGenerateContent (SSE)
    """

    model: str
    base_url: str
    api_key: str
    temperature: float = 0.1
    streaming: bool = True

    @property
    def _llm_type(self) -> str:
        return "google-rest-v1"

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["x-goog-api-key"] = self.api_key
        return headers

    def _params(self) -> Dict[str, str]:
        return {"key": self.api_key} if self.api_key else {}

    def _request_payload(self, messages: Sequence[BaseMessage]) -> Dict[str, Any]:
        payload = _messages_to_google_payload(messages)
        payload["generationConfig"] = {"temperature": self.temperature}
        return payload

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        model_path = _normalize_google_model_name(self.model)
        url = f"{self.base_url.rstrip('/')}/{model_path}:generateContent"
        payload = self._request_payload(messages)
        with httpx.Client(timeout=60) as client:
            last_error: Exception | None = None
            for attempt in range(3):
                resp = client.post(
                    url, headers=self._headers(), params=self._params(), json=payload
                )
                if resp.status_code == 429 and attempt < 2:
                    # Basic backoff for rate limits.
                    delay = 0.5 * (2**attempt)
                    import time

                    time.sleep(delay)
                    continue

                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    body = resp.text
                    last_error = RuntimeError(
                        f"Google generateContent failed: HTTP {resp.status_code}: {body}"
                    )
                    break

                data = resp.json() or {}
                last_error = None
                break

            if last_error:
                raise last_error

        candidates = data.get("candidates") or []
        text = ""
        if isinstance(candidates, list) and candidates:
            text = _extract_text_from_google_candidate(cast(dict, candidates[0]))

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        base = self.base_url.rstrip("/")
        # Use SSE streaming when possible
        model_path = _normalize_google_model_name(self.model)
        url = f"{base}/{model_path}:streamGenerateContent"
        payload = self._request_payload(messages)
        params = dict(self._params())
        params["alt"] = "sse"

        async with httpx.AsyncClient(timeout=None) as client:
            for attempt in range(3):
                async with client.stream(
                    "POST",
                    url,
                    headers=self._headers(),
                    params=params,
                    json=payload,
                ) as resp:
                    if resp.status_code >= 400:
                        # IMPORTANT: read body before raising so error formatting
                        # doesn't crash with "without having called read()".
                        body_bytes = await resp.aread()
                        body = body_bytes.decode("utf-8", "replace")

                        # Retry basic 429 rate limits.
                        if resp.status_code == 429 and attempt < 2:
                            delay = 0.5 * (2**attempt)
                            await asyncio.sleep(delay)
                            continue

                        raise RuntimeError(
                            f"Google streamGenerateContent failed: HTTP {resp.status_code}: {body}"
                        )

                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        # SSE format: "data: {...}"
                        if line.startswith("data:"):
                            line = line[len("data:") :].strip()
                        if line == "[DONE]":
                            return
                        try:
                            event = httpx.Response(200, content=line).json()
                        except Exception:
                            continue

                        if not isinstance(event, dict):
                            continue
                        candidates = event.get("candidates") or []
                        if not isinstance(candidates, list) or not candidates:
                            continue
                        text = _extract_text_from_google_candidate(
                            cast(dict, candidates[0])
                        )
                        if not text:
                            continue

                        yield ChatGenerationChunk(message=AIMessageChunk(content=text))
                # If we exit the stream context normally (no yields), stop retrying.
                return


class GoogleAIProvider(LLMProvider):
    """
    Google AI (Gemini) provider implemented via Google REST endpoints (v1beta).

    Expected config:
    - config["google"]["api_key"]
    - config["google"]["base_url"] (e.g. https://generativelanguage.googleapis.com/v1beta or a gateway ending in /v1beta)
    - config["google"]["model"] (default: gemini-2.0-flash)
    - config["google"]["embedding_model"] (default: text-embedding-004)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = "google"

        google_cfg = config.get("google", {}) or {}

        self.api_key = google_cfg.get("api_key", "") or ""
        self.base_url = google_cfg.get("base_url", "") or DEFAULT_GOOGLE_BASE_URL

        # Default chat/embedding model names (Gemini names)
        self.chat_model = google_cfg.get("model", "gemini-2.0-flash")
        self.embedding_model = google_cfg.get("embedding_model", "text-embedding-004")

        if not self.api_key:
            logger.warning(
                "GoogleAIProvider initialized without api_key. Calls will fail until api_key is provided."
            )
        if not self.base_url:
            logger.warning(
                "GoogleAIProvider initialized without base_url. Calls will default to OpenAI SDK defaults and likely fail."
            )

    def _get_chat_model(self, model: str | None, temperature: float, streaming: bool):
        selected_model = model or self.chat_model
        return GoogleRESTChatModel(
            model=selected_model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=temperature,
            streaming=streaming,
        )

    def create_embedding_model(self):
        """Create embedding model via Google REST `:embedContent`."""
        return GoogleRESTEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.embedding_model,
        )

    def create_retriever_model(self, model: str, temperature: float = 0.1):
        return self._get_chat_model(model=model, temperature=temperature, streaming=False)

    def create_llm(
        self,
        model: str,
        available_models: Dict[str, List[Dict]],
        streaming: bool = True,
        temperature: float = 0.1,
    ):
        google_models = available_models.get("googleModels", [])
        requested_model = model or self.chat_model

        selected_model = (
            requested_model
            if any(m.get("name") == requested_model for m in google_models)
            else self.chat_model
        )

        logger.info(
            "GoogleAIProvider.create_llm: using model=%s (requested=%s) base_url=%s",
            selected_model,
            requested_model,
            self.base_url,
        )
        return self._get_chat_model(
            model=selected_model,
            temperature=temperature,
            streaming=streaming,
        )

    def _is_tool_call_model_name(self, model_name: str) -> bool:
        """
        Heuristically determine if a Google AI model (by name) supports tools.
        Shared between API-based discovery and local fallbacks.
        """
        # Always treat every Google AI model as tool-capable.
        return True

    def _is_vision_model_name(self, model_name: str) -> bool:
        """
        Heuristically determine if a Google AI model (by name) supports vision.
        """
        model_name_lower = str(model_name).lower()
        vision_patterns = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash",
        ]
        return any(pattern in model_name_lower for pattern in vision_patterns)

    def _is_support_thinking_model_name(self, model_name: str) -> bool:
        """
        Heuristically determine if a Google AI model (by name) is reasoning-focused.
        """
        model_name_lower = str(model_name).lower()
        thinking_patterns = [
            "thinking",
            "reasoning",
            "gemini-1.5-pro",
            "gemini-2.0-pro",
        ]
        return any(pattern in model_name_lower for pattern in thinking_patterns)

    async def get_models_from_provider(self) -> Dict[str, List[Dict]]:
        """
        Fetch available models from the provider.
        """
        try:
            url = f"{self.base_url.rstrip('/')}/models"
            headers = {"x-goog-api-key": self.api_key} if self.api_key else None
            params = {"key": self.api_key} if self.api_key else None

            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.get(url, headers=headers, params=params)
                resp.raise_for_status()
                payload = resp.json()

            google_models: List[Dict[str, Any]] = []

            # Google-style list: {"models": [{"name": "models/gemini-2.0-flash", ...}, ...]}
            if isinstance(payload, dict) and isinstance(payload.get("models"), list):
                for m in payload["models"]:
                    if not isinstance(m, dict):
                        continue
                    raw_name = m.get("name", "") or ""
                    # Keep raw model name as returned by the API (e.g. "models/gemini-2.0-flash")
                    if not raw_name:
                        continue

                    # Only expose chat-capable models in the LLM provider list.
                    supported = m.get("supportedGenerationMethods", []) or []
                    if not isinstance(supported, list):
                        supported = []
                    chat_methods = {
                        "generateContent",
                        "streamGenerateContent",
                        "bidiGenerateContent",
                        "batchGenerateContent",
                    }
                    if not any(method in chat_methods for method in supported):
                        continue

                    google_models.append(
                        {
                            "name": raw_name,
                            "provider": "google",
                            "is_toolCall_available": self._is_tool_call_model_name(raw_name),
                            "is_vision_available": self._is_vision_model_name(raw_name),
                            "is_support_thinking": self._is_support_thinking_model_name(raw_name),
                            "is_shown": False,
                        }
                    )

                google_models.sort(key=lambda x: x["name"])
                return {"googleModels": google_models}

            logger.warning(
                "GoogleAIProvider.get_models_from_provider: unexpected /models payload shape from %s; falling back to defaults",
                url,
            )
            return await self.get_available_models()
        except Exception as e:
            logger.error(
                "GoogleAIProvider.get_models_from_provider: failed to fetch models from provider: %s",
                e,
            )
            raise

    async def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get available Google AI models from API or MODELS env var with safe fallback."""
        models_from_env = get_models_from_env("google")
        if models_from_env:
            default_names = models_from_env
        else:
            default_names = [
                "gemini-2.0-flash",
                "gemini-2.0-flash-lite",
                "gemini-pro",
            ]
        models: List[Dict[str, Any]] = []
        for name in default_names:
            models.append(
                {
                    "name": name,
                    "provider": "google",
                    "is_toolCall_available": self._is_tool_call_model_name(name),
                    "is_vision_available": self._is_vision_model_name(name),
                    "is_support_thinking": self._is_support_thinking_model_name(name),
                    "is_shown": False,
                }
            )
        return {"googleModels": models}

    def is_tool_call_available(self, llm) -> bool:
        """
        Check if Google AI model supports tool calling (function calling).

        Always report tool calling as available, consistent with the other
        providers.

        NOTE: Google REST v1 uses a different tool-call schema than our OpenAI
        tool schema, so actually binding tools here may fail upstream.
        """
        return True


