"""
VLLM Provider Implementation
Clean implementation without if/else branching for other providers
"""
from openai import OpenAI

import logging
from typing import Dict, List, Any
from ..agent import LLMProvider
from ..utils.models import get_models_from_env

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class VLLMProvider(LLMProvider):
    """VLLM-specific implementation of LLM provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = "vllm"
        self.host = config.get("vllm", {}).get("host", "http://localhost:11434")
        self.api_key = config.get("vllm", {}).get("api_key", "")
        # Optional dedicated base URL used only when listing available models.
        # Falls back to the chat host when not configured.
        self.base_get_model_url = (
            config.get("vllm", {}).get("base_get_model_url", "") or self.host
        )

        self.retriever_model = config.get("vllm", {}).get("retriever_model", "llama3.1")
        self.embedding_model = config.get("vllm", {}).get("embedding_model", "llama3.2")
        # Embedding may be served by a different vLLM instance than chat; fall
        # back to the chat host when no dedicated embedding host is configured.
        self.embedding_host = (
            config.get("vllm", {}).get("embedding_host", "") or self.host
        )

    def create_embedding_model(self):
        """Create VLLM embedding model"""
        logger.info(
            f"Creating VLLM embedding model: {self.embedding_model} "
            f"(host: {self.embedding_host})"
        )
        return OpenAIEmbeddings(
            openai_api_key=self.api_key or "EMPTY",
            model=self.embedding_model,
            openai_api_base=self.embedding_host,
        )

    def create_retriever_model(self, model: str, temperature: float = 0.1):
        """Create VLLM retriever model"""
        logger.info(f"Creating VLLM retriever model: {model or self.retriever_model}, host: {self.host}, api_key set: {bool(self.api_key)}")

        llm = ChatOpenAI(
            model=model or self.retriever_model,
            api_key=self.api_key,
            base_url=self.host,
            temperature=temperature,
            top_p=0.1,
            # Bound the call: without these the OpenAI SDK defaults (long timeout
            # + 2 retries) turn one dead-upstream call into ~2.5 min. 30s + 1
            # retry fails fast on a dead upstream while still tolerating a real
            # (but slower) retriever reasoning / PDF-table call and one transient
            # blip. This model is retriever-side only; the main chat path
            # (create_llm) is intentionally left untouched.
            timeout=30,
            max_retries=1,
        )
        # Debug: check actual client base_url after ChatOpenAI creation
        client_base_url = getattr(llm, 'openai_api_base', None) or getattr(getattr(llm, 'client', None), '_base_url', None)
        logger.info(f"VLLM retriever ChatOpenAI actual base_url: {client_base_url}")
        return llm

    def create_llm(
        self,
        model: str,
        available_models: Dict[str, List[Dict]],
        streaming: bool = True,
        temperature: float = 0.1,
    ):
        """Create VLLM LLM for main chat"""
        # Check if this is the Kimi model and use Kimi host

        return ChatOpenAI(
            model=model or self.retriever_model,
            api_key=self.api_key,
            base_url=self.host,
            temperature=temperature,
            top_p=0.1,
            streaming=streaming,
        )

    def _is_tool_call_model_name(self, name: str) -> bool:
        """
        Determine if a VLLM model supports tool calling based on its name.

        - Kimi models (names containing 'kimi') are treated as NOT supporting tools
          to avoid 400 errors from the upstream API.
        - All other models are treated as tool-call capable by default.
        """
        if not name:
            # Default to True for non-empty names only; empty means "unknown"
            return True

        name_lower = str(name).lower()

        # Explicitly disable tools for Kimi models such as:
        #   moonshotai/Kimi-VL-A3B-Thinking-2506
        if "kimi" in name_lower:
            return False

        return True

    async def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get available VLLM models from API (safe wrapper)."""
        try:
            return await self.get_models_from_provider()
        except Exception as e:
            logger.error(f"Failed to fetch VLLM models: {str(e)}")
            # Fallback to MODELS env var if API call fails
            models_from_env = get_models_from_env("vllm")
            vllm_models = []
            if models_from_env:
                for name in models_from_env:
                    vllm_models.append(
                        {
                            "name": name,
                            "provider": "vllm",
                            "is_toolCall_available": True,
                            "is_vision_available": False,
                            "is_support_thinking": False,
                            "is_shown": False,
                        }
                    )
            return {"vllmModels": vllm_models}

    async def get_models_from_provider(self) -> Dict[str, List[Dict]]:
        """
        Strictly fetch available VLLM models from the provider API.

        - On success: return models fetched from API.
        - On any error: raise the exception (no local fallback).
        """
        logger.info(
            "VLLMProvider.get_models_from_provider: fetching models from %s",
            self.base_get_model_url,
        )
        client = OpenAI(api_key=self.api_key, base_url=self.base_get_model_url)

        # Fetch models from VLLM API
        models_response = client.models.list()

        # Filter and format models
        vllm_models: List[Dict[str, Any]] = []
        for model in models_response.data:
            model_id = getattr(model, "id", "") or ""
            vllm_models.append(
                {
                    "name": model_id,
                    "provider": "vllm",
                    # VLLM exposes an OpenAI-compatible API; most models allow tool calling.
                    # However, some vendor-specific models (e.g. Kimi) do NOT support tools.
                    "is_toolCall_available": True,
                    # Vision capability is implementation-specific; default to False here.
                    "is_vision_available": False,
                    # Reasoning/"thinking" support depends on the hosted model; default False.
                    "is_support_thinking": False,
                    "is_shown": False,
                }
            )

        # Sort models by name for consistent ordering
        vllm_models.sort(key=lambda x: x["name"])

        logger.info(
            "VLLMProvider.get_models_from_provider: fetched %d models from API",
            len(vllm_models),
        )

        return {"vllmModels": vllm_models}

    def is_tool_call_available(self, llm) -> bool:
        """VLLM models always support tool calling."""
        return True
