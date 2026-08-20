"""
Ollama Provider Implementation
Clean implementation without if/else branching for other providers
"""

import logging
from typing import Dict, List, Any, Optional
from ..agent import LLMProvider
from ..utils.models import get_models_from_env
import aiohttp

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama-specific implementation of LLM provider"""

    # Class-level cache shared across all instances
    _global_cached_models: Optional[Dict[str, List[Dict]]] = None
    _global_models_fetched: bool = False
    _cached_host: Optional[str] = None  # Track which host the cache is for

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = "ollama"
        self.host = config.get("ollama", {}).get("host", "http://localhost:11434")
        self.retriever_model = config.get("ollama", {}).get(
            "retriever_model", "llama3.1"
        )
        self.embedding_model = config.get("ollama", {}).get(
            "embedding_model", "llama3.2"
        )
        self.supported_tool_call_models = config.get("ollama", {}).get(
            "supported_tool_call_models", ["llama3.1", "mistral"]
        )

    def create_embedding_model(self):
        """Create Ollama embedding model"""

        return OllamaEmbeddings(model=self.embedding_model, base_url=self.host)

    def create_retriever_model(self, model: str = None, temperature: float = 0.1):
        """Create Ollama retriever model"""
        # Strip the "imbrace/" prefix if present
        clean_model = model or self.retriever_model
        if clean_model.startswith("imbrace/"):
            clean_model = clean_model[8:]  # Remove "imbrace/" prefix

        return ChatOllama(
            base_url=self.host, model=clean_model, temperature=temperature, top_p=0.5
        )

    def create_llm(
        self,
        model: str,
        available_models: Dict[str, List[Dict]],
        streaming: bool = True,
        temperature: float = 0.1,
    ):
        """Create Ollama LLM for main chat"""

        # Check if model is available in Ollama
        ollama_models = available_models.get("ollamaModels", [])

        # Strip "imbrace/" prefix from model name for comparison and API calls
        clean_model = model
        if clean_model.startswith("imbrace/"):
            clean_model = clean_model[8:]  # Remove "imbrace/" prefix

        if any(m.get("name") == clean_model for m in ollama_models):
            selected_model = clean_model
        else:
            selected_model = self.retriever_model

        return ChatOllama(
            base_url=self.host,
            model=clean_model,
            temperature=temperature,
            top_p=0.5,
            streaming=streaming,
        )

    def _is_vision_model_name(self, name: str) -> bool:
        """
        Heuristically determine if an Ollama model supports vision input.
        """
        name_lower = str(name).lower()
        return "vision" in name_lower or "vl" in name_lower

    def _is_support_thinking_model_name(self, name: str) -> bool:
        """
        Heuristically determine if an Ollama model is reasoning-focused.
        """
        name_lower = str(name).lower()
        thinking_patterns = ["thinking", "reasoning", "r1", "deepseek"]
        return any(p in name_lower for p in thinking_patterns)

    def _is_tool_call_model_name(self, name: str) -> bool:
        """
        Determine if an Ollama model supports tool calling based on configured patterns.
        """
        name_lower = str(name).lower()
        return any(pattern in name_lower for pattern in self.supported_tool_call_models)

    async def get_models_from_provider(self) -> Dict[str, List[Dict]]:
        """
        Strictly fetch available Ollama models from the provider API.

        - On success: return models fetched from API.
        - On any error: raise the exception (no local fallback or cache).
        """
        try:
            logger.info(
                "OllamaProvider.get_models_from_provider: fetching models from %s",
                self.host,
            )
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/tags") as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(
                            f"Ollama API returned status {response.status}: {text}"
                        )

                    data = await response.json()
                    ollama_models: List[Dict[str, Any]] = []
                    for model in data.get("models", []):
                        name = model.get("name", "") or ""
                        is_vision_available = self._is_vision_model_name(name)
                        is_support_thinking = self._is_support_thinking_model_name(name)
                        ollama_models.append(
                            {
                                "name": name,
                                "provider": "ollama",
                                "is_toolCall_available": True,
                                "is_vision_available": is_vision_available,
                                "is_support_thinking": is_support_thinking,
                                "is_shown": False,
                            }
                        )

                    ollama_models.sort(key=lambda x: x["name"])

                    logger.info(
                        "OllamaProvider.get_models_from_provider: fetched %d models from %s",
                        len(ollama_models),
                        self.host,
                    )
                    return {"ollamaModels": ollama_models}
        except Exception as e:
            logger.error(
                "OllamaProvider.get_models_from_provider: failed to fetch models from %s: %s",
                self.host,
                e,
            )
            raise

    async def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get available Ollama models from API with class-level caching.

        Uses the same enriched model interface as `get_models_from_provider` and
        falls back to a small curated list if the API is unavailable.
        """
        # Check if cache is valid for current host
        if (
            OllamaProvider._global_models_fetched
            and OllamaProvider._global_cached_models is not None
            and OllamaProvider._cached_host == self.host
        ):
            logger.debug(f"Returning cached Ollama models for host: {self.host}")
            return OllamaProvider._global_cached_models

        # If host changed, invalidate cache
        if OllamaProvider._cached_host != self.host:
            logger.info(
                f"Host changed from {OllamaProvider._cached_host} to {self.host}, invalidating cache"
            )
            OllamaProvider._global_cached_models = None
            OllamaProvider._global_models_fetched = False
            OllamaProvider._cached_host = self.host

        try:
            logger.info(
                f"Fetching Ollama models from API for host: {self.host} (not cached)"
            )
            # Reuse strict provider method and cache its enriched result
            models = await self.get_models_from_provider()
            OllamaProvider._global_cached_models = models
            OllamaProvider._global_models_fetched = True
            OllamaProvider._cached_host = self.host
            logger.info(
                "Successfully fetched and cached %d Ollama models for host: %s",
                len(models.get("ollamaModels", [])),
                self.host,
            )
            return models
        except Exception as e:
            logger.error(
                "Failed to fetch Ollama models from %s in get_available_models: %s",
                self.host,
                e,
            )

            # If we have cached models from a previous successful call, return them
            if (
                OllamaProvider._global_cached_models is not None
                and OllamaProvider._cached_host == self.host
            ):
                logger.info(
                    f"API call failed, returning previously cached Ollama models for host: {self.host}"
                )
                return OllamaProvider._global_cached_models

            # Fallback to default models if no cache and API call fails
            logger.info(
                "No cached models available for host: %s, using default fallback models",
                self.host,
            )
            
            models_from_env = get_models_from_env("ollama")
            if models_from_env:
                default_names = models_from_env
            else:
                default_names = [
                    "llama2",
                    "llama3.1",
                    "llama3.2",
                    "mistral",
                    "codellama",
                ]
            
            default_models: List[Dict[str, Any]] = []
            for name in default_names:
                default_models.append(
                    {
                        "name": name,
                        "provider": "ollama",
                        "is_toolCall_available": True,
                        "is_vision_available": self._is_vision_model_name(name),
                        "is_support_thinking": self._is_support_thinking_model_name(
                            name
                        ),
                        "is_shown": False,
                    }
                )

            # Cache the fallback models to avoid repeated failures
            OllamaProvider._global_cached_models = {"ollamaModels": default_models}
            OllamaProvider._global_models_fetched = True
            OllamaProvider._cached_host = self.host

            return OllamaProvider._global_cached_models

    @classmethod
    def clear_models_cache(cls):
        """Clear the cached models to force a fresh fetch on next call"""
        logger.info("Clearing Ollama models cache")
        cls._global_cached_models = None
        cls._global_models_fetched = False
        cls._cached_host = None

    @classmethod
    def is_models_cached(cls, host: Optional[str] = None) -> bool:
        """Check if models are currently cached for the given host"""
        if host is None:
            return cls._global_models_fetched and cls._global_cached_models is not None
        return (
            cls._global_models_fetched
            and cls._global_cached_models is not None
            and cls._cached_host == host
        )

    def is_tool_call_available(self, llm) -> bool:
        """Ollama models always support tool calling"""
        return True
