"""
OpenAI Provider Implementation
Clean implementation without if/else branching for other providers
"""

import logging
from typing import Dict, List, Any
from ..agent import LLMProvider
from ..utils.models import get_models_from_env
import openai

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI-specific implementation of LLM provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = "openai"
        openapi_cfg = config.get("openApi", {}) or {}

        self.api_key = openapi_cfg.get("api_key", "")
        self.api_type = openapi_cfg.get("api_type", "gpt-4o")
        self.base_url = openapi_cfg.get("base_url", "") or None

        # Token limits configuration
        self.max_completion_tokens = openapi_cfg.get("max_completion_tokens")
        self.max_output_tokens = openapi_cfg.get("max_output_tokens")
        self.max_tokens = openapi_cfg.get("max_tokens")

        self.embedding_model_name = "text-embedding-3-small"
        # Kimi-specific configuration
        self.kimi_host = config.get("kimi", {}).get(
            "host", "http://localhost:8000")
        self.kimi_model = "moonshotai/Kimi-VL-A3B-Thinking-2506"

    def create_embedding_model(self):
        """Create OpenAI embedding model"""

        kwargs: Dict[str, Any] = {
            "openai_api_key": self.api_key,
            "model": self.embedding_model_name,
        }

        # If a custom base URL is provided, route embeddings through that endpoint.
        if self.base_url:
            kwargs["openai_api_base"] = self.base_url

        return OpenAIEmbeddings(**kwargs)

    def create_retriever_model(self, model: str, temperature: float = 0.1):
        """Create OpenAI retriever model"""
        # Check if this is the Kimi model and use Kimi host
        if model in [self.kimi_model, "Kimi-VL-A3B-Thinking-2506"]:
            logger.info(
                f"Using Kimi host {self.kimi_host} for retriever model {model}")
            kimi_kwargs: Dict[str, Any] = {
                "model": model or self.api_type,
                "api_key": self.api_key,
                "base_url": self.kimi_host,
                "temperature": temperature,
                "top_p": 0.1,
            }
            # Apply token limits if configured
            kimi_kwargs.update(self._get_token_limit_kwargs())
            return ChatOpenAI(**kimi_kwargs)
        else:
            kwargs: Dict[str, Any] = {
                "model": model or self.api_type,
                "api_key": self.api_key,
                "temperature": temperature,
                "top_p": 0.1,
            }

            # For generic OpenAI-compatible backends (including when type="custom"),
            # honour the configured base_url so chat / retrieval calls hit the
            # specified gateway instead of api.openai.com.
            if self.base_url:
                kwargs["base_url"] = self.base_url

            # Apply token limits if configured
            kwargs.update(self._get_token_limit_kwargs())

            return ChatOpenAI(**kwargs)

    def _get_token_limit_kwargs(self) -> Dict[str, Any]:
        """
        Build token limit kwargs based on config for ChatOpenAI.

        ChatOpenAI directly supports 'max_tokens' parameter.
        For non-standard parameters like 'max_completion_tokens', we use 'model_kwargs'.

        Priority: max_completion_tokens > max_output_tokens > max_tokens
        """
        kwargs: Dict[str, Any] = {}
        model_kwargs: Dict[str, Any] = {}

        # max_completion_tokens (for o1/o3) needs to be passed via model_kwargs
        if self.max_completion_tokens is not None:
            model_kwargs["max_completion_tokens"] = self.max_completion_tokens
        # max_output_tokens - also pass via model_kwargs for custom providers
        elif self.max_output_tokens is not None:
            model_kwargs["max_output_tokens"] = self.max_output_tokens
        # max_tokens is standard and supported directly by ChatOpenAI
        elif self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        return kwargs

    def create_llm(
        self,
        model: str,
        available_models: Dict[str, List[Dict]],
        streaming: bool = True,
        temperature: float = 0.1,
    ):
        """Create OpenAI LLM for main chat"""
        # Check if this is the Kimi model and use Kimi host
        if model in [self.kimi_model, "Kimi-VL-A3B-Thinking-2506"]:
            logger.info(
                f"Using Kimi host {self.kimi_host} for main LLM {model}")
            kimi_kwargs: Dict[str, Any] = {
                "model": self.kimi_model,
                "api_key": self.api_key,
                "base_url": self.kimi_host,
                "temperature": temperature,
                "top_p": 0.1,
                "streaming": streaming,
            }
            # Apply token limits if configured
            kimi_kwargs.update(self._get_token_limit_kwargs())
            return ChatOpenAI(**kimi_kwargs)
        else:
            kwargs: Dict[str, Any] = {
                "model": model or self.api_type,
                "api_key": self.api_key,
                "temperature": temperature,
                "top_p": 0.1,
                "streaming": streaming,
            }

            # Route normal chat traffic via custom OpenAI-compatible endpoint when provided.
            if self.base_url:
                kwargs["base_url"] = self.base_url

            # Apply token limits if configured
            kwargs.update(self._get_token_limit_kwargs())

            return ChatOpenAI(**kwargs)

    async def get_models_from_provider(self) -> Dict[str, List[Dict]]:
        """
        Strictly fetch available OpenAI models from the provider API.

        - On success: return models fetched from API.
        - On any error: raise the exception (no local fallback).
        """
        try:
            logger.info(
                "OpenAIProvider.get_models_from_provider: fetching models from OpenAI API",
            )
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            client = openai.OpenAI(**client_kwargs)
            models_response = client.models.list()

            openai_models: List[Dict[str, Any]] = []
            for model in models_response.data:
                raw_id = getattr(model, "id", "") or ""
                model_id_lower = str(raw_id).lower()
                non_chat_patterns = [
                    "whisper",
                    "tts",
                    "audio-",
                ]
                if any(p in model_id_lower for p in non_chat_patterns):
                    continue

                openai_models.append(
                    {
                        "name": raw_id,
                        "provider": "openai",
                        "is_toolCall_available": self._is_tool_call_model_id(raw_id),
                        "is_vision_available": self._is_vision_model_id(raw_id),                        "is_support_thinking": self._is_support_thinking_model_id(
                            raw_id
                        ),
                        "is_parallel_tool_calls_available": self._is_parallel_tool_calls_model_id(
                            raw_id
                        ),
                        "is_prompt_cache_available": self._is_prompt_cache_model_id(
                            raw_id
                        ),
                        "is_shown": False,
                    }
                )

            openai_models.sort(key=lambda x: x["name"])

            logger.info(
                "OpenAIProvider.get_models_from_provider: fetched %d models from API",
                len(openai_models),
            )

            return {"openaiModels": openai_models}
        except Exception as e:
            logger.error(
                "OpenAIProvider.get_models_from_provider: failed to fetch models from API: %s",
                e,
            )
            # Let caller handle the failure
            raise

    def _is_tool_call_model_id(self, model_id: str) -> bool:
        """
        Heuristically determine if an OpenAI model supports tool/function calling.
        """
        # Always treat every model as tool-capable, regardless of provider.
        return True

    def _is_support_thinking_model_id(self, model_id: str) -> bool:
        """
        Heuristically determine if an OpenAI model supports "thinking"/reasoning mode.
        """
        if not model_id:
            return False

        model_id_lower = str(model_id).lower()

        # Explicitly match well-known reasoning / thinking style models
        thinking_patterns = [
            "thinking",  # e.g. vendor-specific thinking models
            "reasoning",
            "o1",  # OpenAI o1 family
            "o3",  # Future reasoning variants
            "r1",  # Reasoning-focused variants (e.g. deepseek-style)
        ]
        return any(p in model_id_lower for p in thinking_patterns)

    def _is_parallel_tool_calls_model_id(self, model_id: str) -> bool:
        """
        Heuristically determine if an OpenAI model supports parallel tool calls.

        For now we assume that any tool-capable GPT model also supports
        parallel tool calls, which matches most modern GPT-* chat models.
        """
        return self._is_tool_call_model_id(model_id)

    def _is_prompt_cache_model_id(self, model_id: str) -> bool:
        """
        Determine if an OpenAI model supports prompt caching.

        Based on OpenAI's prompt caching docs, support is guaranteed for:
        - gpt-4o-2024-08-06
        - gpt-4o-mini-2024-07-18
        - o1-preview
        - o1-mini

        We keep this list explicit and conservative so it matches documented
        behaviour instead of guessing from partial names.
        """
        if not model_id:
            return False

        model_id_lower = str(model_id).lower()
        prompt_cache_models = [
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "o1-preview",
            "o1-mini",
        ]
        return any(p in model_id_lower for p in prompt_cache_models)

    def _is_vision_model_id(self, model_id: str) -> bool:
        """
        Heuristically determine if an OpenAI model supports image/vision input.
        """
        if not model_id:
            return False

        model_id_lower = str(model_id).lower()

        # Known vision-capable GPT models (gpt-4o family, etc.)
        vision_patterns = [
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.5",
            "gpt-5-vision",
            "gpt-4.1-mini",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4.1-mini",
        ]

        if any(p in model_id_lower for p in vision_patterns):
            return True

        # Image-specific models
        if "gpt-image" in model_id_lower:
            return True

        # Kimi-VL is a vision model
        if "kimi-vl" in model_id_lower:
            return True

        return False

    async def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get available OpenAI models with safe fallback.

        This does NOT call the OpenAI list models API; instead it exposes a curated
        subset of models that are known-good for the app, and derives capabilities
        using the same helper methods as `get_models_from_provider`.
        """
        try:
            models_from_env = get_models_from_env("openai")

            # Also include groq, nokia, lmdeploy if present in MODELS,
            # as they are usually OpenAI-compatible and don't have separate providers here.
            for extra_provider in ["groq", "nokia", "lmdeploy"]:
                models_from_env.extend(get_models_from_env(extra_provider))

            if models_from_env:
                default_names = models_from_env
            else:
                default_names = []
            models: List[Dict[str, Any]] = []
            for name in default_names:
                models.append(
                    {
                        "name": name,
                        "provider": "openai",
                        "is_toolCall_available": self._is_tool_call_model_id(name),
                        "is_vision_available": self._is_vision_model_id(name),
                        "is_support_thinking": self._is_support_thinking_model_id(
                            name
                        ),
                        "is_parallel_tool_calls_available": self._is_parallel_tool_calls_model_id(
                            name
                        ),
                        "is_prompt_cache_available": self._is_prompt_cache_model_id(
                            name
                        ),
                        "is_shown": False,
                    }
                )

            return {"openaiModels": models}
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {str(e)}")
            # If something goes wrong, return empty list
            default_names = []
            models: List[Dict[str, Any]] = []
            for name in default_names:
                models.append(
                    {
                        "name": name,
                        "provider": "openai",
                        "is_toolCall_available": self._is_tool_call_model_id(name),
                        "is_vision_available": self._is_vision_model_id(name),
                        "is_support_thinking": self._is_support_thinking_model_id(
                            name
                        ),
                        "is_parallel_tool_calls_available": self._is_parallel_tool_calls_model_id(
                            name
                        ),
                        "is_prompt_cache_available": self._is_prompt_cache_model_id(
                            name
                        ),
                        "is_shown": False,
                    }
                )

            return {"openaiModels": models}

    def is_tool_call_available(self, llm) -> bool:
        """Check if OpenAI model supports tool calling."""
        model_name = getattr(
            llm, "model_name", "") or getattr(llm, "model", "")
        return self._is_tool_call_model_id(model_name)

    def is_vision_available(self, llm=None) -> bool:
        """Check if vision/image input is available."""
        if llm:
            model_name = getattr(
                llm, "model_name", "") or getattr(llm, "model", "")
            return self._is_vision_model_id(model_name)
