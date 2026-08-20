"""Shared factory for auxiliary "task" LLMs (image summary, table description,
PDF summary, ...).

These features used to hardcode `ChatOpenAI(model="gpt-4o...")`, which broke on
deployments without an OpenAI key (bedrock / vLLM self-host). This helper routes
them through the same OpenAI->provider migration as `echart.py`:

  - Provider: `<FEATURE>_PROVIDER` env overrides; otherwise LLM_DEFAULT_PROVIDER.
  - Model:    `<FEATURE>_MODEL` env overrides; on the openai path it defaults to
              `default_openai_model`, on other providers an empty model means
              "use the provider's own default".

Example: build_task_llm("IMAGE_SUMMARY", default_openai_model="gpt-4o-mini")
reads IMAGE_SUMMARY_PROVIDER / IMAGE_SUMMARY_MODEL.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_task_llm(
    feature: str,
    *,
    default_openai_model: str,
    temperature: float = 0.2,
    streaming: bool = False,
    api_key: Optional[str] = None,
    **openai_kwargs,
):
    """Build a chat LLM for an auxiliary task, following LLM_DEFAULT_PROVIDER.

    Args:
        feature: Uppercase env prefix, e.g. "IMAGE_SUMMARY" -> reads
            IMAGE_SUMMARY_PROVIDER / IMAGE_SUMMARY_MODEL.
        default_openai_model: Model used on the legacy openai path when
            <FEATURE>_MODEL is not set.
        temperature: Sampling temperature.
        streaming: Whether to stream (these tasks are usually one-shot -> False).
        api_key: Per-request OpenAI key (openai path only; ignored by other
            providers which authenticate via their own config/creds).
        **openai_kwargs: Extra kwargs applied ONLY on the openai path
            (e.g. max_tokens, timeout, max_retries).
    """
    # Lazy import to avoid a circular import with llm.agent at module load.
    from open_webui.llm.agent import get_llm_provider, CONFIG, PROVIDER

    provider_name = (
        os.getenv(f"{feature}_PROVIDER") or PROVIDER or "openai"
    ).strip().lower()
    model_env = os.getenv(f"{feature}_MODEL", "")

    if provider_name == "openai":
        # Legacy OpenAI path (honours a per-request api_key + the OpenAI proxy).
        from langchain_openai import ChatOpenAI
        from open_webui.config import OPENAI_API_KEY, OPENAI_API_BASE_URLS

        kwargs = {
            "model": model_env or default_openai_model,
            "api_key": api_key or OPENAI_API_KEY,
            "temperature": temperature,
        }
        base_url = OPENAI_API_BASE_URLS.value[0] if OPENAI_API_BASE_URLS.value else None
        if base_url:
            kwargs["base_url"] = base_url
        if streaming:
            kwargs["streaming"] = True
        kwargs.update(openai_kwargs)
        return ChatOpenAI(**kwargs)

    # Bedrock / vLLM / others: build via the provider factory. When
    # <FEATURE>_MODEL is unset, fall back to the app-wide LLM_DEFAULT_MODEL so
    # task LLMs use the same model as the main chat (instead of each provider's
    # own default like BEDROCK_MODEL / VLLM_RETRIEVER_MODEL). openai_kwargs are
    # intentionally not forwarded (each provider's create_llm applies its own
    # sane defaults).
    from open_webui.config import LLM_CONFIG

    model = model_env or LLM_CONFIG.get("defaultModel", "") or ""
    logger.info(
        "Building %s LLM via provider '%s' (model=%r)",
        feature,
        provider_name,
        model or "<provider default>",
    )
    provider = get_llm_provider(provider_name, CONFIG)
    return provider.create_llm(model, {}, streaming=streaming, temperature=temperature)
