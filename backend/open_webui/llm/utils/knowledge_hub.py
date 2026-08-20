import json
import logging
from typing import Dict, Any, Optional, Tuple
import aiohttp

logger = logging.getLogger(__name__)

# Self-hosted, OpenAI-compatible providers whose models may be reasoning models
# (Qwen3, etc.). For these we disable the model's "thinking" mode for tag
# generation. Hosted APIs (openai, bedrock, google) reject the vLLM-specific
# chat_template_kwargs, so they are intentionally excluded.
_NO_THINK_PROVIDERS = {"vllm", "lmstudio", "kimi"}


def _resolve_from_config(config: Dict[str, Any], source: str) -> Tuple[Any, Optional[str]]:
    """Pick a provider from a CONFIG-like dict and build its chat model.

    Provider-agnostic: tries the configured default provider (PROVIDER) first,
    then the first other enabled provider (LLM_PROVIDER) that has a non-empty
    config. The model is resolved explicitly from each provider's own config
    (different providers store it under different keys, and not all of them fall
    back to a configured model when given an empty string).

    Returns (llm, provider_type) or (None, None) if nothing usable is found.
    """
    from open_webui.utils.provider import PROVIDER_TYPE_TO_CONFIG_KEY
    from open_webui.utils.models_imbrace import LLM_PROVIDER
    from open_webui.llm.agent import get_llm_provider, PROVIDER

    if not config or not isinstance(config, dict):
        return None, None

    # Candidate provider types: configured default first, then the rest.
    ordered_types = [PROVIDER] + [
        pt for pt in PROVIDER_TYPE_TO_CONFIG_KEY.keys() if pt != PROVIDER
    ]

    for provider_type in ordered_types:
        config_key = PROVIDER_TYPE_TO_CONFIG_KEY.get(provider_type)
        if not config_key:
            continue

        # Respect LLM_PROVIDER env (e.g. "openai,ollama" or "all").
        if "all" not in LLM_PROVIDER and provider_type not in LLM_PROVIDER:
            continue

        provider_config = config.get(config_key)
        if not provider_config or not isinstance(provider_config, dict):
            continue

        # Resolve the model name from the provider's own config.
        model_name = ""
        for key in ("api_type", "model", "retriever_model"):
            value = provider_config.get(key)
            if isinstance(value, str) and value.strip():
                model_name = value
                break

        try:
            provider = get_llm_provider(provider_type, config)
            llm = provider.create_llm(
                model=model_name,
                available_models={},
                streaming=False,
                temperature=0.3,
            )

            # Self-hosted OpenAI-compatible models are often reasoning models
            # (e.g. Qwen3 on vLLM) that emit a long "thinking" trace before the
            # answer. For tag generation that turns a 1s call into a 1-3 min one
            # and causes upstream gateway timeouts (504), so disable thinking.
            if provider_type in _NO_THINK_PROVIDERS:
                try:
                    llm = llm.bind(
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    )
                except Exception as bind_err:
                    logger.warning(
                        f"Could not disable thinking for provider "
                        f"'{provider_type}': {bind_err}"
                    )

            logger.info(
                f"Resolved LLM provider for tag generation: {provider_type} "
                f"(model={model_name or 'provider default'}, source={source})"
            )
            return llm, provider_type
        except Exception as e:
            logger.warning(
                f"Failed to create LLM for provider '{provider_type}' "
                f"(source={source}): {e}"
            )
            continue

    return None, None


async def _resolve_llm(organization_id: str = "") -> Tuple[Any, Optional[str]]:
    """Resolve a LangChain chat model for tag generation.

    Tries the env config first (``CONFIG``, built from environment variables at
    startup), and only falls back to the DB system providers when no usable
    provider is configured in env. The env path always reflects the current
    ``.env`` (a restart is required for env changes to take effect) and does not
    depend on the DB sync, which can go stale when an endpoint is unreachable.

    Returns (llm, provider_type) or (None, None) if nothing is configured.
    """
    from open_webui.llm.agent import CONFIG

    # 1) Env config first.
    llm, provider_type = _resolve_from_config(CONFIG, source="env")
    if llm is not None:
        return llm, provider_type

    # 2) Fall back to DB system providers.
    from open_webui.utils.provider import get_system_providers_config_from_db

    db_config = await get_system_providers_config_from_db(organization_id)
    if not db_config and organization_id:
        # System providers are typically stored globally; fall back to global.
        db_config = await get_system_providers_config_from_db("")

    llm, provider_type = _resolve_from_config(db_config, source="db")
    if llm is not None:
        return llm, provider_type

    logger.error("No enabled+configured provider available for tag generation")
    return None, None


class KnowledgeHubService:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        """Initialize aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Close aiohttp session."""
        await self.session.close()

    async def generate_tags(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate tags based on folder name and description using the configured provider."""
        try:
            logger.info(f"Generating tags with context: {context}")

            # Get folder name and description from context
            folder_name = context.get("folder_name", "")
            description = context.get("description", "")
            organization_id = context.get("organization_id", "")

            if not folder_name or not description:
                logger.error("Missing folder_name or description in context")
                return None

            # Resolve an LLM through the model-provider abstraction
            llm, provider_type = await _resolve_llm(organization_id)
            if llm is None:
                logger.error("No LLM provider available for tag generation")
                return None

            # Prepare the prompt for tag generation
            prompt = f"""
            Based on the following folder name and description, generate exactly 5 relevant tags that would help categorize and organize this content.

            Folder Name: {folder_name}
            Description: {description}

            Please return only a JSON array of 5 tag strings, like: ["tag1", "tag2", "tag3", "tag4", "tag5"]
            """

            # Make the LLM request (non-streaming)
            from langchain_core.messages import HumanMessage

            response = await llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content

            # Clean the content by removing markdown code blocks if present
            cleaned_content = content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content[7:]  # Remove ```json
            if cleaned_content.startswith("```"):
                cleaned_content = cleaned_content[3:]  # Remove ```
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]  # Remove trailing ```
            cleaned_content = cleaned_content.strip()

            # Parse the JSON response
            try:
                tags = json.loads(cleaned_content)
                if not isinstance(tags, list) or len(tags) != 5:
                    raise ValueError("Invalid tag format")
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response: {content}")
                # Fallback: try to extract tags from text
                import re

                tag_matches = re.findall(r'"([^"]*)"', cleaned_content)
                if len(tag_matches) >= 5:
                    tags = tag_matches[:5]
                else:
                    # Last resort fallback
                    tags = ["general", "content", "folder", "organization", "tags"]

            # Best-effort resolved model name for the response
            resolved_model = getattr(llm, "model_name", None) or getattr(
                llm, "model", None
            )

            return {
                "tags": tags[:5],  # Ensure max 5 tags
                "folderName": folder_name,
                "description": description,
                "model": resolved_model or provider_type,
            }

        except Exception as error:
            logger.error(f"Error generating tags: {error}")
            return None


# Factory function
def create_knowledge_hub_service() -> KnowledgeHubService:
    return KnowledgeHubService()


# Module-level async function for convenience
async def generate_tags(context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    async with KnowledgeHubService() as service:
        return await service.generate_tags(context)
