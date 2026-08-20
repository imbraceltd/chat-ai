import logging
import os
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def get_models_from_env(provider_name: str) -> List[str]:
    """
    Parse the MODELS environment variable and return model names for a specific provider.
    The MODELS env var is expected to be a comma-separated list of 'provider/model_id'.
    """
    models_env = os.environ.get("MODELS", "")
    if not models_env:
        return []

    logger.info(f"Parsing MODELS env var for provider '{provider_name}'")
    
    provider_models = []
    # Split by comma and strip whitespace
    model_entries = [m.strip() for m in models_env.split(",")]
    
    for entry in model_entries:
        if "/" in entry:
            p, m = entry.split("/", 1)
            if p.strip().lower() == provider_name.lower():
                provider_models.append(m.strip())
        else:
            # If no provider prefix, and provider_name matches some default logic or is specifically requested?
            # For now, we expect provider/model_id format.
            pass
            
    return provider_models


def detect_provider(
    model_id: str, models: Dict[str, List[Dict]], default_provider: str
) -> str:
    """
    Detect the provider based on model_id by searching through all available models.

    Args:
        model_id: The model ID to search for
        models: Dictionary containing models from all providers (from get_all_available_models)
        default_provider: Fallback provider if model is not found

    Returns:
        Provider name ("openai", "ollama", "lmstudio") or default_provider
    """
    if not model_id or not model_id.strip():
        logger.warning("Empty model_id provided, using default provider")
        return default_provider

    logger.info(f"Detecting provider for model_id: {model_id}")

    # Search through all provider model lists to find the requested model
    for provider_key, model_list in models.items():
        if not isinstance(model_list, list):
            continue

        for model_info in model_list:
            if not isinstance(model_info, dict):
                continue

            # Check if the model name matches
            model_name = model_info.get("name", "")
            logger.debug(
                f"Checking model '{model_name}' under provider '{provider_key}'"
            )

            # Check for exact match first
            if model_name == model_id:
                detected_provider = model_info.get("provider", "")
                if detected_provider:
                    logger.info(
                        f"Detected provider '{detected_provider}' for model '{model_id}'"
                    )
                    return detected_provider

            # If model_id has imbrace/ prefix and model name matches without prefix, treat as Ollama
            elif model_id.startswith("imbrace/") and model_name == model_id[8:]:
                logger.info(
                    f"Detected provider 'ollama' for imbrace-prefixed model '{model_id}' (matches '{model_name}')"
                )
                return "ollama"

            # If not found, check for imbrace/ prefix to identify Ollama models
            elif model_id.startswith("imbrace/"):
                logger.debug(
                    f"Model '{model_id}' has imbrace prefix but doesn't match '{model_name}' in '{provider_key}', continuing search"
                )

        # If we didn't find exact match, but the provider key looks like ollama, assume it's an Ollama model
        if model_id.startswith("imbrace/") and "ollama" in provider_key.lower():
            logger.info(
                f"Inferred provider 'ollama' for imbrace-prefixed model '{model_id}' from provider key '{provider_key}'"
            )
            return "ollama"

    # Heuristic: Bedrock model IDs follow a "vendor.model-name" pattern
    # e.g., qwen.qwen3-vl-235b-a22b, anthropic.claude-3-sonnet, meta.llama3-70b-instruct
    bedrock_vendor_prefixes = [
        "qwen.", "anthropic.", "meta.", "amazon.", "cohere.", "ai21.",
        "mistral.", "stability.", "deepseek.", "us.anthropic.", "us.meta.",
        "us.amazon.", "eu.anthropic.", "eu.meta.", "ap.anthropic.", "ap.meta.",
        "minimax.",
    ]
    if any(model_id.lower().startswith(prefix) for prefix in bedrock_vendor_prefixes):
        logger.info(
            f"Inferred provider 'bedrock' for model '{model_id}' based on naming pattern"
        )
        return "bedrock"

    # If model not found in any provider, log warning and return default
    logger.warning(f"Model '{model_id}' not found in any provider.")
    return default_provider
