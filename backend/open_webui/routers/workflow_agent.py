import asyncio
import logging
from typing import Optional, List, Dict, Any
import httpx
import requests
import os

from fastapi import APIRouter, HTTPException, Request, status, Depends, Header
from open_webui.utils.models_imbrace import LLM_PROVIDER
from open_webui.utils.provider import (
    get_system_providers_config_from_db,
    CONFIG_KEY_TO_PROVIDER_TYPE,
)
from open_webui.env import SRC_LOG_LEVELS, IMBRACE_PLATFORM_HOST
from open_webui.config import WORKFLOW_CONFIG

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])

router = APIRouter()

###########################
# China Whitelist
###########################

CHINA_WHITELIST = [
    "deepseek-r1-distill-qwen-7b",
    "qwen2.5-14b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct-mlx",
    "qwen2.5-coder-32b-instruct",
    "qwen2.5-vl-32b-instruct",
    "qwen3-30b-a3b-dwq",
]

###########################
# Default Models
###########################

DEFAULT_MODELS = []

###########################
# Helper Functions
###########################


async def get_org_ai_settings(org_id: str) -> Dict[str, Any]:
    """Fetch organization AI settings from platform-service.

    Org is identified via the `x-organization-id` header (platform-service
    does not accept org_id in the URL path).
    """
    url = f"{IMBRACE_PLATFORM_HOST}/v1/organizations/ai-settings"
    headers = {"x-organization-id": org_id}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Return the actual response data
            return data
    except httpx.RequestError as e:
        log.error(f"Error fetching org AI settings: {e}")
        # No config available -> treat as "not configured" (allow all), distinct from [] which means "block all"
        return {}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            log.warning(f"Organization {org_id} AI settings not found, returning empty")
            return {}
        else:
            raise


async def get_models_from_all_sources() -> List[Dict[str, Any]]:
    """
    Fetch models from all configured *system* providers stored in the database.

    This replaces the previous ModelService-based implementation and instead:
      1. Reads system provider configs from DB (via build_config_from_system_providers).
      2. Filters providers based on the LLM_PROVIDER env (e.g. \"openai,ollama\").
      3. For each allowed provider, instantiates the LLM provider class and calls
         its `get_available_models()` method.
      4. Flattens all provider-specific model lists into a single list of models.
    """
    try:
        # Build CONFIG-like structure from system providers in DB only
        config = await get_system_providers_config_from_db()
        log.info("[DEBUG-MODEL] Loaded system providers CONFIG keys: %s", list(config.keys()))

        # Import here to avoid potential circular imports at module load time
        from open_webui.llm.agent import get_llm_provider

        all_models: List[Dict[str, Any]] = []

        # CONFIG key -> provider type mapping is centralized in utils.provider
        for config_key, provider_type in CONFIG_KEY_TO_PROVIDER_TYPE.items():
            # Respect LLM_PROVIDER env config (e.g. \"openai,ollama\")
            if "all" not in LLM_PROVIDER and provider_type not in LLM_PROVIDER:
                log.info(
                    "[DEBUG-MODEL] Skipping provider '%s' because it is not in LLM_PROVIDER=%s",
                    provider_type,
                    LLM_PROVIDER,
                )
                continue

            provider_config = config.get(config_key)
            if not provider_config or not isinstance(provider_config, dict):
                # No system provider config for this type in DB; skip
                log.warning(
                    "[DEBUG-MODEL] Skipping provider '%s' (config_key='%s') "
                    "because config from DB is empty or invalid: %r",
                    provider_type,
                    config_key,
                    provider_config,
                )
                continue

            try:
                log.info(
                    "[DEBUG-MODEL] Creating LLM provider '%s' with config_key='%s', "
                    "config_keys=%s",
                    provider_type,
                    config_key,
                    list(provider_config.keys()),
                )

                # Pass full CONFIG-like dict (with all providers) to the factory,
                # same as other code paths using get_llm_provider.
                provider = get_llm_provider(provider_type, config)

                log.info(
                    "[DEBUG-MODEL] Calling get_available_models() for provider '%s'",
                    provider_type,
                )
                provider_models = await provider.get_available_models()

                # Flatten models and keep count per provider for debugging
                provider_models_flat: List[Dict[str, Any]] = []
                if isinstance(provider_models, dict):
                    for value in provider_models.values():
                        if isinstance(value, list):
                            provider_models_flat.extend(
                                [m for m in value if isinstance(m, dict)]
                            )

                log.info(
                    "[DEBUG-MODEL] Provider '%s' returned %d models",
                    provider_type,
                    len(provider_models_flat),
                )

                all_models.extend(provider_models_flat)
            except Exception as e:
                log.error(
                    "[DEBUG-MODEL] Error fetching models from provider '%s': %s",
                    provider_type,
                    e,
                )

        log.info(
            "[DEBUG-MODEL] Retrieved %d total models from all system providers",
            len(all_models),
        )

        return all_models
    except Exception as e:
        log.error(f"Error fetching models from sources: {e}")
        return []


###########################
# Main Endpoint
###########################


@router.get("/models")
async def get_workflow_models(
    request: Request,
    chinaVersion: Optional[str] = None,
    vision: Optional[str] = None,
    filter: Optional[str] = None,
    x_organization_id: str = Header(..., alias="x-organization-id"),
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Get workflow-agent models with filtering based on various parameters.

    Args:
        chinaVersion: Set to "true" to apply China whitelist filtering
        vision: Set to "true" to filter for vision-capable models only
        filter: Set to "false" to disable organization-specific filtering
        x_organization_id: Organization ID for user context (required)
        x_api_key: API key for authorization (optional)

    Returns:
        dict: Response containing success status, message, and data
    """
    try:
        log.info(
            f"Workflow models request - org: {x_organization_id}, china: {chinaVersion}, vision: {vision}, filter: {filter}"
        )

        # Input validation
        if not x_organization_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="x-organization-id header is required",
            )

        # Fetch all available models
        all_models = await get_models_from_all_sources()

        # Add default models if vision != "true"
        models = []
        if vision != "true":
            models.extend(DEFAULT_MODELS)

        # Apply China whitelist filtering
        if chinaVersion == "true":
            china_models = []
            for model in models + all_models:
                if model["name"] in CHINA_WHITELIST:
                    china_models.append(model)
            models = china_models
            log.info(f"Applied China whitelist: {len(models)} models")
        else:
            models.extend(all_models)

        # Apply vision filtering
        if vision == "true":
            models = [m for m in models if m.get("is_vision_available", False)]
            log.info(f"Filtered for vision models: {len(models)} models")

        # Apply organization-based filtering (default behavior unless filter="false")
        if filter != "false":
            try:
                org_settings = await get_org_ai_settings(x_organization_id)
                allowed_models = org_settings.get("allowed_models")

                if allowed_models is not None:
                    # [] means "block all"; non-empty list filters to those names
                    filtered_models = [
                        model for model in models if model["name"] in allowed_models
                    ]
                    models = filtered_models
                    log.info(
                        f"Applied org filtering: {len(models)} models from {len(allowed_models)} allowed"
                    )
                else:
                    log.info("No org-specific filtering applied, returning all models")

            except Exception as e:
                log.error(f"Error fetching org settings: {e}")
                # Continue with all models if org settings fetch fails
        else:
            log.info("Organization filtering disabled")

        # Remove duplicates by name (preserve capabilities from full model list)
        unique_models = {}
        for model in models:
            name = model["name"]
            if name not in unique_models:
                unique_models[name] = model
            else:
                # Merge capabilities if needed (prefer True values)
                existing = unique_models[name]
                unique_models[name] = {
                    "name": name,
                    "is_toolCall_available": existing.get(
                        "is_toolCall_available", False
                    )
                    or model.get("is_toolCall_available", False),
                    "is_vision_available": existing.get("is_vision_available", False)
                    or model.get("is_vision_available", False),
                }

        final_models = list(unique_models.values())
        log.info(f"Final result: {len(final_models)} unique models")

        return {
            "success": True,
            "message": "Models retrieved successfully",
            "data": final_models,
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Unexpected error in get_workflow_models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )
