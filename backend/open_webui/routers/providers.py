from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Optional, Dict, Any
import logging

from open_webui.constants import ERROR_MESSAGES
from open_webui.utils.auth import auth_imbrace
from open_webui.utils import provider as provider_service
from open_webui.routers.helper import (
    validate_create_provider_input,
    InvalidProviderInputError,
    InvalidProviderConfigError,
    validate_provider_runtime_config,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("PROVIDERS", logging.INFO))

router = APIRouter()


@router.get("")
async def get_all_providers(
    userContext=Depends(auth_imbrace),
    skip: int = 0,
    limit: int = -1,
    search: Optional[str] = None,
):
    """
    List all LLM providers configured for the current organization.
    Only providers with source="custom" are returned.
    """
    try:
        filter_options: Dict[str, Any] = {
            "skip": skip,
            "limit": limit,
            "search": search,
            # Always restrict to custom providers; no need for caller to pass source
            "source": "custom",
        }
        log.info("get_all_providers called with filters: %s", filter_options)

        providers_response = await provider_service.get_all(
            organization_id=userContext["organization_id"],
            filter_options=filter_options,
        )

        # Align response shape with assistants endpoint for consistency
        return providers_response.get("data", providers_response.get("documents", []))
    except Exception as e:
        log.error(f"Failed to get providers: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.get("/{provider_id}")
async def get_provider_by_id(
    provider_id: str,
    userContext=Depends(auth_imbrace),
):
    """
    Get a single provider configuration by id.
    """
    try:
        provider = await provider_service.get_by_id(
            organization_id=userContext["organization_id"],
            provider_id=provider_id,
        )
        if not provider:
            raise HTTPException(status_code=404, detail="Provider not found")
        return provider
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get provider: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.post("")
async def create_provider(request: Request, userContext=Depends(auth_imbrace)):
    """
    Create a new LLM provider configuration for the current organization.

    Payload example:
    {
        "name": "My LMStudio Provider",
        "type": "lmstudio",
        "config": {
            "host": "http://localhost:8000",
            "api_type": "llama3.2",
            "embedding_model": "llama3.2"
        },
        "metadata": {
            "description": "Local LMStudio instance"
        }
    }
    """
    try:
        body = await request.json()
        log.info(f"create_provider called with body: {body}")

        provider_details = validate_create_provider_input(body)

        # DEBUG: log provider config high level (avoid secrets)
        log.info(
            "[DEBUG-LLM] Creating provider; type=%s, config_keys=%s",
            provider_details.get("type"),
            list(provider_details.get("config", {}).keys()),
        )

        # Validate provider config by actually trying to fetch available models
        # from the underlying LLM provider implementation. If this fails,
        # we treat the configuration as invalid.
        provider_models = await validate_provider_runtime_config(
            provider_type=provider_details["type"],
            config=provider_details["config"],
        )
        log.info(
            "[DEBUG-LLM] Provider config validated successfully for type=%s",
            provider_details.get("type"),
        )

        # Populate the provider "models" field with the models discovered at runtime.
        # validate_provider_runtime_config returns a dict like:
        #   {"openaiModels": [...]} | {"ollamaModels": [...]} | ...
        # We flatten all list values into a single list of model descriptors,
        # and set a default visibility flag (is_shown=False) for each discovered model.
        # If the client provided explicit visibility for a model, that will be respected
        # on subsequent updates (PATCH/PUT) where we no longer refetch models.
        models_list = []
        if isinstance(provider_models, dict):
            for value in provider_models.values():
                if isinstance(value, list):
                    for model in value:
                        if not isinstance(model, dict):
                            continue
                        merged_model = dict(model)
                        # Default new models to hidden in UI unless changed later
                        if "is_shown" not in merged_model:
                            merged_model["is_shown"] = False
                        models_list.append(merged_model)
        provider_details["models"] = models_list

        created_provider = await provider_service.create(
            organization_id=userContext["organization_id"],
            provider_details=provider_details,
        )
        return created_provider
    except (InvalidProviderInputError, InvalidProviderConfigError) as e:
        log.info("[DEBUG-LLM] Provider create failed validation: %s", e)
        log.error(f"Invalid provider input/config: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        log.error(f"Failed to create provider: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.post("/{provider_id}/models/refresh")
async def refresh_provider_models(
    provider_id: str,
    userContext=Depends(auth_imbrace),
):
    """
    Fetch the latest model list from the provider and merge with existing models.

    - Models that already exist in the saved config are kept as-is.
    - Only new models (by name) are appended with is_shown=False.
    """
    try:
        log.info(
            "refresh_provider_models called: provider_id=%s, organization_id=%s",
            provider_id,
            userContext["organization_id"],
        )

        updated_provider = await provider_service.refresh_models(
            organization_id=userContext["organization_id"],
            provider_id=provider_id,
        )

        if not updated_provider:
            raise HTTPException(status_code=404, detail="Provider not found")

        return updated_provider
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to refresh provider models: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.put("/{provider_id}")
async def update_provider(
    provider_id: str,
    request: Request,
    userContext=Depends(auth_imbrace),
):
    """
    Update an existing LLM provider configuration for the current organization.
    """
    try:
        body = await request.json()
        if body is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body cannot be empty",
            )

        log.info(
            f"update_provider called with provider_id={provider_id}, body: {body}"
        )

        provider_details = validate_create_provider_input(body)

        # DEBUG: log provider config high level (avoid secrets)
        log.info(
            "[DEBUG-LLM] Updating provider %s; type=%s, config_keys=%s",
            provider_id,
            provider_details.get("type"),
            list(provider_details.get("config", {}).keys()),
        )

        updated_provider = await provider_service.update(
            organization_id=userContext["organization_id"],
            provider_id=provider_id,
            provider_details=provider_details,
        )

        if not updated_provider:
            raise HTTPException(status_code=404, detail="Provider not found")

        return updated_provider
    except (InvalidProviderInputError, InvalidProviderConfigError) as e:
        log.info(
            "[DEBUG-LLM] Provider update failed validation; id=%s, error=%s",
            provider_id,
            e,
        )
        log.error(f"Invalid provider input/config: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        # Re-raise explicit HTTP errors (e.g. 404, 400) without wrapping.
        raise
    except Exception as e:
        log.error(f"Failed to update provider: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.delete("/{provider_id}")
async def remove_provider(
    provider_id: str,
    userContext=Depends(auth_imbrace),
):
    """
    Soft delete a provider configuration.
    """
    try:
        if not provider_id:
            raise HTTPException(status_code=400, detail="Provider ID is required")

        log.info(
            f"remove_provider called with provider_id={provider_id} "
            f"for organization_id={userContext['organization_id']}"
        )

        # Optionally you can check ownership here by loading the provider first.
        deleted = await provider_service.remove(
            organization_id=userContext["organization_id"],
            provider_id=provider_id,
        )
        return deleted
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to remove provider: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


