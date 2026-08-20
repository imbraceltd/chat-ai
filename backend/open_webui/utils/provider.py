from typing import Dict, Any
import uuid
import logging

from open_webui.env import SRC_LOG_LEVELS
from open_webui.repository.provider import ProviderRepository

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MODELS", logging.INFO))

provider_repo = ProviderRepository()

"""
Central mapping between provider types and CONFIG keys.

- CONFIG keys are what `CONFIG` in `llm.agent` uses, e.g. "openApi", "ollama", ...
- Provider types are what we store in DB and use in LLM_PROVIDER/env, e.g. "openai", "ollama", ...

Keep these mappings here and re-use everywhere else to avoid duplication.
"""
CONFIG_KEY_TO_PROVIDER_TYPE: Dict[str, str] = {
    "openApi": "openai",
    "ollama": "ollama",
    "lmstudio": "lmstudio",
    "kimi": "kimi",
    "bedrock": "bedrock",
    "vllm": "vllm",
}

PROVIDER_TYPE_TO_CONFIG_KEY: Dict[str, str] = {
    provider_type: config_key
    for config_key, provider_type in CONFIG_KEY_TO_PROVIDER_TYPE.items()
}


async def get_all(
    organization_id: str = "",
    filter_options: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Service layer to fetch all providers for an organization.
    """
    filter_options = filter_options or {}
    limit = filter_options.get("limit", -1)
    skip = filter_options.get("skip", 0)
    search = filter_options.get("search", "")
    source = filter_options.get("source")

    providers = await provider_repo.get_all_by_organization_id(
        organization_id=organization_id,
        search=search,
        skip=skip,
        limit=limit,
        source=source,
    )

    return providers


async def get_by_id(
    organization_id: str = "",
    provider_id: str = "",
) -> Dict[str, Any] | None:
    """
    Get a single provider by id.
    """
    log.info(
        f"Fetching provider by id: organization_id={organization_id}, provider_id={provider_id}"
    )
    provider = await provider_repo.get_by_provider_id_and_organization_id(
        organization_id=organization_id, provider_id=provider_id
    )
    return provider


async def get_system_provider_by_type(
    provider_type: str,
    organization_id: str = "",
) -> Dict[str, Any] | None:
    """
    Get a system provider configuration by type.
    System providers are global configurations with source="system".
    
    Args:
        provider_type: Provider type (e.g., "openai", "ollama", "bedrock", etc.)
        organization_id: Organization ID (default: "" for global)
    
    Returns:
        Provider configuration dict or None if not found
    """
    log.info(
        f"Fetching system provider by type: provider_type={provider_type}, organization_id={organization_id}"
    )
    provider = await provider_repo.get_system_provider_by_type(
        provider_type=provider_type,
        organization_id=organization_id,
    )
    return provider


async def get_system_providers_config_from_db(
    organization_id: str = "",
) -> Dict[str, Any]:
    """
    Build CONFIG-like structure from system providers in DB only.
    Only includes providers that exist in DB.

    Args:
        organization_id: Organization ID (default: "" for global)

    Returns:
        CONFIG dict structure with provider configs from DB only
    """
    config: Dict[str, Any] = {}

    # Get config from system providers in DB only, using central mapping
    for provider_type, config_key in PROVIDER_TYPE_TO_CONFIG_KEY.items():
        system_provider = await get_system_provider_by_type(
            provider_type=provider_type,
            organization_id=organization_id,
        )
        
        if system_provider and system_provider.get("config"):
            # Only use config from DB
            config[config_key] = system_provider["config"]
            log.debug(f"Using system provider config from DB for {provider_type}")
    
    return config


async def create(
    organization_id: str = "",
    provider_details: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """
    Create a new provider configuration for an organization.

    Expected provider_details (after validation):
    {
        "name": str,
        "type": str,
        "config": dict,
        "metadata": dict | None
    }
    """
    provider_details = provider_details or {}

    log.info(
        f"Creating provider for organization_id={organization_id} with details={provider_details}"
    )

    # Generate stable ids similar to Assistant service
    provider_id = str(uuid.uuid4())
    provider_details["id"] = provider_id
    provider_details["provider_id"] = provider_id

    created_provider = await provider_repo.create(
        organization_id=organization_id, provider_details=provider_details
    )

    if created_provider:
        # Strip internal fields before returning
        for key in ["_id", "updated_at", "deleted_at"]:
            if key in created_provider:
                created_provider.pop(key, None)

    return created_provider


async def remove(
    organization_id: str = "",
    provider_id: str = "",
) -> Dict[str, Any]:
    """
    Soft delete a provider configuration.
    """
    log.info(
        f"Removing provider: organization_id={organization_id}, provider_id={provider_id}"
    )

    # We don't need organization_id in the repository call for now,
    # but we keep it at service level for future checks / auth.
    await provider_repo.remove(provider_id)
    return {"id": provider_id, "object": "provider.deleted", "deleted": True}


async def update(
    organization_id: str = "",
    provider_id: str = "",
    provider_details: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """
    Update an existing provider configuration for an organization.

    The validation of `provider_details` (required fields, types, etc.)
    is handled at the router/helper level, similar to the assistants flow.
    """
    provider_details = provider_details or {}

    log.info(
        f"Updating provider for organization_id={organization_id}, "
        f"provider_id={provider_id} with details={provider_details}"
    )

    # Fetch current provider to ensure it exists and to preserve immutable fields.
    current_provider = await provider_repo.get_by_provider_id_and_organization_id(
        organization_id=organization_id,
        provider_id=provider_id,
    )

    if not current_provider:
        log.warning(
            f"Provider not found for update: organization_id={organization_id}, "
            f"provider_id={provider_id}"
        )
        return None

    # Capture original values so we can detect real config changes.
    original_type = (current_provider.get("type") or "").lower()
    original_config = current_provider.get("config")

    # Only allow updating logical fields; keep identifiers and soft-delete flags.
    # D75a: allow updating new logical fields as well.
    updatable_fields = [
        "name",
        "type",
        "config",
        "metadata",
        "source",
        "is_shown",
        "models",
    ]
    for field in updatable_fields:
        if field in provider_details:
            current_provider[field] = provider_details[field]

    # If provider config/type is updated, refresh models from the provider using the
    # new runtime config (same behaviour as create_provider validation), so the
    # response includes the latest model list.
    #
    # Preserve visibility flags (is_shown) from either:
    # - models provided in this update request (if any), else
    # - models already stored on the provider.
    # Many clients always include `config` in update payloads; avoid refetching
    # models unless the config actually changed.
    should_refresh_models = False
    if "config" in provider_details or "type" in provider_details:
        new_type = (current_provider.get("type") or "").lower()
        new_config = current_provider.get("config")
        should_refresh_models = (new_type != original_type) or (new_config != original_config)
    if should_refresh_models:
        try:
            from open_webui.llm.agent import get_llm_provider
            from open_webui.routers.helper import InvalidProviderConfigError

            provider_type = (current_provider.get("type") or "").lower()
            provider_config = current_provider.get("config") or {}

            provider = get_llm_provider(provider_type, provider_config)
            provider_models = await provider.get_models_from_provider()

            # Flatten dict like {"googleModels":[...]} into a single list
            refreshed_models: list[dict] = []
            if isinstance(provider_models, dict):
                for value in provider_models.values():
                    if isinstance(value, list):
                        for m in value:
                            if isinstance(m, dict):
                                refreshed_models.append(dict(m))

            # Build visibility map from request models (preferred) or existing provider models
            visibility_source = provider_details.get("models")
            if not isinstance(visibility_source, list):
                visibility_source = current_provider.get("models", [])
            visibility_map: dict[str, bool] = {}
            for m in visibility_source:
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                if isinstance(name, str) and "is_shown" in m:
                    visibility_map[name] = bool(m.get("is_shown"))

            for m in refreshed_models:
                name = m.get("name")
                if isinstance(name, str) and name in visibility_map:
                    m["is_shown"] = visibility_map[name]
                else:
                    # Default hidden unless explicitly enabled
                    m.setdefault("is_shown", False)

            # Preserve manually added models (is_add_manually=True)
            existing_models = current_provider.get("models") or []
            for m in existing_models:
                if isinstance(m, dict) and m.get("is_add_manually"):
                    refreshed_models.append(m)

            current_provider["models"] = refreshed_models
        except InvalidProviderConfigError:
            raise
        except Exception as e:
            # Fail update if the new runtime config can't connect, consistent with create.
            from open_webui.routers.helper import InvalidProviderConfigError

            raise InvalidProviderConfigError(
                f"Unable to connect to LLM provider '{current_provider.get('type')}' with the given config: {e}"
            ) from e

    updated_provider = await provider_repo.update(provider_id, current_provider)

    if updated_provider:
        for key in ["_id", "updated_at", "deleted_at"]:
            if key in updated_provider:
                updated_provider.pop(key, None)

    return updated_provider


async def refresh_models(
    organization_id: str = "",
    provider_id: str = "",
) -> Dict[str, Any] | None:
    """
    Fetch the latest model list from the provider and merge with existing models.

    - Models already saved in the provider config are preserved as-is (keep is_shown,
      is_toolCall_available, etc.).
    - Only models that are NEW (not yet in the saved list) are appended with
      is_shown=False by default.

    Returns the updated provider dict, or None if the provider was not found.
    """
    current_provider = await provider_repo.get_by_provider_id_and_organization_id(
        organization_id=organization_id,
        provider_id=provider_id,
    )

    if not current_provider:
        log.warning(
            "Provider not found for refresh_models: organization_id=%s, provider_id=%s",
            organization_id,
            provider_id,
        )
        return None

    provider_type = (current_provider.get("type") or "").lower()
    provider_config = current_provider.get("config") or {}

    # Fetch latest models from the remote provider
    from open_webui.llm.agent import get_llm_provider

    provider = get_llm_provider(provider_type, provider_config)
    provider_models = await provider.get_models_from_provider()

    # Flatten fetched models into a single list
    fetched_models: list[dict] = []
    if isinstance(provider_models, dict):
        for value in provider_models.values():
            if isinstance(value, list):
                for m in value:
                    if isinstance(m, dict):
                        fetched_models.append(dict(m))

    # Build a lookup of existing models by name to preserve their settings
    existing_models: list[dict] = current_provider.get("models") or []
    existing_by_name: dict[str, dict] = {
        m.get("name"): m
        for m in existing_models
        if isinstance(m, dict) and m.get("name")
    }

    # Result = exactly what the provider returns.
    # If a model already existed in DB, preserve its saved settings;
    # otherwise use the fetched model with defaults.
    merged_models: list[dict] = []
    new_count = 0
    for m in fetched_models:
        name = m.get("name")
        if isinstance(name, str) and name in existing_by_name:
            # Keep the saved version (preserves is_shown, is_toolCall_available, etc.)
            merged_models.append(existing_by_name[name])
        else:
            m.setdefault("is_shown", False)
            m.setdefault("is_toolCall_available", True)
            merged_models.append(m)
            new_count += 1

    # Preserve manually added models (is_add_manually=True)
    for m in existing_models:
        if isinstance(m, dict) and m.get("is_add_manually"):
            merged_models.append(m)

    current_provider["models"] = merged_models

    updated_provider = await provider_repo.update(provider_id, current_provider)

    if updated_provider:
        for key in ["_id", "updated_at", "deleted_at"]:
            updated_provider.pop(key, None)

    log.info(
        "refresh_models completed for provider_id=%s: existing=%d, new=%d, total=%d",
        provider_id,
        len(existing_models),
        new_count,
        len(merged_models),
    )

    return updated_provider


async def sync_system_providers_from_env(
    organization_id: str = "",
) -> Dict[str, Any]:
    """
    Sync system providers from environment variables based on CONFIG in agent.py.
    
    For each provider type in CONFIG, create or update a system provider with whatever config is available.
    No default value checks - uses whatever is in CONFIG.
    
    Args:
        organization_id: Organization ID for system providers (default: "" for global)
    
    Returns:
        Dict with results: {"created": [...], "updated": [...], "skipped": [...]}
    """
    try:
        # Import CONFIG from agent.py
        from open_webui.llm.agent import CONFIG
        
        results = {
            "created": [],
            "updated": [],
            "skipped": [],
        }
        
        # Only sync providers listed in LLM_PROVIDER env var
        from open_webui.utils.models_imbrace import LLM_PROVIDER
        enabled_providers = set(LLM_PROVIDER)

        # Map CONFIG keys to provider types using central mapping
        for config_key, provider_type in CONFIG_KEY_TO_PROVIDER_TYPE.items():
            if config_key not in CONFIG:
                continue

            # Skip providers not in LLM_PROVIDER
            if "all" not in enabled_providers and provider_type not in enabled_providers:
                log.info(f"Skipping {provider_type} - not in LLM_PROVIDER={LLM_PROVIDER}")
                results["skipped"].append(provider_type)
                continue
                
            provider_config = CONFIG.get(config_key, {})
            
            # Skip only if config is empty or None
            if not provider_config or not isinstance(provider_config, dict):
                log.info(f"Skipping {provider_type} - config is empty or invalid")
                results["skipped"].append(provider_type)
                continue
            
            # Use CONFIG directly - no need to rebuild config dict
            # CONFIG already has all the fields we need, just use it as-is
            provider_config_dict = provider_config.copy() if provider_config else {}
            
            # Check if system provider already exists
            existing_provider = await provider_repo.get_system_provider_by_type(
                provider_type=provider_type,
                organization_id=organization_id,
            )
            
            # Build provider details
            provider_name = f"System {provider_type.upper()} Provider"
            provider_details = {
                "name": provider_name,
                "type": provider_type,
                "config": provider_config_dict,
                "source": "system",
                "is_shown": True,
            }
            
            if existing_provider:
                # Update existing provider
                log.info(f"Updating system provider for {provider_type}")
                provider_id = existing_provider.get("provider_id")
                
                # Get available models using get_llm_provider and get_available_models
                try:
                    from open_webui.llm.agent import get_llm_provider
                    
                    # Use CONFIG from env directly (since we're syncing from env)
                    provider = get_llm_provider(provider_type, CONFIG)
                    provider_models = await provider.get_available_models()
                    
                    # Populate models
                    models_list = []
                    if isinstance(provider_models, dict):
                        for value in provider_models.values():
                            if isinstance(value, list):
                                models_list.extend(value)
                    provider_details["models"] = models_list
                except Exception as e:
                    log.warning(f"Failed to get models from provider {provider_type}: {e}")
                    provider_details["models"] = existing_provider.get("models", [])
                
                updated = await update(
                    organization_id=organization_id,
                    provider_id=provider_id,
                    provider_details=provider_details,
                )
                if updated:
                    results["updated"].append(provider_type)
            else:
                # Create new provider
                log.info(f"Creating system provider for {provider_type}")
                
                # Get available models using get_llm_provider and get_available_models
                try:
                    from open_webui.llm.agent import get_llm_provider
                    
                    # Use CONFIG from env directly (since we're syncing from env)
                    provider = get_llm_provider(provider_type, CONFIG)
                    provider_models = await provider.get_available_models()
                    
                    # Populate models
                    models_list = []
                    if isinstance(provider_models, dict):
                        for value in provider_models.values():
                            if isinstance(value, list):
                                models_list.extend(value)
                    provider_details["models"] = models_list
                except Exception as e:
                    log.warning(f"Failed to get models from provider {provider_type}: {e}")
                    provider_details["models"] = []
                
                # Validate input
                try:
                    from open_webui.routers.helper import validate_create_provider_input
                    validated_details = validate_create_provider_input(provider_details)
                    created = await create(
                        organization_id=organization_id,
                        provider_details=validated_details,
                    )
                    if created:
                        results["created"].append(provider_type)
                except Exception as e:
                    log.error(f"Failed to create system provider for {provider_type}: {e}")
                    results["skipped"].append(provider_type)
        
        log.info(f"Sync system providers completed: {results}")
        return results
        
    except Exception as e:
        log.error(f"Failed to sync system providers from env: {e}", exc_info=True)
        raise


