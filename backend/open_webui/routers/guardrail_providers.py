import logging
import uuid
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from open_webui.utils.auth import auth_imbrace
from open_webui.repository.guardrail_provider import GuardrailProviderRepository
from open_webui.llm.utils.custom_guardrail import create_guardrail_service

log = logging.getLogger(__name__)

router = APIRouter()
repo = GuardrailProviderRepository()


# --- Supported provider types ---
GUARDRAIL_PROVIDER_TYPES = ["openai", "nim-nemo", "llamaguard", "ollama"]


# --- Pydantic models ---

class GuardrailProviderConfig(BaseModel):
    # Common
    api_key: str = ""
    timeout: int = 30

    # OpenAI Moderation
    model: Optional[str] = None  # e.g. "omni-moderation-latest", "text-moderation-latest"

    # NIM NeMo / LlamaGuard (vLLM) / Custom HTTP
    base_url: Optional[str] = None  # endpoint URL
    check_endpoint: Optional[str] = None  # for custom HTTP, default "/check"

    # LlamaGuard
    model_name: Optional[str] = None  # e.g. "meta-llama/Llama-Guard-3-8B"
    max_tokens: Optional[int] = None  # default 512


class GuardrailProviderCreateRequest(BaseModel):
    name: str
    type: str = "custom"  # openai | nim-nemo | llamaguard | custom
    config: GuardrailProviderConfig


class GuardrailProviderUpdateRequest(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    config: Optional[GuardrailProviderConfig] = None
    is_shown: Optional[bool] = None
    # Per-model metadata, keyed by model name. e.g. {"llama-guard3": {"image_url": "https://..."}}
    models_metadata: Optional[Dict[str, Any]] = None


class GuardrailProviderResponse(BaseModel):
    guardrail_provider_id: str
    organization_id: Optional[str] = None
    name: str
    type: str = "custom"
    config: Optional[Dict[str, Any]] = None
    # Per-model metadata, keyed by model name (e.g. custom icon image_url)
    models_metadata: Optional[Dict[str, Any]] = None
    source: str = "custom"
    is_shown: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# --- Endpoints ---

@router.get("", response_model=List[GuardrailProviderResponse])
async def list_guardrail_providers(
    ctx=Depends(auth_imbrace),
):
    """List all custom guardrail providers for the organization."""
    try:
        org_id = ctx["organization_id"]
        providers = await repo.get_all_by_organization_id(org_id)

        results = []
        for p in providers:
            # Convert datetime to string
            for field in ("created_at", "updated_at"):
                if hasattr(p.get(field), "isoformat"):
                    p[field] = p[field].isoformat()

            results.append(p)

        return results
    except Exception as e:
        log.error(f"Error listing guardrail providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{guardrail_provider_id}", response_model=GuardrailProviderResponse)
async def get_guardrail_provider(
    guardrail_provider_id: str,
    ctx=Depends(auth_imbrace),
):
    """Get a single guardrail provider by ID."""
    try:
        org_id = ctx["organization_id"]
        provider = await repo.get_by_id(org_id, guardrail_provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guardrail provider not found",
            )

        for field in ("created_at", "updated_at"):
            if hasattr(provider.get(field), "isoformat"):
                provider[field] = provider[field].isoformat()

        return provider
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error getting guardrail provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("", response_model=GuardrailProviderResponse)
async def create_guardrail_provider(
    body: GuardrailProviderCreateRequest,
    ctx=Depends(auth_imbrace),
):
    """Create a new custom guardrail provider."""
    try:
        org_id = ctx["organization_id"]

        if body.type not in GUARDRAIL_PROVIDER_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider type '{body.type}'. Must be one of: {GUARDRAIL_PROVIDER_TYPES}",
            )

        provider_id = str(uuid.uuid4())

        details = {
            "guardrail_provider_id": provider_id,
            "name": body.name,
            "type": body.type,
            "config": body.config.dict(exclude_none=True),
        }

        result = await repo.create(org_id, details)

        for field in ("created_at", "updated_at"):
            if hasattr(result.get(field), "isoformat"):
                result[field] = result[field].isoformat()

        return result
    except Exception as e:
        log.error(f"Error creating guardrail provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.put("/{guardrail_provider_id}", response_model=GuardrailProviderResponse)
async def update_guardrail_provider(
    guardrail_provider_id: str,
    body: GuardrailProviderUpdateRequest,
    ctx=Depends(auth_imbrace),
):
    """Update an existing guardrail provider."""
    try:
        org_id = ctx["organization_id"]

        # Verify exists
        existing = await repo.get_by_id(org_id, guardrail_provider_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guardrail provider not found",
            )

        update_data = {}
        if body.name is not None:
            update_data["name"] = body.name
        if body.config is not None:
            update_data["config"] = body.config.dict()
        if body.is_shown is not None:
            update_data["is_shown"] = body.is_shown
        if body.models_metadata is not None:
            update_data["models_metadata"] = body.models_metadata

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update",
            )

        result = await repo.update(guardrail_provider_id, update_data)

        for field in ("created_at", "updated_at"):
            if result and hasattr(result.get(field), "isoformat"):
                result[field] = result[field].isoformat()

        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error updating guardrail provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{guardrail_provider_id}")
async def delete_guardrail_provider(
    guardrail_provider_id: str,
    ctx=Depends(auth_imbrace),
):
    """Soft delete a guardrail provider."""
    try:
        org_id = ctx["organization_id"]

        existing = await repo.get_by_id(org_id, guardrail_provider_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guardrail provider not found",
            )

        await repo.remove(guardrail_provider_id)
        return {
            "message": "Guardrail provider deleted successfully",
            "guardrail_provider_id": guardrail_provider_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error deleting guardrail provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/test-connection")
async def test_guardrail_provider_connection(
    body: GuardrailProviderCreateRequest,
    ctx=Depends(auth_imbrace),
):
    """Test connection to a guardrail provider without creating it."""
    try:
        if body.type not in GUARDRAIL_PROVIDER_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid provider type '{body.type}'",
            )

        config = body.config.dict(exclude_none=True)
        service = create_guardrail_service(body.type, config)
        is_reachable = await service.validate_connection()

        if not is_reachable:
            return {
                "status": "unreachable",
                "is_connected": False,
            }

        result = {
            "status": "connected",
            "is_connected": True,
        }

        # Fetch models if supported — must succeed for connection to be valid
        if hasattr(service, "fetch_models"):
            models = await service.fetch_models()
            if not models:
                return {
                    "status": "unreachable",
                    "is_connected": False,
                    "detail": "Connected but failed to fetch models. Check the endpoint URL.",
                }
            result["models"] = models

        return result
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error testing guardrail provider connection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/{guardrail_provider_id}/test")
async def test_guardrail_provider(
    guardrail_provider_id: str,
    ctx=Depends(auth_imbrace),
):
    """Test connection to a guardrail provider."""
    try:
        org_id = ctx["organization_id"]
        provider = await repo.get_by_id(org_id, guardrail_provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guardrail provider not found",
            )

        config = provider.get("config", {})
        provider_type = provider.get("type", "custom")
        service = create_guardrail_service(provider_type, config)
        is_reachable = await service.validate_connection()

        return {
            "guardrail_provider_id": guardrail_provider_id,
            "status": "connected" if is_reachable else "unreachable",
            "is_connected": is_reachable,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error testing guardrail provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{guardrail_provider_id}/models")
async def list_guardrail_provider_models(
    guardrail_provider_id: str,
    ctx=Depends(auth_imbrace),
):
    """Fetch available models from a guardrail provider (Ollama, NIM NeMo, LlamaGuard)."""
    try:
        org_id = ctx["organization_id"]
        provider = await repo.get_by_id(org_id, guardrail_provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guardrail provider not found",
            )

        config = provider.get("config", {})
        provider_type = provider.get("type", "custom")
        service = create_guardrail_service(provider_type, config)

        if not hasattr(service, "fetch_models"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider type '{provider_type}' does not support listing models",
            )

        models = await service.fetch_models()
        if not models:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to fetch models from provider. Check the endpoint URL and credentials.",
            )

        # Merge persisted per-model metadata (e.g. custom icon image_url)
        models_metadata = provider.get("models_metadata") or {}
        if models_metadata:
            for m in models:
                meta = models_metadata.get(m.get("name"))
                if meta and meta.get("image_url"):
                    m["image_url"] = meta["image_url"]

        return {"models": models}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error fetching guardrail provider models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
