import json
import logging
import ast
from typing import Dict, Any, Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from open_webui.utils.auth import auth_imbrace
from open_webui.env import SRC_LOG_LEVELS, GUARDRAILS
from open_webui.llm.utils.model_armor import create_model_armor_service
from open_webui.llm.utils.guardrail import Guardrail, create_guardrail_service
from open_webui.llm.utils.safety import check_safety
from open_webui.constants import ERROR_MESSAGES
from open_webui.repository.guardrail import guardrail_repo

logger = logging.getLogger(__name__)

router = APIRouter()


def get_enabled_providers() -> List[str]:
    """Parse GUARDRAILS env var into list of enabled providers."""
    if not GUARDRAILS:
        return ["nim-nemo", "model-armor"]  # Default providers
    return [p.strip() for p in GUARDRAILS.split(",") if p.strip()]


# Pydantic models
class GuardrailProviderInfo(BaseModel):
    guardrail_provider_id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class GuardrailModel(BaseModel):
    org_id: Optional[str] = None
    guardrails_config_id: Optional[str] = None
    unsafe_categories: Optional[List[str]] = None
    custom_unsafe_patterns: Optional[List[str]] = None
    competitor_keywords: Optional[List[str]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    model: Optional[str] = ""
    name: Optional[str] = None
    description: Optional[str] = ""
    instructions: Optional[str] = ""  # Required for create/update, optional when listing
    guardrail_provider_id: Optional[str] = None  # For custom providers
    provider: Optional[GuardrailProviderInfo] = None  # Provider config for custom guardrails


class GuardrailCreateRequest(BaseModel):
    unsafe_categories: Optional[List[str]] = None
    custom_unsafe_patterns: Optional[List[str]] = None
    competitor_keywords: Optional[List[str]] = None
    model: str
    name: str
    description: Optional[str] = ""
    instructions: str  # Required by NIM Guardrails API
    guardrail_provider_id: Optional[str] = None  # Required when model="custom"


class GuardrailUpdateRequest(BaseModel):
    unsafe_categories: Optional[List[str]] = None
    custom_unsafe_patterns: Optional[List[str]] = None
    competitor_keywords: Optional[List[str]] = None
    model: str
    name: str
    description: Optional[str] = ""
    instructions: str  # Required by NIM Guardrails API
    guardrail_provider_id: Optional[str] = None


class SafetyCheckRequest(BaseModel):
    org_id: str
    message: str
    guardrails_config_id: str


class GuardrailCRUDResponse(BaseModel):
    message: str
    guardrails_config_id: str
    status: str = "success"


class SafetyCheckResponse(BaseModel):
    is_safe: bool
    safety_categories: Optional[str] = "Unknown"
    guardrails_config_id: str


class ErrorResponse(BaseModel):
    message: str


def handle_controller_error(error: Exception) -> tuple[str, int]:
    """Handle controller errors."""
    logger.error(
        f"Controller error: {type(error).__name__}: {str(error)}", exc_info=True
    )
    if isinstance(error, HTTPException):
        return error.detail, error.status_code
    elif isinstance(error, ValueError):
        return str(error), status.HTTP_400_BAD_REQUEST
    elif isinstance(error, KeyError):
        return "Missing required field", status.HTTP_400_BAD_REQUEST
    else:
        return "An internal error occurred.", status.HTTP_500_INTERNAL_SERVER_ERROR


def validate_guardrail_details(guardrail_details: Dict[str, Any]) -> bool:
    """Validate guardrail details."""
    errors = {
        "INVALID_GUARDRAIL_DETAILS_ERR": "Invalid guardrail details. Expected a dictionary.",
        "REQUIRED_FIELD_ERR": lambda field: f"The field '{field}' is required.",
        "INVALID_TYPE_ERR": lambda field, expected: f"The field '{field}' must be of type {expected}.",
        "INVALID_ARRAY_ERR": lambda field: f"The field '{field}' must be an array of strings.",
        "INVALID_STRING_ERR": lambda field: f"The field '{field}' must be a non-empty string.",
    }

    if not isinstance(guardrail_details, dict):
        raise ValueError(errors["INVALID_GUARDRAIL_DETAILS_ERR"])

    required_fields = {
        "unsafe_categories": "array",
        "name": "string",
        "model": "string",
        "instructions": "string",  # Required by NIM Guardrails API
    }

    for field, expected_type in required_fields.items():
        if field not in guardrail_details:
            raise ValueError(errors["REQUIRED_FIELD_ERR"](field))

        value = guardrail_details[field]

        if expected_type == "string":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(errors["INVALID_STRING_ERR"](field))
        elif expected_type == "array":
            if not isinstance(value, list):
                raise ValueError(errors["INVALID_ARRAY_ERR"](field))
            if not all(isinstance(item, str) for item in value):
                raise ValueError(errors["INVALID_ARRAY_ERR"](field))

    # Validate description if present
    if (
        "description" in guardrail_details
        and guardrail_details["description"] is not None
    ):
        if not isinstance(guardrail_details["description"], str):
            raise ValueError(errors["INVALID_TYPE_ERR"]("description", "string"))

    return True


async def get_guardrail_service():
    """Dependency to provide Guardrail instance."""
    async with create_guardrail_service() as service:
        yield service



@router.get("/all", response_model=List[GuardrailModel])
async def list_guardrails(
    ctx=Depends(auth_imbrace),
):
    """List all guardrails for a given org_id."""
    try:
        org_id = ctx["organization_id"]

        rows = await guardrail_repo.list_by_org(org_id)

        db_configs = []
        for config in rows or []:
            if not isinstance(config, dict):
                continue
            if isinstance(config.get("created_at"), datetime):
                config["created_at"] = config["created_at"].isoformat()
            if isinstance(config.get("updated_at"), datetime):
                config["updated_at"] = config["updated_at"].isoformat()
            if not isinstance(config.get("model"), str):
                config["model"] = ""
            if not isinstance(config.get("description"), str):
                config["description"] = ""
            db_configs.append(config)

        # Enrich custom guardrails with provider config
        custom_configs = [c for c in db_configs if c.get("model") == "custom"]
        if custom_configs:
            try:
                from open_webui.repository.guardrail_provider import (
                    GuardrailProviderRepository,
                )
                gp_repo = GuardrailProviderRepository()
                provider_ids = list(
                    {
                        c["guardrail_provider_id"]
                        for c in custom_configs
                        if c.get("guardrail_provider_id")
                    }
                )
                provider_map = {}
                for pid in provider_ids:
                    p = await gp_repo.get_by_id(org_id, pid)
                    if p:
                        p_config = p.get("config", {})
                        provider_map[pid] = {
                            "guardrail_provider_id": p.get("guardrail_provider_id"),
                            "name": p.get("name"),
                            "type": p.get("type"),
                            "config": p_config,
                        }
                for c in custom_configs:
                    pid = c.get("guardrail_provider_id")
                    if pid and pid in provider_map:
                        c["provider"] = provider_map[pid]
            except Exception as e:
                logger.error(f"Error enriching custom guardrails with provider config: {e}")

        combined_configs = db_configs
        # Filter out None values but keep important keys
        keep_keys = {
            "org_id", "guardrails_config_id", "model", "unsafe_categories",
            "custom_unsafe_patterns", "competitor_keywords", "created_at",
            "updated_at", "name", "description", "status", "version",
            "arn", "guardrail_provider_id", "provider",
        }
        combined_configs = [
            {k: v for k, v in config.items() if v is not None or k in keep_keys}
            for config in combined_configs
            if isinstance(config, dict)
        ]

        try:
            return [GuardrailModel(**item) for item in combined_configs]
        except Exception as e:
            logger.error(f"Error validating GuardrailModel: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Invalid guardrail data: {str(e)}",
            )
    except Exception as error:
        message, status_code = handle_controller_error(error)
        raise HTTPException(status_code=status_code, detail=message)


@router.get("/{guardrail_id}", response_model=GuardrailModel)
async def get_guardrail(
    guardrail_id: str,
    guardrail_service: Guardrail = Depends(get_guardrail_service),
    ctx=Depends(auth_imbrace),
):
    """Get a specific guardrail by ID."""
    try:
        # Always check PostgreSQL first — all guardrail types are stored there.
        config = await guardrail_repo.get_by_config_id(
            guardrail_id, ctx["organization_id"]
        )
        if config:
            if isinstance(config.get("created_at"), datetime):
                config["created_at"] = config["created_at"].isoformat()
            if isinstance(config.get("updated_at"), datetime):
                config["updated_at"] = config["updated_at"].isoformat()
            if not isinstance(config.get("model"), str):
                config["model"] = ""
            if not isinstance(config.get("description"), str):
                config["description"] = ""

            # Enrich custom guardrails with provider config
            if config.get("model") == "custom":
                pid = config.get("guardrail_provider_id")
                if pid:
                    try:
                        from open_webui.repository.guardrail_provider import (
                            GuardrailProviderRepository,
                        )
                        gp_repo = GuardrailProviderRepository()
                        p = await gp_repo.get_by_id(ctx["organization_id"], pid)
                        if p:
                            config["provider"] = {
                                "guardrail_provider_id": p.get("guardrail_provider_id"),
                                "name": p.get("name"),
                                "type": p.get("type"),
                                "config": p.get("config", {}),
                            }
                    except Exception as e:
                        logger.error(f"Error enriching custom guardrail with provider: {e}")

            return GuardrailModel(**config)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Guardrail not found"
        )
    except Exception as error:
        message, status_code = handle_controller_error(error)
        raise HTTPException(status_code=status_code, detail=message)


@router.post("/create", response_model=GuardrailCRUDResponse)
async def create_guardrail(
    body: GuardrailCreateRequest,
    guardrail_service: Guardrail = Depends(get_guardrail_service),
    ctx=Depends(auth_imbrace),
):
    """Create a new guardrail."""
    try:
        data = body.dict(exclude_unset=True)
        validate_guardrail_details(data)

        # Prevent creating Bedrock guardrails via API
        if data.get("model") == "bedrock":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bedrock guardrails must be created via AWS Console. This API only fetches existing Bedrock guardrails.",
            )

        # Custom guardrail provider
        if data.get("model") == "custom":
            if not data.get("guardrail_provider_id"):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="guardrail_provider_id is required when model is 'custom'.",
                )
            # Validate provider exists
            from open_webui.repository.guardrail_provider import GuardrailProviderRepository
            gp_repo = GuardrailProviderRepository()
            provider = await gp_repo.get_by_id(ctx["organization_id"], data["guardrail_provider_id"])
            if not provider:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Guardrail provider {data['guardrail_provider_id']} not found.",
                )

            data["org_id"] = ctx["organization_id"]
            data["created_at"] = datetime.utcnow().isoformat()
            data["updated_at"] = datetime.utcnow().isoformat()

            import uuid as _uuid
            config_id = str(_uuid.uuid4())
            data["guardrails_config_id"] = config_id

            inserted = await guardrail_repo.create(data)
            if not inserted:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to store custom guardrail config",
                )

            return GuardrailCRUDResponse(
                message="Custom guardrail created successfully.",
                guardrails_config_id=config_id,
                status="success",
            )

        data["org_id"] = ctx["organization_id"]
        # Set created_at and updated_at as strings
        data["created_at"] = datetime.utcnow().isoformat()
        data["updated_at"] = datetime.utcnow().isoformat()

        if data["model"] == "model-armor":
            logger.info("=== CREATING MODEL ARMOR GUARDRAIL ===")
            logger.info("Input data: %s", json.dumps(data, indent=2, default=str))

            # Build filter config
            filter_config = {
                "filterConfig": {
                    "raiSettings": {
                        "raiFilters": [
                            {
                                "filterType": category.strip()
                                .replace(" ", "_")
                                .upper(),
                                "confidenceLevel": "MEDIUM_AND_ABOVE",
                            }
                            for category in data["unsafe_categories"]
                        ],
                    },
                    "piAndJailbreakFilterSettings": {
                        "filterEnforcement": "ENABLED",
                        "confidenceLevel": "LOW_AND_ABOVE",
                    },
                    "maliciousUriFilterSettings": {
                        "filterEnforcement": "ENABLED",
                    },
                }
            }

            logger.info("Filter config to send: %s", json.dumps(filter_config, indent=2))
            logger.info("Template name: %s", data["name"].replace(" ", ""))

            async with create_model_armor_service() as model_armor_service:
                try:
                    created_template = await model_armor_service.create_template(
                        filter_config, data["name"].replace(" ", "")
                    )
                    logger.info("✅ Model Armor template created successfully")
                    logger.info("Created template response: %s", json.dumps(created_template, indent=2, default=str))
                except Exception as e:
                    logger.error("❌ Model Armor create_template failed: %s", str(e))
                    logger.error("Exception type: %s", type(e).__name__)
                    logger.error("Full exception:", exc_info=True)

                    # Handle specific error codes
                    error_msg = str(e)
                    if "409" in error_msg or "Conflict" in error_msg:
                        raise HTTPException(
                            status_code=status.HTTP_409_CONFLICT,
                            detail=f"Guardrail template '{data['name']}' already exists. Please use a different name or delete the existing template first."
                        )
                    elif "400" in error_msg or "Bad Request" in error_msg:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid Model Armor configuration: {error_msg}"
                        )
                    elif "401" in error_msg or "Unauthorized" in error_msg:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Model Armor authentication failed. Please check credentials."
                        )
                    elif "403" in error_msg or "Forbidden" in error_msg:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Access denied to Model Armor API. Please check permissions."
                        )
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Model Armor API error: {error_msg}"
                        )

                try:
                    guardrail_config_id = model_armor_service.extract_template_id(
                        created_template.get("name", "")
                    )
                    logger.info("Extracted guardrail_config_id: %s", guardrail_config_id)
                except Exception as e:
                    logger.error("❌ Failed to extract template ID: %s", str(e))
                    logger.error("Template response was: %s", created_template)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to extract template ID: {str(e)}"
                    )

            # Store in PostgreSQL
            data["guardrails_config_id"] = guardrail_config_id
            logger.info("Storing in MongoDB with data: %s", json.dumps(data, indent=2, default=str))

            try:
                inserted_doc = await guardrail_repo.create(data)
                logger.info("PostgreSQL insert result: %s", inserted_doc)
            except Exception as e:
                logger.error("❌ PostgreSQL insert failed: %s", str(e))
                logger.error("Full exception:", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to store in MongoDB: {str(e)}"
                )

            if not inserted_doc:
                logger.error("❌ MongoDB insert returned None/False")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to store guardrail in MongoDB",
                )

            if not inserted_doc:
                logger.error("❌ PostgreSQL insert returned None")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to store guardrail",
                )

            logger.info(
                "✅ Model Armor guardrail created successfully with ID: %s",
                guardrail_config_id,
            )
            return GuardrailCRUDResponse(
                message="Model Armor template created successfully.",
                guardrails_config_id=guardrail_config_id,
                status="success",
            )

        # Log payload trước khi gửi
        logger.info("=== CREATING NIM GUARDRAIL ===")
        logger.info("Model: %s", data.get("model"))
        logger.info("Payload keys: %s", list(data.keys()))
        logger.info("Full payload: %s", json.dumps(data, indent=2, default=str))

        # create_guardrail now returns (result, status_code, error_detail)
        result, error_status, error_detail = await guardrail_service.create_guardrail(
            data
        )

        if result is None:
            # NIM API trả về error - preserve status code và detail
            logger.error(
                "❌ NIM Guardrail API error %s: %s", error_status, error_detail
            )

            # Map status code hoặc dùng luôn từ NIM API
            final_status = error_status or status.HTTP_500_INTERNAL_SERVER_ERROR

            # Format detail message
            if error_status == 422:
                detail_msg = f"NIM Guardrails validation error: {error_detail}"
            elif error_status == 400:
                detail_msg = f"NIM Guardrails bad request: {error_detail}"
            else:
                detail_msg = f"NIM Guardrails error ({error_status}): {error_detail}"

            raise HTTPException(
                status_code=final_status,
                detail=detail_msg,
            )

        logger.info("✅ NIM Guardrail created successfully: %s", result)
        return GuardrailCRUDResponse(
            message="NVIDIA Guardrail created successfully.",
            guardrails_config_id=result.get("guardrails_config_id"),
            status="success",
        )
    except HTTPException:
        # Re-raise HTTPException để giữ nguyên status code
        raise
    except Exception as error:
        logger.error(
            "❌ Exception in create_guardrail endpoint: %s: %s",
            type(error).__name__,
            str(error),
        )
        logger.error("Full error: %s", error, exc_info=True)

        message, status_code = handle_controller_error(error)
        if status_code == 409:
            message = "Model Armor template name already exists."
        elif status_code == 400:
            message = "Invalid input for guardrail creation."
        raise HTTPException(status_code=status_code, detail=message)


@router.put("/update/{guardrail_id}", response_model=GuardrailCRUDResponse)
async def update_guardrail(
    guardrail_id: str,
    body: GuardrailUpdateRequest,
    guardrail_service: Guardrail = Depends(get_guardrail_service),
    ctx=Depends(auth_imbrace),
):
    """Update a guardrail."""
    try:
        data = body.dict(exclude_unset=True)
        validate_guardrail_details(data)
        data["org_id"] = ctx["organization_id"]
        data["updated_at"] = datetime.utcnow().isoformat()

        if data["model"] == "model-armor":
            filter_config = {
                "filterConfig": {
                    "raiSettings": {
                        "raiFilters": [
                            {
                                "filterType": category.strip()
                                .replace(" ", "_")
                                .upper(),
                                "confidenceLevel": "MEDIUM_AND_ABOVE",
                            }
                            for category in data["unsafe_categories"]
                        ],
                    },
                    "piAndJailbreakFilterSettings": {
                        "filterEnforcement": "ENABLED",
                        "confidenceLevel": "LOW_AND_ABOVE",
                    },
                    "maliciousUriFilterSettings": {
                        "filterEnforcement": "ENABLED",
                    },
                }
            }
            # Pass description to Model Armor if present
            if "description" in data:
                filter_config["description"] = data["description"]
            async with create_model_armor_service() as model_armor_service:
                updated_template = await model_armor_service.update_template(
                    guardrail_id, filter_config
                )
            logger.debug(f"Updated Model Armor template: {updated_template}")

            # Update in PostgreSQL
            updated_doc = await guardrail_repo.update(
                guardrail_id, ctx["organization_id"], data
            )
            if not updated_doc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Guardrail not found",
                )

            return GuardrailCRUDResponse(
                message="Model Armor template updated successfully.",
                guardrails_config_id=guardrail_id,
                status="success",
            )

        result = await guardrail_service.update_guardrail(guardrail_id, data)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Guardrail not found"
            )
        return GuardrailCRUDResponse(
            message="NVIDIA Guardrail updated successfully.",
            guardrails_config_id=guardrail_id,
            status="success",
        )
    except Exception as error:
        message, status_code = handle_controller_error(error)
        if status_code == 404:
            message = "Guardrail not found"
        raise HTTPException(status_code=status_code, detail=message)


@router.delete("/delete/{guardrail_id}", response_model=GuardrailCRUDResponse)
async def delete_guardrail(
    guardrail_id: str,
    model: Optional[str] = Query(None),
    guardrail_service: Guardrail = Depends(get_guardrail_service),
    ctx=Depends(auth_imbrace),
):
    """Delete a guardrail."""
    try:
        if model == "model-armor" or not model:
            try:
                async with create_model_armor_service() as model_armor_service:
                    response = await model_armor_service.delete_template(guardrail_id)
                logger.debug(
                    f"Model Armor delete_template response for {guardrail_id}: {response}"
                )
                # Treat empty dict or non-None response as success (status 200 is ensured by raise_for_status)
                if response is not None or response == {}:
                    deleted_count = await guardrail_repo.delete(
                        guardrail_id, ctx["organization_id"]
                    )
                    logger.debug(
                        f"PostgreSQL delete result for {guardrail_id}: {deleted_count} row(s)"
                    )
                    if deleted_count == 0:
                        logger.warning(
                            f"No Model Armor guardrail found in PostgreSQL for id: {guardrail_id}"
                        )
                    return GuardrailCRUDResponse(
                        message="Model Armor template deleted successfully.",
                        guardrails_config_id=guardrail_id,
                        status="success",
                    )
                else:
                    logger.info(
                        f"Model Armor template not found for id: {guardrail_id}"
                    )
            except Exception as e:
                logger.info(
                    f"No Model Armor template found or error deleting for id: {guardrail_id}, error: {str(e)}"
                )

        result = await guardrail_service.delete_guardrail(guardrail_id, ctx["organization_id"])
        logger.debug(f"NVIDIA Guardrail delete result for {guardrail_id}: {result}")
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Guardrail not found"
            )
        return GuardrailCRUDResponse(
            message="NVIDIA Guardrail deleted successfully.",
            guardrails_config_id=guardrail_id,
            status="success",
        )
    except Exception as error:
        message, status_code = handle_controller_error(error)
        if status_code == 404:
            message = "Guardrail not found"
        raise HTTPException(status_code=status_code, detail=message)


@router.post("/check", response_model=SafetyCheckResponse)
async def check_safety_endpoint(
    body: SafetyCheckRequest,
    guardrail_service: Guardrail = Depends(get_guardrail_service),
    ctx=Depends(auth_imbrace),
):
    """Check message safety using guardrails."""
    try:
        return await check_safety(
            org_id=body.org_id,
            message=body.message,
            guardrails_config_id=body.guardrails_config_id,
            model_armor_service=None,
            guardrail_service=guardrail_service,
        )
    except Exception as error:
        message, status_code = handle_controller_error(error)
        raise HTTPException(status_code=status_code, detail=message)


__all__ = ["router"]
