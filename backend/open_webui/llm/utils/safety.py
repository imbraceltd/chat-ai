import json
import ast
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel
from open_webui.llm.utils.model_armor import (
    ModelArmorService,
    create_model_armor_service,
)
from open_webui.llm.utils.guardrail import Guardrail, create_guardrail_service
from open_webui.llm.utils.bedrock_guardrail import (
    BedrockGuardrailService,
    create_bedrock_guardrail_service,
)
from open_webui.internal.mongo_db import query_one
from open_webui.config import MONGODB_CONFIG
from open_webui.env import GUARDRAILS


logger = logging.getLogger(__name__)

OPENAI_DB_NAME = MONGODB_CONFIG.get("openai_db_name", "imbrace_dev")
GUARDRAIL_COLLECTION = "guardrails"


class SafetyCheckResponse(BaseModel):
    is_safe: bool
    safety_categories: Optional[str] = "Unknown"
    guardrails_config_id: str


async def check_safety(
    org_id: str,
    message: str,
    guardrails_config_id: str,
    model_armor_service: Optional[ModelArmorService] = None,
    guardrail_service: Optional[Guardrail] = None,
    bedrock_service: Optional[BedrockGuardrailService] = None,
) -> SafetyCheckResponse:
    """
    Check message safety using guardrails (Model Armor or NVIDIA Guardrail).

    Args:
        org_id: The organization ID.
        message: The message to check.
        guardrails_config_id: The guardrail configuration ID.
        model_armor_service: The ModelArmorService instance (optional, created if None).
        guardrail_service: The Guardrail service instance (optional, created if None).

    Returns:
        SafetyCheckResponse with safety check results.

    Raises:
        Exception: If an error occurs during the safety check.
    """
    try:
        # Create Guardrail if not provided
        if guardrail_service is None:
            async with create_guardrail_service(
                backend_host=None
            ) as created_guardrail_service:
                return await check_safety(
                    org_id=org_id,
                    message=message,
                    guardrails_config_id=guardrails_config_id,
                    model_armor_service=model_armor_service,
                    guardrail_service=created_guardrail_service,
                    bedrock_service=bedrock_service,
                )

        # Create BedrockGuardrailService if not provided
        if bedrock_service is None:
            async with create_bedrock_guardrail_service() as created_bedrock_service:
                return await check_safety(
                    org_id=org_id,
                    message=message,
                    guardrails_config_id=guardrails_config_id,
                    model_armor_service=model_armor_service,
                    guardrail_service=guardrail_service,
                    bedrock_service=created_bedrock_service,
                )

        # Try to find config in guardrails collection (Model Armor + custom)
        config = await query_one(
            database_name=OPENAI_DB_NAME,
            collection_name=GUARDRAIL_COLLECTION,
            query_dict={"guardrails_config_id": guardrails_config_id, "org_id": org_id},
        )
        if config and config.get("model") == "model-armor":
            # Convert datetime to string and handle model/description
            if isinstance(config.get("created_at"), datetime):
                config["created_at"] = config["created_at"].isoformat()
            if isinstance(config.get("updated_at"), datetime):
                config["updated_at"] = config["updated_at"].isoformat()
            config["model"] = (
                config.get("model", "") if isinstance(config.get("model"), str) else ""
            )
            config["description"] = (
                config.get("description", "")
                if isinstance(config.get("description"), str)
                else ""
            )
            armor_input = {
                "user_prompt_data": {"text": message},
                "multi_language_detection_metadata": {
                    "enable_multi_language_detection": True
                },
            }
            # Lazy-init ModelArmorService only when we actually need it
            if model_armor_service is None:
                async with create_model_armor_service() as created_model_armor_service:
                    model_armor_response = await created_model_armor_service.sanitize_prompt(
                        guardrails_config_id, armor_input
                    )
            else:
                model_armor_response = await model_armor_service.sanitize_prompt(
                    guardrails_config_id, armor_input
                )

            is_safe = (
                model_armor_response.get("sanitizationResult", {}).get(
                    "filterMatchState"
                )
                != "MATCH_FOUND"
            )
            safety_categories = "Unknown"

            if not is_safe:
                categories = (
                    model_armor_response.get("sanitizationResult", {})
                    .get("filterResults", {})
                    .get("rai", {})
                    .get("raiFilterResult", {})
                    .get("raiFilterTypeResults", {})
                )
                matched_categories = [
                    key.replace("_", " ").title()
                    for key, value in categories.items()
                    if value.get("matchState") == "MATCH_FOUND"
                ]
                safety_categories = (
                    ", ".join(matched_categories) if matched_categories else "Unknown"
                )

            return SafetyCheckResponse(
                is_safe=is_safe,
                safety_categories=safety_categories,
                guardrails_config_id=guardrails_config_id,
            )

        # Try Custom guardrail provider
        if config and config.get("model") == "custom":
            provider_id = config.get("guardrail_provider_id")
            if provider_id:
                try:
                    from open_webui.repository.guardrail_provider import GuardrailProviderRepository
                    from open_webui.llm.utils.custom_guardrail import create_guardrail_service

                    gp_repo = GuardrailProviderRepository()
                    provider = await gp_repo.get_by_id(org_id, provider_id)
                    if provider and provider.get("config"):
                        provider_type = provider.get("type", "custom")
                        custom_service = create_guardrail_service(provider_type, provider["config"])
                        result = await custom_service.check_safety(
                            message=message,
                            guardrail_config_id=guardrails_config_id,
                        )
                        return SafetyCheckResponse(
                            is_safe=result.get("is_safe", False),
                            safety_categories=result.get("safety_categories", "Unknown"),
                            guardrails_config_id=guardrails_config_id,
                        )
                    else:
                        logger.error(f"Custom guardrail provider {provider_id} not found or has no config")
                except Exception as e:
                    logger.error(f"Custom guardrail provider check failed: {e}")

        # Try Bedrock Guardrail (only if bedrock is enabled in GUARDRAILS)
        enabled_providers = [
            p.strip() for p in (GUARDRAILS or "").split(",") if p.strip()
        ]
        if "bedrock" in enabled_providers:
            try:
                bedrock_result = await bedrock_service.apply_guardrail(
                    guardrail_id=guardrails_config_id,
                    content=message,
                    source="INPUT",
                )

                is_safe = bedrock_result.get("is_safe", False)
                violated_categories = bedrock_result.get("violated_categories", [])

                # Format categories as comma-separated string
                safety_categories = (
                    ", ".join(violated_categories) if violated_categories else "Unknown"
                )

                return SafetyCheckResponse(
                    is_safe=is_safe,
                    safety_categories=safety_categories,
                    guardrails_config_id=guardrails_config_id,
                )
            except Exception as e:
                logger.info(
                    f"Bedrock guardrail check failed, falling back to NVIDIA: {e}"
                )

        # Fallback to NVIDIA Guardrail
        safety_result = await guardrail_service.check_safety(
            {
                "org_id": org_id,
                "message": message,
                "guardrails_config_id": guardrails_config_id,
            }
        )
        is_safe = safety_result
        safety_categories = "Unknown"

        if isinstance(safety_result, dict):
            is_safe_str = safety_result.get(
                "is_safe", "{'User Safety': 'safe', 'Safety Categories': 'None'}"
            )
            try:
                safety_info = json.loads(is_safe_str)
            except json.JSONDecodeError:
                try:
                    safety_info = ast.literal_eval(is_safe_str)
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Failed to parse is_safe: {e}")
                    safety_info = {}
            is_safe = safety_info.get("User Safety") != "unsafe"
            safety_categories = safety_info.get("Safety Categories", "Unknown")

        return SafetyCheckResponse(
            is_safe=is_safe,
            safety_categories=safety_categories,
            guardrails_config_id=guardrails_config_id,
        )
    except Exception as error:
        logger.error(f"Error in check_safety: {str(error)}")
        raise
