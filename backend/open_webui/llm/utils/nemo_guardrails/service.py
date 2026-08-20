import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from open_webui.repository.guardrail import guardrail_repo
from open_webui.llm.utils.nemo_guardrails.guardrails_loader import (
    load_rail_from_config,
    llm_rails_instances,
)

logger = logging.getLogger(__name__)


class NemoGuardrailService:
    """Direct in-process NemoGuardrails service, replacing the external HTTP client."""

    async def check_safety(self, data: Dict[str, Any]) -> Dict[str, Any]:
        config_id = data.get("guardrails_config_id", "")
        message = data.get("message", "")

        if (
            config_id not in llm_rails_instances
            or llm_rails_instances[config_id] is None
        ):
            return {
                "status": "error",
                "message": f"Rules not loaded for config {config_id}.",
            }

        try:
            import json
            rails_instance = llm_rails_instances[config_id]
            results = await rails_instance.generate_async(
                messages=[{"role": "user", "content": message}]
            )
            response_content = json.dumps(results)
            json_response = json.loads(response_content)

            return {
                "is_safe": json_response["content"],
                "guardrails_config_id": config_id,
            }
        except Exception as e:
            logger.error(f"Error in NemoGuardrail check_safety: {e}")
            return {
                "status": "error",
                "message": f"Error processing request: {str(e)}",
            }

    async def list_guardrails(
        self, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            org_id = (params or {}).get("org_id", "")
            configs = await guardrail_repo.list_by_org(org_id)
            return configs if configs else []
        except Exception as e:
            logger.error(f"Error listing NemoGuardrail configs: {e}")
            return []

    async def get_guardrail(self, guardrail_id: str, org_id: str = "") -> Optional[Dict[str, Any]]:
        try:
            doc = await guardrail_repo.get_by_config_id(guardrail_id, org_id)
            if doc:
                doc["rails_loaded"] = (
                    guardrail_id in llm_rails_instances
                    and llm_rails_instances[guardrail_id] is not None
                )
            return doc
        except Exception as e:
            logger.error(f"Error getting NemoGuardrail config {guardrail_id}: {e}")
            return None

    async def create_guardrail(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
        try:
            generated_id = str(uuid.uuid4())
            config_data = dict(data)
            config_data["guardrails_config_id"] = generated_id

            inserted = await guardrail_repo.create(config_data)
            if not inserted:
                return None, 500, "Failed to store guardrail config"

            try:
                load_rail_from_config(generated_id, config_data)
            except Exception as e:
                logger.error(f"Config stored but failed to load rails: {e}")

            return (
                {
                    "message": "Config created successfully",
                    "guardrails_config_id": generated_id,
                    "status": "success",
                },
                None,
                None,
            )
        except Exception as e:
            logger.error(f"Error creating NemoGuardrail config: {e}")
            return None, 500, str(e)

    async def update_guardrail(
        self, guardrail_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        try:
            org_id = data.get("org_id", "")
            update_data = {
                k: v for k, v in data.items()
                if k not in ("guardrails_config_id", "org_id") and v is not None
            }
            if not update_data:
                return None

            await guardrail_repo.update(guardrail_id, org_id, update_data)

            doc = await guardrail_repo.get_by_config_id(guardrail_id, org_id)
            if doc:
                try:
                    load_rail_from_config(guardrail_id, doc)
                except Exception as e:
                    logger.error(f"Config updated but failed to reload rails: {e}")

            is_loaded = (
                guardrail_id in llm_rails_instances
                and llm_rails_instances[guardrail_id] is not None
            )

            return {
                "message": (
                    "Config updated and Guardrails reloaded successfully."
                    if is_loaded
                    else "Config updated but failed to reload Guardrails."
                ),
                "guardrails_config_id": guardrail_id,
                "status": "success" if is_loaded else "partial_success",
            }
        except Exception as e:
            logger.error(f"Error updating NemoGuardrail config {guardrail_id}: {e}")
            return None

    async def delete_guardrail(self, guardrail_id: str, org_id: str = "") -> bool:
        try:
            deleted_count = await guardrail_repo.delete(guardrail_id, org_id)
            if deleted_count > 0:
                llm_rails_instances.pop(guardrail_id, None)
                logger.info(f"Deleted NemoGuardrail config: {guardrail_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting NemoGuardrail config {guardrail_id}: {e}")
            return False
