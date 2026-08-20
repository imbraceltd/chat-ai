from ..repository.assistant import AssistantRepository

# Assuming you have this utility
from ..utils.misc import generate_ai_assistant_instruction
from typing import Dict, List
from typing import List, Dict
import uuid
from open_webui.config import OPENAI_API_KEY
import logging
from open_webui.env import SRC_LOG_LEVELS
from open_webui.repository.workflow.ai import get_all, update_workflow_name, update
from ..repository.api_key import ApiKeyRepository
from ..repository.rag import RAGRepository
from open_webui.models.ai_agents.files import files_model
from ..utils.board_embedding_job import create_board_embedding_jobs_for_assistant

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])


# open_webui/services/assistant_service.py
assistant_repo = AssistantRepository()
api_key_repo = ApiKeyRepository()
rag_repo = RAGRepository()

log = logging.getLogger(__name__)


async def get_all(organization_id: str, api_key: str) -> List:
    """
    Retrieve all assistants for the given organization with filtering options.
    """
    # Validate API key and check if it's paid
    api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

    log.info(
        f"get_all_assistants called for organization_id={organization_id} with filters:"
    )

    assistants_data = await assistant_repo.get_assistant_meta_by_organization_id(
        organization_id=organization_id,
    )

    # Handle None or empty response
    if not assistants_data:
        return []

    # Transform the data to extract assistant_id and workflow_id
    assistant_workflows = [
        {
            "assistant_id": item.get("assistant_id"),
            "workflow_id": (item.get("metadata") or {}).get("workflow_id"),
        }
        for item in assistants_data
        if item is not None
    ]

    return assistant_workflows


async def get_by_assistant_id(
    organization_id: str = "", api_key: str = "", assistant_id: str = ""
) -> Dict:
    """
    Retrieve the workflow_id for a given assistant within an organization.
    Returns a dict with `assistant_id` and `workflow_id` (or None).
    """
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        assistant = await assistant_repo.get_by_assistant_ids_and_organization_id(
            organization_id, assistant_id
        )

        workflow_id = None
        if assistant:
            workflow_id = (assistant.get("metadata") or {}).get("workflow_id")

        return {"assistant_id": assistant_id, "workflow_id": workflow_id}
    except Exception as error:
        raise error
