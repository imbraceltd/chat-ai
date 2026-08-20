# open_webui/routers/assistants.py

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from typing import List, Optional
import logging

from ..models.ai_agents.assistants import AssistantUpdate
from ..constants import ERROR_MESSAGES
from ..utils.auth import auth_imbrace
from open_webui.utils import workflow as workflow_service
from open_webui.routers.helper import (
    validate_create_assistant_app_input,
    validate_create_assistant_input,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("ASSISTANTS", logging.INFO))
router = APIRouter()


@router.get("")
async def get_all(userContext=Depends(auth_imbrace)):
    """
    List all assistant workflows for the current organization.
    Returns assistant_id and workflow_id for each assistant.
    """
    try:
        organization_id = userContext.get("organization_id")
        api_key = userContext.get("api_key")

        data = await workflow_service.get_all(organization_id, api_key)

        return data

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get workflows: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.get("/assistants/{id}/workflows")
async def get_by_assistant_id(
    assistant_id: str,
    userContext=Depends(auth_imbrace),
):
    try:
        organization_id = userContext.get("organization_id")
        api_key = userContext.get("api_key")

        data = await workflow_service.get_by_assistant_id(organization_id, api_key)

        return data

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Failed to get workflows: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )
