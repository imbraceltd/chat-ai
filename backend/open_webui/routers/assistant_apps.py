# open_webui/routers/assistants.py

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from typing import List, Optional
import logging
from ..models.ai_agents.assistants import AssistantResponse, AssistantCreate, AssistantUpdate
from ..constants import ERROR_MESSAGES
from ..utils.auth import auth_imbrace
from open_webui.utils import app_service as app_service
from open_webui.routers.helper import validate_create_assistant_app_input
from open_webui.env import SRC_LOG_LEVELS
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("ASSISTANT_APPS", logging.INFO))
router = APIRouter()


@router.post("")
async def create_assistant(
    request: Request,
    userContext=Depends(auth_imbrace)
):
    try:
        body = await request.json()
        log.info(f"create_assistant called with body: {body}")
        assistant_details = validate_create_assistant_app_input(body)

        created_assistant = await app_service.create(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_details=assistant_details
        )
        return created_assistant
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.put("/{assistant_id}")
async def update_assistant(
    request: Request,
    assistant_id: str,
    userContext=Depends(auth_imbrace)
):
    try:
        body = await request.json()
        # log.info(f"update_assistant called with body: {body}")
        assistant_details = validate_create_assistant_app_input(body)
        updated_assistant = await app_service.update(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_id=assistant_id,
            assistant_details=assistant_details
        )
        if not updated_assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return updated_assistant
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


@router.delete("/{assistant_id}")
async def remove_assistant(
    assistant_id: str,
    userContext=Depends(auth_imbrace)
):
    try:
        if not assistant_id:
            raise HTTPException(
                status_code=400, detail="Assistant ID is required")

        log.info(f"remove_assistant called with assistant_id: {assistant_id}")
        deleted = await app_service.remove(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_id=assistant_id
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return deleted
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e))
        )


