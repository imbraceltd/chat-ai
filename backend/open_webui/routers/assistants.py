# open_webui/routers/assistants.py

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from typing import List, Optional
import logging

from ..models.ai_agents.assistants import AssistantUpdate
from ..constants import ERROR_MESSAGES
from ..utils.auth import auth_imbrace
from open_webui.utils import assistant as assistant_service
from open_webui.routers.helper import (
    validate_create_assistant_app_input,
    validate_create_assistant_input,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("ASSISTANTS", logging.INFO))
router = APIRouter()


@router.get("/agents")
async def get_agents(
    userContext=Depends(auth_imbrace), assistant_id: Optional[str] = Query(None)
):
    try:
        agents = await assistant_service.get_agents(
            organization_id=userContext["organization_id"], assistant_id=assistant_id
        )
        return agents
    except Exception as e:
        log.error(f"Failed to get agents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.get("")
async def get_all_assistants(
    userContext=Depends(auth_imbrace),
    skip: int = 0,
    limit: int = -1,
    search: Optional[str] = None,
):
    try:
        # Pass the query parameters to the service layer
        filter_options = {"skip": skip, "limit": limit, "search": search}
        log.info(f"get_all_assistants called with filters: {filter_options}")

        # Call the service with the correctly populated filter_options
        assistants_response = await assistant_service.get_all(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            filter_options=filter_options,
        )

        return assistants_response["data"]

    except Exception as e:
        log.error(f"Failed to get assistants: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.get("/check-name")
async def check_assistant_name_exists(name: str, userContext=Depends(auth_imbrace)):
    try:
        assistant = await assistant_service.get_by_name(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            name=name,
        )
        return {"exist": bool(assistant)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.get("/{assistant_id}")
async def get_assistant_by_id(assistant_id: str, userContext=Depends(auth_imbrace)):
    try:
        assistant = await assistant_service.get_by_id(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_id=assistant_id,
        )
        if not assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return assistant
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.post("")
async def create_assistant(request: Request, userContext=Depends(auth_imbrace)):
    try:
        body = await request.json()
        log.info(f"create_assistant called with body: {body}")
        assistant_details = validate_create_assistant_input(body)
        created_assistant = await assistant_service.create(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_details=assistant_details,
        )
        return created_assistant
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.put("/{assistant_id}")
async def update_assistant(
    request: Request, assistant_id: str, userContext=Depends(auth_imbrace)
):
    try:
        body = await request.json()
        if body is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Request body cannot be empty",
            )
        # log.info(f"update_assistant called with body: {body}")
        assistant_details = validate_create_assistant_input(body)
        updated_assistant = await assistant_service.update(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_id=assistant_id,
            assistant_details=assistant_details,
        )
        if not updated_assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return updated_assistant
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.delete("/{assistant_id}")
async def remove_assistant(assistant_id: str, userContext=Depends(auth_imbrace)):
    try:
        if not assistant_id:
            raise HTTPException(status_code=400, detail="Assistant ID is required")

        log.info(f"remove_assistant called with assistant_id: {assistant_id}")
        deleted = await assistant_service.remove(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_id=assistant_id,
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return deleted
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.delete("")
async def bulk_remove_assistants(
    assistant_ids: List[str], userContext=Depends(auth_imbrace)
):
    try:
        deleted = await assistant_service.bulk_remove(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_ids=assistant_ids,
        )
        return {"deleted": assistant_ids}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


@router.patch("/{assistant_id}/instructions")
async def update_assistant_instructions(
    assistant_id: str, request: Request, userContext=Depends(auth_imbrace)
):
    try:
        body = await request.json()
        core_task = body.get("instructions", "")

        if not core_task:
            raise HTTPException(status_code=400, detail="core_task is required")

        log.info(
            f"update_assistant_instructions called with assistant_id: {assistant_id}, core_task: {core_task}"
        )

        updated_assistant = await assistant_service.update_assistant_instructions(
            organization_id=userContext["organization_id"],
            api_key=userContext["api_key"],
            assistant_id=assistant_id,
            core_task=core_task,
        )

        if not updated_assistant:
            raise HTTPException(status_code=404, detail="Assistant not found")

        return updated_assistant
    except Exception as e:
        log.error(f"Failed to update assistant instructions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )
