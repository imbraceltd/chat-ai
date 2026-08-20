import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from open_webui.utils.auth import auth_imbrace
from open_webui.constants import ERROR_MESSAGES
from open_webui.llm.utils.knowledge_hub import (
    KnowledgeHubService,
    create_knowledge_hub_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class TagGenerationRequest(BaseModel):
    folder_name: str
    description: str


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


async def get_knowledge_hub_service():
    """Dependency to provide KnowledgeHubService instance."""
    async with create_knowledge_hub_service() as service:
        yield service


@router.post("/tag-generation", response_model=Dict[str, Any])
async def generate_tags(
    body: TagGenerationRequest,
    knowledge_hub_service: KnowledgeHubService = Depends(get_knowledge_hub_service),
    ctx=Depends(auth_imbrace),
):
    """Generate tags based on folder name and description."""
    try:
        context = {
            "folder_name": body.folder_name,
            "description": body.description,
            "organization_id": ctx.get("organization_id", ""),
        }
        result = await knowledge_hub_service.generate_tags(context)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tag generation failed",
            )

        return result
    except Exception as error:
        message, status_code = handle_controller_error(error)
        raise HTTPException(status_code=status_code, detail=message)


__all__ = ["router"]
