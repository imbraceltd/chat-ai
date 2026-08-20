import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from open_webui.utils.auth import auth_imbrace
from open_webui.constants import ERROR_MESSAGES
from open_webui.llm.utils.websearch import WebSearch, create_web_search_service

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class WebSearchRequest(BaseModel):
    data: str
    prompt: Optional[str] = None


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


async def get_web_search_service():
    """Dependency to provide WebSearch instance."""
    async with create_web_search_service() as service:
        yield service


@router.post("/company", response_model=Dict[str, Any])
async def search_company_details(
    body: WebSearchRequest,
    web_search_service: WebSearch = Depends(get_web_search_service),
    ctx=Depends(auth_imbrace),
):
    """Search for company details using web search."""
    try:
        context = {"data": body.data}
        result = await web_search_service.search_company_details(context, body.prompt)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Company details not found or search failed",
            )

        return result
    except Exception as error:
        message, status_code = handle_controller_error(error)
        raise HTTPException(status_code=status_code, detail=message)


__all__ = ["router"]
