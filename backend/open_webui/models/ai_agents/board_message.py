import logging
from typing import Optional, Dict, Any

from open_webui.env import SRC_LOG_LEVELS
from open_webui.repository.board_message import board_message_repo
from pydantic import BaseModel, ConfigDict

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MODELS", logging.INFO))


class BoardMessageForm(BaseModel):
    conversation_id: str = ""
    message_id: str = ""
    board_item_id: str = ""
    created_at: str = ""


class BoardMessageModel:
    async def create(
        self,
        organization_id: str = "",
        details: BoardMessageForm = None,
    ) -> Optional[Dict[str, Any]]:
        if not details:
            details = BoardMessageForm()
        return await board_message_repo.create(
            conversation_id=details.conversation_id,
            board_item_id=details.board_item_id,
            organization_id=organization_id,
            created_at=details.created_at,
        )

    async def get_by_conversation_id(
        self,
        organization_id: str = "",
        conversation_id: str = "",
        created_at: str = "",
    ) -> Optional[Dict[str, Any]]:
        return await board_message_repo.get_closest_by_conversation_id(
            conversation_id=conversation_id,
            organization_id=organization_id,
            created_at=created_at,
        )


board_message_model = BoardMessageModel()
