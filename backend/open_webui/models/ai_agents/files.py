import logging
from typing import List, Dict, Any, Optional

from open_webui.env import SRC_LOG_LEVELS
from open_webui.repository.openai_file import openai_file_repo

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("FILES", logging.INFO))


class FilesModel:
    async def create(self, file_details: Dict[str, Any]) -> Dict[str, Any]:
        return await openai_file_repo.create(file_details)

    async def get_all(self, file_ids: List[str] = None) -> List[Dict[str, Any]]:
        return await openai_file_repo.get_all(file_ids or [])

    async def get_by_id(self, file_id: str) -> Optional[Dict[str, Any]]:
        return await openai_file_repo.get_by_id(file_id)

    async def update(self, file_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return await openai_file_repo.update(file_id, update_data)

    async def update_assistant_id(self, file_ids: List[str], assistant_id: str) -> Any:
        return await openai_file_repo.update_assistant_id(file_ids, assistant_id)

    async def delete(self, file_id: str) -> Dict[str, Any]:
        count = await openai_file_repo.soft_delete(file_id)
        return {"deleted_count": count}

    async def hard_delete(self, file_id: str) -> Dict[str, Any]:
        count = await openai_file_repo.hard_delete(file_id)
        return {"deleted_count": count}

    async def get_all_active(self) -> List[Dict[str, Any]]:
        return await openai_file_repo.get_all_active()

    async def search(self, search_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Direct column filter: any key in the query that matches a column
        org_id = search_query.get("org_id") or search_query.get("organization_id")
        if org_id:
            return await openai_file_repo.get_by_organization(org_id)
        return await openai_file_repo.get_all_active()

    async def get_by_organization(self, org_id: str) -> List[Dict[str, Any]]:
        return await openai_file_repo.get_by_organization(org_id)


files_model = FilesModel()


async def create(file_details: Dict[str, Any]) -> Dict[str, Any]:
    return await files_model.create(file_details)

async def get_all(file_ids: List[str] = None) -> List[Dict[str, Any]]:
    return await files_model.get_all(file_ids)

async def get_by_id(file_id: str) -> Optional[Dict[str, Any]]:
    return await files_model.get_by_id(file_id)

async def update(file_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return await files_model.update(file_id, update_data)

async def delete(file_id: str) -> Dict[str, Any]:
    return await files_model.delete(file_id)

async def hard_delete(file_id: str) -> Dict[str, Any]:
    return await files_model.hard_delete(file_id)

async def get_all_active() -> List[Dict[str, Any]]:
    return await files_model.get_all_active()

async def search(search_query: Dict[str, Any]) -> List[Dict[str, Any]]:
    return await files_model.search(search_query)

async def get_by_organization(org_id: str) -> List[Dict[str, Any]]:
    return await files_model.get_by_organization(org_id)

async def update_assistant_id(file_ids: List[str], assistant_id: str) -> Any:
    return await files_model.update_assistant_id(file_ids, assistant_id)


__all__ = [
    "FilesModel", "files_model",
    "create", "get_all", "get_by_id", "update", "update_assistant_id",
    "delete", "hard_delete", "get_all_active", "search", "get_by_organization",
]
