import logging
from typing import List, Dict, Any, Optional

from open_webui.env import SRC_LOG_LEVELS
from open_webui.repository.parquet import parquet_repo

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("PARQUETS", logging.INFO))


class ParquetsModel:
    async def create(self, parquet_details: Dict[str, Any]) -> Dict[str, Any]:
        return await parquet_repo.create(parquet_details)

    async def get_by_filters(
        self,
        organization_id: Optional[str] = None,
        folder_ids: Optional[List[str]] = None,
        board_ids: Optional[List[str]] = None,
        boarditem_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await parquet_repo.get_by_filters(
            organization_id=organization_id,
            folder_ids=folder_ids,
            board_ids=board_ids,
            boarditem_id=boarditem_id,
        )

    async def get_by_id(self, parquet_id: str) -> Optional[Dict[str, Any]]:
        return await parquet_repo.get_by_id(parquet_id)

    async def update(
        self,
        parquet_id: str,
        update_data: Dict[str, Any],
        delete_old_s3_file: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if update_data.get("s3_path") and delete_old_s3_file:
            current = await parquet_repo.get_by_id(parquet_id)
            if current:
                old_path = current.get("s3_path")
                if old_path and old_path != update_data["s3_path"]:
                    try:
                        from open_webui.storage.provider import Storage
                        Storage.delete_file(old_path)
                    except Exception as e:
                        log.warning("Failed to delete old S3 file %s: %s", old_path, e)
        return await parquet_repo.update(parquet_id, update_data)

    async def delete(self, parquet_id: str, delete_s3_file: bool = True) -> Dict[str, Any]:
        record = await parquet_repo.get_by_id(parquet_id)
        if not record:
            return {"deleted_count": 0, "message": "Parquet record not found"}
        if delete_s3_file and record.get("s3_path"):
            try:
                from open_webui.storage.provider import Storage
                Storage.delete_file(record["s3_path"])
            except Exception as e:
                log.warning("Failed to delete S3 file %s: %s", record["s3_path"], e)
        count = await parquet_repo.soft_delete(parquet_id)
        return {"deleted_count": count}

    async def hard_delete(self, parquet_id: str) -> Dict[str, Any]:
        count = await parquet_repo.hard_delete(parquet_id)
        return {"deleted_count": count}

    async def get_all_active(self) -> List[Dict[str, Any]]:
        return await parquet_repo.get_all_active()


parquets_model = ParquetsModel()


async def create(parquet_details: Dict[str, Any]) -> Dict[str, Any]:
    return await parquets_model.create(parquet_details)

async def get_by_filters(
    organization_id: Optional[str] = None,
    folder_ids: Optional[List[str]] = None,
    board_ids: Optional[List[str]] = None,
    boarditem_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return await parquets_model.get_by_filters(organization_id, folder_ids, board_ids, boarditem_id)

async def get_by_id(parquet_id: str) -> Optional[Dict[str, Any]]:
    return await parquets_model.get_by_id(parquet_id)

async def update(parquet_id: str, update_data: Dict[str, Any], delete_old_s3_file: bool = False) -> Optional[Dict[str, Any]]:
    return await parquets_model.update(parquet_id, update_data, delete_old_s3_file)

async def delete(parquet_id: str, delete_s3_file: bool = True) -> Dict[str, Any]:
    return await parquets_model.delete(parquet_id, delete_s3_file)

async def hard_delete(parquet_id: str) -> Dict[str, Any]:
    return await parquets_model.hard_delete(parquet_id)

async def get_all_active() -> List[Dict[str, Any]]:
    return await parquets_model.get_all_active()


__all__ = [
    "ParquetsModel", "parquets_model",
    "create", "get_by_filters", "get_by_id",
    "update", "delete", "hard_delete", "get_all_active",
]
