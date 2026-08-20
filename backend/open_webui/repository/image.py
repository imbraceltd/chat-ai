# open_webui/repositories/image_repository.py

from typing import Dict, Any, Optional
import logging
from datetime import datetime
from open_webui.internal.mongo_db import (
    mongodb_client,
    get_mongodb_session,
    OPENAI_DB_NAME,
)
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

IMAGES_COLLECTION = "images"


class ImageRepository:
    def __init__(self):
        self.db_client = mongodb_client
        self.db_name = OPENAI_DB_NAME
        self.collection_name = IMAGES_COLLECTION

    async def get_by_image_id(self, organization_id: str, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves an image from the database by image_id and organization_id.
        """
        try:
            query = {
                "organization_id": organization_id,
                "image_id": image_id
            }

            async with get_mongodb_session() as client:
                data = await client.query_one(
                    database_name=self.db_name,
                    collection_name=self.collection_name,
                    query=query,
                )

                return data

        except Exception as error:
            log.error(f"Error getting image by id: {error}")
            raise error

    async def create(self, organization_id: str, image_details: Dict) -> Optional[Dict[str, Any]]:
        """
        Creates a new image record in the database.
        """
        try:
            image_details["organization_id"] = organization_id
            image_details["created_at"] = datetime.utcnow()
            image_details["updated_at"] = datetime.utcnow()
            image_details["deleted_at"] = None

            log.info(f"Creating image with details: {image_details}")

            async with get_mongodb_session() as client:
                result = await client.insert(
                    database_name=self.db_name,
                    collection_name=self.collection_name,
                    data=image_details,
                )

                if result:
                    log.info(f"Successfully created image")
                    return result
                else:
                    log.error("Failed to create image")
                    return None

        except Exception as error:
            log.error(f"Error creating image: {error}", exc_info=True)
            raise error

    async def update(self, image_id: str, image_details: Dict) -> Optional[Dict[str, Any]]:
        """
        Updates an image record in the database.
        """
        try:
            image_details["updated_at"] = datetime.utcnow()

            log.info(
                f"Updating image {image_id} with details: {image_details}")

            async with get_mongodb_session() as client:
                result = await client.update(
                    database_name=self.db_name,
                    collection_name=self.collection_name,
                    query={"image_id": image_id, "deleted_at": None},
                    data=image_details,
                )

                if result:
                    log.info(f"Successfully updated image {image_id}")
                    return result
                else:
                    log.warning(f"No image updated for id: {image_id}")
                    return None

        except Exception as error:
            log.error(f"Error updating image: {error}", exc_info=True)
            raise error

    async def remove(self, image_id: str) -> bool:
        """
        Soft deletes an image record in the database.
        """
        try:
            update_data = {
                "deleted_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            log.info(f"Soft deleting image {image_id}")

            async with get_mongodb_session() as client:
                result = await client.update(
                    database_name=self.db_name,
                    collection_name=self.collection_name,
                    query={"image_id": image_id, "deleted_at": None},
                    data=update_data,
                )

                if result:
                    log.info(f"Successfully deleted image {image_id}")
                    return True
                else:
                    log.warning(f"No image deleted for id: {image_id}")
                    return False

        except Exception as error:
            log.error(f"Error deleting image: {error}", exc_info=True)
            raise error
