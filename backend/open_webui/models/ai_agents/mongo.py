# open_webui/db/mongo.py

from pymongo import MongoClient
from typing import Optional
import logging

# Assuming these are defined in your config/env files
from open_webui.config import MONGODB_URL, DATABASE_NAME
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MODELS", "INFO"))

log.info(f"MONGODB_URL: {MONGODB_URL}")
log.info(f"DATABASE_NAME: {DATABASE_NAME}")


class MongoDB:
    """
    A singleton class to manage the MongoDB connection.
    The connection is established on application startup and closed on shutdown
    via the FastAPI lifespan event handler.
    """
    client: Optional[MongoClient] = None
    db = None

    @classmethod
    def connect_to_database(cls):
        """Establishes the connection to the MongoDB database."""
        log.info("Connecting to MongoDB...")
        if cls.client is None:
            try:
                # Connect directly using the MongoDB URL
                cls.client = MongoClient(str(MONGODB_URL))
                cls.db = cls.client[str(DATABASE_NAME)]
                # You can ping the server to confirm the connection
                cls.client.admin.command('ping')
                log.info("Successfully connected to MongoDB.")
            except Exception as e:
                log.error(f"Could not connect to MongoDB: {e}")
                raise
        else:
            log.info("MongoDB connection already established.")

    @classmethod
    def close_database_connection(cls):
        """Closes the existing MongoDB connection."""
        log.info("Closing MongoDB connection...")
        if cls.client:
            cls.client.close()
            cls.client = None
            cls.db = None
            log.info("MongoDB connection closed.")

    @classmethod
    def get_db(cls):
        """
        Returns the database instance.
        Establishes connection if not already connected.
        """
        if cls.db is None:
            cls.connect_to_database()
        return cls.db


# Collection names
ASSISTANTS_COLLECTION = "openai_assistants"


def get_assistants_collection():
    """Returns the assistants collection from the database."""
    return MongoDB.get_db()[ASSISTANTS_COLLECTION]
