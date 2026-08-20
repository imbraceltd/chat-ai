"""Document store factory.

Reads DB_TYPE from the environment and returns the appropriate
BaseDocumentClient implementation:

    DB_TYPE=mongodb      → MongoDocumentClient  (default)
    DB_TYPE=postgresql   → PgDocumentClient

All logic/repository layers should import `document_client` from here
(or from the mongo_db shim which re-exports it).
"""

import os
import logging

from open_webui.internal.base_document_client import BaseDocumentClient

log = logging.getLogger(__name__)

DB_TYPE = os.getenv("DB_TYPE", "mongodb").lower()


def _create_client() -> BaseDocumentClient:
    if DB_TYPE == "postgresql":
        log.info("Document store: using PostgreSQL (PgDocumentClient)")
        from open_webui.internal.pg_document_client import PgDocumentClient
        return PgDocumentClient()
    else:
        log.info("Document store: using MongoDB (MongoDocumentClient)")
        from open_webui.internal.mongo_document_client import MongoDocumentClient
        return MongoDocumentClient()


# Singleton — all layers share this instance
document_client: BaseDocumentClient = _create_client()
