import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from open_webui.internal.mongo_db import mongodb_client, get_mongodb_session, OPENAI_DB_NAME
from open_webui.repository.assistant import AssistantRepository
from open_webui.models.ai_agents.rag import rag_model
from open_webui.env import SRC_LOG_LEVELS
from open_webui.models.ai_agents.files import files_model

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("MIGRATION", logging.INFO))

# Migration tracking collection
MIGRATION_COLLECTION = "migrations"
MIGRATION_NAME = "rag_to_openai_files_migration"

class RAGToOpenAIFilesMigration:
    """Migration class to transfer RAG data to openai_files collection."""
    
    def __init__(self):
        self.db_name = OPENAI_DB_NAME
        self.assistant_repo = AssistantRepository()
        
    async def _check_migration_completed(self) -> bool:
        """Check if migration has already been completed."""
        try:
            async with get_mongodb_session() as client:
                migration_record = await client.query_one(
                    database_name=self.db_name,
                    collection_name=MIGRATION_COLLECTION,
                    query={"migration_name": MIGRATION_NAME, "status": "completed"}
                )
                
                if migration_record:
                    log.info(f"Migration {MIGRATION_NAME} already completed at {migration_record.get('completed_at')}")
                    return True
                    
                return False
        except Exception as e:
            log.error(f"Error checking migration status: {e}")
            return False
    
    async def _mark_migration_started(self) -> None:
        """Mark migration as started."""
        try:
            async with get_mongodb_session() as client:
                migration_record = {
                    "migration_name": MIGRATION_NAME,
                    "status": "started",
                    "started_at": datetime.utcnow().isoformat(),
                    "assistants_processed": 0,
                    "files_migrated": 0
                }
                
                await client.insert(
                    database_name=self.db_name,
                    collection_name=MIGRATION_COLLECTION,
                    data=migration_record
                )
                log.info(f"Migration {MIGRATION_NAME} marked as started")
        except Exception as e:
            log.error(f"Error marking migration as started: {e}")
            raise
    
    async def _mark_migration_completed(self, stats: Dict[str, Any]) -> None:
        """Mark migration as completed with stats."""
        try:
            async with get_mongodb_session() as client:
                update_data = {
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                    "assistants_processed": stats.get("assistants_processed", 0),
                    "assistants_skipped_no_files": stats.get("assistants_skipped_no_files", 0),
                    "assistants_skipped_already_migrated": stats.get("assistants_skipped_already_migrated", 0),
                    "files_migrated": stats.get("files_migrated", 0),
                    "errors": stats.get("errors", [])
                }
                
                await client.update(
                    database_name=self.db_name,
                    collection_name=MIGRATION_COLLECTION,
                    query={"migration_name": MIGRATION_NAME},
                    data=update_data
                )
                log.info(f"Migration {MIGRATION_NAME} marked as completed")
        except Exception as e:
            log.error(f"Error marking migration as completed: {e}")
    
    async def _check_assistant_already_migrated(self, assistant_id: str, organization_id: str) -> bool:
        """Check if an assistant already has entries in openai_files collection."""
        try:
            async with get_mongodb_session() as client:
                existing_file = await client.query_one(
                    database_name=self.db_name,
                    collection_name="openai_files",
                    query={
                        "assistant_id": assistant_id,
                        "organization_id": organization_id
                    }
                )
                
                # query_one returns None if no document found, otherwise returns the document
                has_files = existing_file is not None
                if has_files:
                    log.debug(f"Assistant {assistant_id} already has files in openai_files collection")
                
                return has_files
                
        except Exception as e:
            log.error(f"Error checking if assistant {assistant_id} already migrated: {e}")
            # Return False to be safe - continue with migration if check fails
            return False
    
    async def _get_all_assistants(self) -> List[Dict[str, Any]]:
        """Get all assistants from the database across all organizations."""
        try:
            # In Postgres mode `openai_assistants` is a typed relational table
            # (no JSONB `data` column), so the generic document client's
            # `SELECT id, data` fails. Read via the typed repository instead;
            # keep the Mongo path for DB_TYPE=mongodb.
            from open_webui.internal.document_store import DB_TYPE

            if DB_TYPE == "postgresql":
                assistants = await AssistantRepository().get_all()
                log.info(f"Found {len(assistants)} assistants to process across all organizations")
                return assistants

            # Query all assistants directly using MongoDB client since we need all organizations
            async with get_mongodb_session() as client:
                query = {
                    "deleted_at": None  # Only get non-deleted assistants
                }

                assistants = await client.query(
                    database_name=self.db_name,
                    collection_name="openai_assistants",
                    query=query
                )

                log.info(f"Found {len(assistants)} assistants to process across all organizations")
                return assistants

        except Exception as e:
            log.error(f"Error getting assistants: {e}")
            raise
    
    async def _get_rag_files_for_assistant(self, assistant: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get RAG files for a specific assistant."""
        try:
            organization_id = assistant.get("organization_id", "")
            assistant_id = assistant.get("assistant_id", "")
            
            if not organization_id or not assistant_id:
                log.warning(f"Missing organization_id or assistant_id for assistant: {assistant.get('name', 'unknown')}")
                return []
            
            # Query RAG files using get_all function
            query = {
                "org_id": organization_id,
                "assistant_id": assistant_id
            }
            
            rag_files = await rag_model.get_all(query)
            log.info(f"Found {len(rag_files)} RAG files for assistant {assistant_id}")
            return rag_files
            
        except Exception as e:
            log.error(f"Error getting RAG files for assistant {assistant.get('assistant_id', 'unknown')}: {e}")
            return []
    
    async def _create_openai_files_entries(self, rag_files: List[Dict[str, Any]], assistant: Dict[str, Any]) -> int:
        """Create entries in openai_files collection from RAG files using files_model.create()."""
        try:
            if not rag_files:
                return 0
            
            migrated_count = 0
            
            for rag_file in rag_files:
                try:
                    # Map RAG file fields to files_model format (matching upload function exactly)
                    file_entry = {
                        "id": rag_file.get("file_id", rag_file.get("id")),  # Use file_id or fallback to id
                        "organization_id": rag_file.get("org_id", assistant.get("organization_id")),
                        "assistant_id": rag_file.get("assistant_id", assistant.get("assistant_id")),
                        "filename": rag_file.get("filename", rag_file.get("original_name", "unknown")),
                        "file_size": rag_file.get("bytes", 0),
                        "file_type": rag_file.get("blobType", "application/octet-stream"),
                        "board_id": rag_file.get("board_id", ""),
                        "boarditem_id": rag_file.get("boarditem_id", ""),
                        "url": rag_file.get("url", "")
                    }
                    
                    # Only migrate if we have essential fields
                    if not file_entry["id"] or not file_entry["filename"]:
                        log.warning(f"Skipping file with missing essential fields: {rag_file}")
                        continue
                    
                    # Use files_model.create() method (matching upload function approach)
                    
                    result = await files_model.create(file_entry)
                    
                    if result:
                        migrated_count += 1
                        log.debug(f"Successfully migrated file {file_entry['id']} for assistant {assistant.get('assistant_id', 'unknown')}")
                    else:
                        log.warning(f"Failed to migrate file {file_entry['id']} - files_model.create returned None")
                        
                except Exception as file_error:
                    log.error(f"Error migrating individual file {rag_file.get('file_id', 'unknown')}: {file_error}")
                    # Continue with next file instead of failing entire assistant
                    continue
            
            log.info(f"Successfully migrated {migrated_count} out of {len(rag_files)} files for assistant {assistant.get('assistant_id', 'unknown')}")
            return migrated_count
                
        except Exception as e:
            log.error(f"Error creating openai_files entries for assistant {assistant.get('assistant_id', 'unknown')}: {e}")
            raise
    
    async def run_migration(self) -> Dict[str, Any]:
        """Run the complete migration process."""
        try:
            # Check if migration already completed
            if await self._check_migration_completed():
                return {
                    "status": "already_completed",
                    "message": "Migration already completed"
                }
            
            # Mark migration as started
            await self._mark_migration_started()
            
            # Initialize stats
            stats = {
                "assistants_processed": 0,
                "assistants_skipped_no_files": 0,
                "assistants_skipped_already_migrated": 0,
                "files_migrated": 0,
                "errors": []
            }
            
            log.info("Starting RAG to openai_files migration...")
            
            # Step 1: Get all assistants
            assistants = await self._get_all_assistants()
            
            # Step 2: Process each assistant
            for assistant in assistants:
                try:
                    assistant_id = assistant.get("assistant_id", "")
                    file_ids = assistant.get("file_ids", [])
                    
                    # Check if assistant has file_ids
                    if not file_ids:
                        log.debug(f"Assistant {assistant_id} has no file_ids, skipping")
                        stats["assistants_skipped_no_files"] += 1
                        continue
                    
                    log.info(f"Processing assistant {assistant_id} with {len(file_ids)} file_ids")
                    
                    # Check if this assistant already has migrated files
                    organization_id = assistant.get("organization_id", "")
                    already_migrated = await self._check_assistant_already_migrated(assistant_id, organization_id)
                    if already_migrated:
                        log.info(f"Assistant {assistant_id} already has entries in openai_files collection, skipping")
                        stats["assistants_skipped_already_migrated"] += 1
                        continue
                    
                    # Step 3: Get RAG files for this assistant
                    rag_files = await self._get_rag_files_for_assistant(assistant)
                    
                    if rag_files:
                        # Step 4: Create openai_files entries
                        migrated_count = await self._create_openai_files_entries(rag_files, assistant)
                        stats["files_migrated"] += migrated_count
                    
                    stats["assistants_processed"] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing assistant {assistant.get('assistant_id', 'unknown')}: {str(e)}"
                    log.error(error_msg)
                    stats["errors"].append(error_msg)
                    continue
            
            # Mark migration as completed
            await self._mark_migration_completed(stats)
            
            log.info(f"Migration completed successfully. Processed {stats['assistants_processed']} assistants, "
                    f"migrated {stats['files_migrated']} files. "
                    f"Skipped {stats['assistants_skipped_no_files']} assistants with no files, "
                    f"skipped {stats['assistants_skipped_already_migrated']} assistants already migrated.")
            
            return {
                "status": "completed",
                "assistants_processed": stats["assistants_processed"],
                "assistants_skipped_no_files": stats["assistants_skipped_no_files"],
                "assistants_skipped_already_migrated": stats["assistants_skipped_already_migrated"],
                "files_migrated": stats["files_migrated"],
                "errors": stats["errors"]
            }
            
        except Exception as e:
            log.error(f"Migration failed: {e}")
            
            # Mark migration as failed
            try:
                async with get_mongodb_session() as client:
                    await client.update(
                        database_name=self.db_name,
                        collection_name=MIGRATION_COLLECTION,
                        query={"migration_name": MIGRATION_NAME},
                        data={
                            "status": "failed",
                            "failed_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                    )
            except Exception as mark_error:
                log.error(f"Error marking migration as failed: {mark_error}")
            
            raise

# Global migration instance
rag_migration = RAGToOpenAIFilesMigration()

async def run_rag_to_openai_files_migration() -> Dict[str, Any]:
    """Run the RAG to openai_files migration."""
    return await rag_migration.run_migration()

# Export the migration function
__all__ = [
    "RAGToOpenAIFilesMigration",
    "rag_migration",
    "run_rag_to_openai_files_migration"
]