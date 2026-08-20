from ..repository.assistant import AssistantRepository

# Assuming you have this utility
from ..utils.misc import convert_date_to_timestamp, generate_ai_assistant_instruction
from typing import Dict, List
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from open_webui.config import OPENAI_API_KEY
import logging
from open_webui.env import SRC_LOG_LEVELS
from open_webui.repository.workflow.ai import get_all, update_workflow_name, update
from ..models.ai_agents.assistants import AssistantResponse
from ..repository.api_key import ApiKeyRepository
from ..repository.rag import RAGRepository
from open_webui.models.ai_agents.files import files_model
from ..utils.board_embedding_job import create_board_embedding_jobs_for_assistant
from ..utils.schema_link import (
    provision_document_ai_boards,
    collect_owned_board_ids,
    drop_unowned_board_ids,
    carry_over_board_ids,
    unlink_removed_document_ai_schemas,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])


# open_webui/services/assistant_service.py
assistant_repo = AssistantRepository()
api_key_repo = ApiKeyRepository()
rag_repo = RAGRepository()

log = logging.getLogger(__name__)


async def create(
    organization_id: str = "", api_key: str = "", assistant_details: Dict = None
):
    if not api_key:
        api_key = OPENAI_API_KEY

    api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

    try:
        log.info(f"assistant_details: {assistant_details}")

        if not assistant_details.get("mode"):
            # Use constant if available
            assistant_details["mode"] = "advanced"

        id = str(uuid.uuid4())
        assistant_details["id"] = id
        assistant_details["assistant_id"] = id
        assistant_details["file_ids"] = assistant_details.get("fileIds", [])
        assistant_details["folder_ids"] = assistant_details.get("folder_ids", [])
        assistant_details["sub_agents"] = assistant_details.get("subAgents", [])
        # Set to your LLM model if needed
        assistant_details["model"] = "rag"

        if "fileIds" in assistant_details:
            del assistant_details["fileIds"]
        if "folder_ids" in assistant_details and not assistant_details["folder_ids"]:
            del assistant_details["folder_ids"]
        if "subAgents" in assistant_details:
            del assistant_details["subAgents"]

        # Generate custom instructions if not provided
        if not assistant_details.get("instructions"):
            assistant_details["instructions"] = generate_ai_assistant_instruction(
                personality_role=assistant_details.get("personality_role"),
                core_task=assistant_details.get("core_task"),
                tone_and_style=assistant_details.get("tone_and_style"),
                response_length=assistant_details.get("response_length"),
                banned_words=assistant_details.get("banned_words"),
                other_requirements=(assistant_details.get("metadata") or {}).get(
                    "other_requirements"
                ),
            )

        # For Document-AI assistants, provision the data boards from the bundled
        # schemas and link them to this assistant (id is already generated above),
        # folding each board_id back into document_ai before persisting so it is
        # saved in a single write. Best-effort: a link failure won't block create.
        #
        # A brand-new assistant owns no boards yet, so drop any board_id the FE
        # prefilled (copied from the schema picker / another assistant) — each
        # assistant gets its own board per schema, and a stale board_id would
        # otherwise skip linking and leave this assistant off the schema's
        # agent_ids.
        drop_unowned_board_ids(assistant_details.get("document_ai"), set())
        await provision_document_ai_boards(
            organization_id, id, assistant_details.get("document_ai")
        )

        created_assistant = await assistant_repo.create(
            organization_id=organization_id, assistant_details=assistant_details
        )

        if created_assistant:
            for key in ["_id", "updated_at", "deleted_at", "assistant_id"]:
                if key in created_assistant:
                    del created_assistant[key]

            if assistant_details.get("file_ids"):
                await rag_repo.updateAssistantId(
                    organization_id,
                    assistant_details["file_ids"],
                    created_assistant["id"],
                )
                await files_model.update_assistant_id(
                    file_id=assistant_details["file_ids"],
                    assistant_id=created_assistant["id"],
                )

        return created_assistant
    except Exception as error:
        raise error


async def update(
    organization_id: str = "",
    api_key: str = "",
    assistant_id: str = "",
    assistant_details: Dict = None,
    workflow_details: Dict = None,
):

    if not api_key:
        api_key = OPENAI_API_KEY

    api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

    try:

        if not assistant_details.get("mode"):
            assistant_details["mode"] = "standard"

        current_assistant = (
            await assistant_repo.get_by_assistant_id_and_organization_id(
                organization_id, assistant_id
            )
        )

        if not current_assistant:
            raise ValueError("Assistant not found")

        if assistant_details.get("metadata"):
            # Ensure current_assistant has a metadata field
            if not current_assistant.get("metadata"):
                current_assistant["metadata"] = {}
            current_assistant["metadata"].update(assistant_details["metadata"])

        # Create custom instructions if not provided

        assistant_details["instructions"] = generate_ai_assistant_instruction(
            personality_role=assistant_details.get("personality_role"),
            core_task=assistant_details.get("core_task"),
            tone_and_style=assistant_details.get("tone_and_style"),
            response_length=assistant_details.get("response_length"),
            banned_words=assistant_details.get("banned_words"),
            other_requirements=(assistant_details.get("metadata") or {}).get(
                "other_requirements"
            ),
        )

        # Capture the boards this assistant already owns + its full document_ai
        # BEFORE the update below overwrites document_ai: the owned-board set
        # tells its own boards apart from a board_id copied in via the payload,
        # and the old document_ai lets us detect which schemas were removed.
        prev_owned_board_ids = collect_owned_board_ids(
            current_assistant.get("document_ai")
        )
        prev_document_ai = current_assistant.get("document_ai")

        # Update assistant fields

        current_assistant.update(
            {
                "name": assistant_details["name"],
                "description": assistant_details["description"],
                "instructions": assistant_details["instructions"],
                "file_ids": assistant_details.get("fileIds") if assistant_details.get("fileIds") is not None else current_assistant.get("file_ids", []),
                "folder_ids": assistant_details.get("folder_ids") if assistant_details.get("folder_ids") is not None else current_assistant.get("folder_ids", []),
                "board_ids": assistant_details.get("board_ids") if assistant_details.get("board_ids") is not None else current_assistant.get("board_ids", []),
                "mode": assistant_details["mode"],
                "streaming": assistant_details.get("streaming", False),
                "agent_type": assistant_details.get("agent_type", "agent"),
                "sub_agents": assistant_details.get("subAgents", []),
                "channel": assistant_details.get(
                    "channel", current_assistant.get("channel")
                ),
                "category": assistant_details.get(
                    "category", current_assistant.get("category", [])
                ),
                "personality_role": assistant_details.get(
                    "personality_role", current_assistant.get("personality_role")
                ),
                "core_task": assistant_details.get(
                    "core_task", current_assistant.get("core_task")
                ),
                "tone_and_style": assistant_details.get(
                    "tone_and_style", current_assistant.get("tone_and_style")
                ),
                "response_length": assistant_details.get(
                    "response_length", current_assistant.get("response_length")
                ),
                "banned_words": assistant_details.get(
                    "banned_words", current_assistant.get("banned_words")
                ),
                "knowledge_hubs": assistant_details.get(
                    "knowledge_hubs", current_assistant.get("knowledge_hubs", [])
                ),
                "workflow_function_call": assistant_details.get(
                    "workflow_function_call",
                    current_assistant.get("workflow_function_call", []),
                ),
                "model_id": assistant_details.get(
                    "model_id", current_assistant.get("model_id")
                ),
                "show_thinking_process": assistant_details.get(
                    "show_thinking_process", False
                ),
                "preload_information": assistant_details.get(
                    "preload_information", False
                ),
                "guardrail_id": assistant_details.get(
                    "guardrail_id", current_assistant.get("guardrail_id")
                ),
                "temperature": assistant_details.get(
                    "temperature", current_assistant.get("temperature", 0.5)
                ),
                "use_memory": assistant_details.get("use_memory", True),
                "vibe_code": assistant_details.get(
                    "vibe_code", current_assistant.get("vibe_code", False)
                ),
                "default_folder_id": assistant_details.get("default_folder_id"),
                # Persist provider_id if provided, otherwise keep existing value
                "provider_id": assistant_details.get(
                    "provider_id", current_assistant.get("provider_id")
                ),
                "document_ai": assistant_details.get(
                    "document_ai", current_assistant.get("document_ai")
                ),
            }
        )

        # Provision boards for any newly-added Document-AI schemas before
        # persisting, folding board_ids into document_ai. Already-linked schemas
        # (those with a board_id) are skipped to avoid duplicate boards.
        #
        # First drop board_ids this assistant doesn't actually own — a schema
        # added to this assistant may arrive carrying another assistant's
        # board_id; without this it would skip linking and never land in the
        # schema's agent_ids. Stripping it provisions a fresh per-assistant
        # board and links the agent.
        drop_unowned_board_ids(
            current_assistant.get("document_ai"), prev_owned_board_ids
        )
        # Then restore board_ids the assistant already owns but that the incoming
        # payload omitted (e.g. the channel service re-saving to wire its workflow
        # resends schemas without board_id). This stops a second board being
        # provisioned per schema on that redundant update — the real fix for the
        # create+update duplicate-board issue.
        carry_over_board_ids(
            current_assistant.get("document_ai"), prev_document_ai
        )
        await provision_document_ai_boards(
            organization_id, assistant_id, current_assistant.get("document_ai")
        )

        # Mirror the reverse direction: any schema dropped from document_ai must
        # be unlinked from this assistant in data-board (agent_ids + its board),
        # otherwise the schema keeps listing an assistant that no longer uses it.
        await unlink_removed_document_ai_schemas(
            organization_id,
            assistant_id,
            prev_document_ai,
            current_assistant.get("document_ai"),
        )

        updated_assistant = await assistant_repo.update(assistant_id, current_assistant)

        if updated_assistant:
            for key in ["_id", "updated_at", "assistant_id"]:
                updated_assistant.pop(key, None)

            if updated_assistant.get("file_ids"):
                await rag_repo.updateAssistantId(
                    organization_id,
                    updated_assistant["file_ids"],
                    updated_assistant["id"],
                )
                await files_model.update_assistant_id(
                    file_id=updated_assistant["file_ids"],
                    assistant_id=updated_assistant["id"],
                )

            # Create board embedding jobs if board_ids are provided
            # The create_board_embedding_jobs_for_assistant function will check for existing jobs
            new_board_ids = assistant_details.get("board_ids", [])

            if new_board_ids:
                log.info(
                    f"Creating board embedding jobs for assistant {assistant_id} with board_ids: {new_board_ids}"
                )
                try:
                    created_job_ids = await create_board_embedding_jobs_for_assistant(
                        organization_id=organization_id,
                        assistant_id=assistant_id,
                        board_ids=new_board_ids,
                    )
                    if created_job_ids:
                        log.info(
                            f"Created {len(created_job_ids)} board embedding jobs for updated assistant {assistant_id}"
                        )
                    else:
                        log.info(
                            f"No new board embedding jobs needed for assistant {assistant_id} (jobs may already exist)"
                        )
                except Exception as job_error:
                    log.error(
                        f"Error creating board embedding jobs for updated assistant {assistant_id}: {job_error}"
                    )
                    # Don't fail the assistant update if job creation fails
            else:
                log.info(
                    f"No board_ids provided for assistant {assistant_id}, skipping job creation"
                )

        return updated_assistant
    except Exception as error:
        raise error


async def remove(
    organization_id: str = "",
    api_key: str = "",
    assistant_id: str = "",
    disable_check_api_key: bool = False,
    workflows: List = None,
):
    if not disable_check_api_key:
        api_key = OPENAI_API_KEY

    api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

    try:
        assistant = await assistant_repo.get_by_assistant_id_and_organization_id(
            organization_id, assistant_id
        )
        log.info(f"assistant: {assistant}")

        if not assistant:
            raise ValueError("Assistant not found")

        await assistant_repo.remove(assistant_id)

        await rag_repo.remove_assistant_id(organization_id, assistant_id)

        return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}
    except Exception as error:
        raise error
