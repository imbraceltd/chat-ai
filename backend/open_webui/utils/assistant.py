from ..repository.assistant import AssistantRepository

# Assuming you have this utility
from ..utils.misc import generate_ai_assistant_instruction
from typing import Dict, List
from typing import List, Dict
import uuid
from open_webui.config import OPENAI_API_KEY
import logging
from open_webui.env import SRC_LOG_LEVELS
from open_webui.repository.workflow.ai import get_all, update_workflow_name, update
from ..repository.api_key import ApiKeyRepository
from ..repository.rag import RAGRepository
from open_webui.models.ai_agents.files import files_model
from ..utils.board_embedding_job import create_board_embedding_jobs_for_assistant

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])


# open_webui/services/assistant_service.py
assistant_repo = AssistantRepository()
api_key_repo = ApiKeyRepository()
rag_repo = RAGRepository()

log = logging.getLogger(__name__)


async def get_all(
    organization_id: str = "", api_key: str = "", filter_options: Dict = None
):
    # Set default values for pagination and search
    limit = filter_options.get("limit")
    skip = filter_options.get("skip")
    search = filter_options.get("search")

    log.info(f"Fetching all assistants with filters: {limit}, {skip}, {search}")

    if not api_key:
        api_key = OPENAI_API_KEY

    api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

    try:
        # Pass pagination and search parameters to the repository
        db_assistants = await assistant_repo.get_all_by_organization_id(
            organization_id=organization_id, search=search, skip=skip, limit=limit
        )

        log.info(f"Found {len(db_assistants['data'])} assistants in the database.")

        # Convert ObjectId to string in the response
        for assistant in db_assistants["data"]:
            if "_id" in assistant:
                assistant["_id"] = str(assistant["_id"])
            if "assistant_id" in assistant:
                assistant["assistant_id"] = str(assistant["assistant_id"])

        return db_assistants

    except Exception as error:
        log.error(f"Error fetching assistants: {error}")
        raise error


async def get_by_id(
    organization_id: str = "", api_key: str = "", assistant_id: str = ""
):
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        assistant = await assistant_repo.get_by_assistant_id_and_organization_id(
            organization_id=organization_id, assistant_id=assistant_id
        )

        if not assistant:
            raise ValueError("Assistant not found")

        if "_id" in assistant:
            assistant["_id"] = str(assistant["_id"])

        # If this is an agent, find parent assistants that have this agent as a sub_agent
        if assistant and assistant.get("agent_type") == "agent":
            try:
                parent_assistants = (
                    await assistant_repo.get_by_sub_agent_id_and_organization_id(
                        organization_id=organization_id, sub_agent_id=assistant_id
                    )
                )

                # Format parent assistants information
                formatted_parent_assistants = []
                for parent in parent_assistants:
                    formatted_parent_assistants.append(
                        {
                            "assistant_id": parent.get("assistant_id", ""),
                            "name": parent.get("name", ""),
                        }
                    )

                assistant["team_leads"] = formatted_parent_assistants
                log.info(
                    f"Found {len(formatted_parent_assistants)} parent assistants for agent {assistant_id}"
                )

            except Exception as e:
                log.error(
                    f"Error finding parent assistants for agent {assistant_id}: {e}"
                )
                # Set empty list if there's an error
                assistant["team_leads"] = []

        # Check if assistant has sub_agents and populate their details
        if (
            assistant
            and assistant.get("sub_agents")
            and len(assistant["sub_agents"]) > 0
        ):
            try:
                # Get detailed information for sub-agents
                sub_agent_details = (
                    await assistant_repo.get_by_assistant_ids_and_organization_id(
                        organization_id=organization_id,
                        assistant_ids=assistant["sub_agents"],
                    )
                )

                # Reformat sub_agents with detailed information
                formatted_sub_agents = []
                for sub_agent in sub_agent_details:
                    formatted_sub_agents.append(
                        {
                            "assistant_id": sub_agent.get("assistant_id", ""),
                            "name": sub_agent.get("name", ""),
                            "instructions": sub_agent.get("core_task", ""),
                        }
                    )

                assistant["sub_agents"] = formatted_sub_agents
                log.info(
                    f"Populated {len(formatted_sub_agents)} sub-agents for assistant {assistant_id}"
                )

            except Exception as e:
                log.error(
                    f"Error populating sub-agents for assistant {assistant_id}: {e}"
                )
                # Keep the original sub_agents list if there's an error

        return assistant
    except Exception as error:
        raise error


async def get_by_name(organization_id: str = "", api_key: str = "", name: str = ""):
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        assistant = await assistant_repo.get_by_assistant_name_and_organization_id(
            name=name, organization_id=organization_id
        )

        return assistant
    except Exception as error:
        raise error


async def get_agents(organization_id: str, assistant_id: str = None):
    try:
        agents = await assistant_repo.get_agents(
            organization_id=organization_id, assistant_id=assistant_id
        )

        # Convert ObjectId to string in the response
        for agent in agents:
            if "_id" in agent:
                agent["_id"] = str(agent["_id"])
            if "assistant_id" in agent:
                agent["assistant_id"] = str(agent["assistant_id"])

        log.info(f"Found {len(agents)} agents for organization {organization_id}")
        return agents

    except Exception as error:
        log.error(f"Error getting agents: {error}")
        raise error


def _coerce_assistant_id(entry):
    """Accept an assistant id as a plain string or as a dict carrying
    `assistant_id`/`id` (the webapp sends objects; the CLI sends ids)."""
    if isinstance(entry, str):
        return entry or None
    if isinstance(entry, dict):
        return entry.get("assistant_id") or entry.get("id")
    return None


def _normalize_assistant_id_list(value):
    """Normalize a list of assistant references to a deduped list of ids."""
    if not isinstance(value, list):
        return []
    out = []
    for entry in value:
        cid = _coerce_assistant_id(entry)
        if cid and cid not in out:
            out.append(cid)
    return out


async def link_assistant_to_team_leads(organization_id, assistant_id, team_leads):
    """Persist `team_leads` by reverse-writing the canonical `sub_agents` relation.

    The orchestrator hierarchy is modeled everywhere (webapp create via the
    use-case/template flow, webapp edit via assistant_apps, marketplace) by the
    PARENT's `sub_agents`; `team_leads` is the derived inverse, computed on read.
    So "this assistant's team_leads are Y" is stored by adding this assistant to
    each Y's `sub_agents`. Idempotent; skips unknown ids and self-references so
    it never corrupts an unrelated assistant.
    """
    tl_ids = _normalize_assistant_id_list(team_leads)
    for tl_id in tl_ids:
        if tl_id == assistant_id:
            continue
        try:
            parent = await assistant_repo.get_by_assistant_id_and_organization_id(
                organization_id, tl_id
            )
            if not parent:
                log.warning(
                    f"team_lead {tl_id} not found in org {organization_id}; skipping link"
                )
                continue
            subs = list(parent.get("sub_agents") or [])
            if assistant_id not in subs:
                subs.append(assistant_id)
                await assistant_repo.update(tl_id, {"sub_agents": subs})
                log.info(
                    f"Linked assistant {assistant_id} into team_lead {tl_id} sub_agents"
                )
        except Exception as link_error:
            log.error(
                f"Failed to link assistant {assistant_id} to team_lead {tl_id}: {link_error}"
            )


async def create(
    organization_id: str = "", api_key: str = "", assistant_details: Dict = None
):
    if not api_key:
        api_key = OPENAI_API_KEY

    api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

    try:
        # if not api_key:
        #     api_key = OPENAI_API_KEY

        log.info(f"assistant_details: {assistant_details}")

        if not assistant_details.get("mode"):
            # Use constant if available
            assistant_details["mode"] = "standard"

        id = str(uuid.uuid4())
        assistant_details["id"] = id
        assistant_details["assistant_id"] = id
        assistant_details["file_ids"] = assistant_details.get("fileIds", [])
        assistant_details["folder_ids"] = assistant_details.get("folder_ids", [])
        assistant_details["board_ids"] = assistant_details.get("board_ids", [])
        # Normalize sub_agents (children) to a deduped id list when provided.
        if "sub_agents" in assistant_details:
            assistant_details["sub_agents"] = _normalize_assistant_id_list(
                assistant_details.get("sub_agents")
            )
        # Set to your LLM model if needed
        assistant_details["model"] = "rag"

        if "fileIds" in assistant_details:
            del assistant_details["fileIds"]
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

        created_assistant = await assistant_repo.create(
            organization_id=organization_id, assistant_details=assistant_details
        )

        if created_assistant:
            for key in ["_id", "updated_at", "deleted_at", "assistant_id"]:
                if key in created_assistant:
                    del created_assistant[key]

            # team_leads (parents) — reverse-write into each parent's sub_agents.
            if "team_leads" in assistant_details:
                await link_assistant_to_team_leads(
                    organization_id,
                    created_assistant["id"],
                    assistant_details.get("team_leads"),
                )

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

            # Create board embedding jobs if board_ids are provided
            if (
                assistant_details.get("board_ids")
                and len(assistant_details["board_ids"]) > 0
            ):
                try:
                    created_job_ids = await create_board_embedding_jobs_for_assistant(
                        organization_id=organization_id,
                        assistant_id=created_assistant["id"],
                        board_ids=assistant_details["board_ids"],
                    )
                    if created_job_ids:
                        log.info(
                            f"Created {len(created_job_ids)} board embedding jobs for assistant {created_assistant['id']}"
                        )
                except Exception as job_error:
                    log.error(
                        f"Error creating board embedding jobs for assistant {created_assistant['id']}: {job_error}"
                    )
                    # Don't fail the assistant creation if job creation fails

        return created_assistant
    except Exception as error:
        log.error(f"Error creating assistant: {error}")
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
            (current_assistant["metadata"] or {}).update(assistant_details["metadata"])

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

        # sub_agents (children): honor the body ONLY when explicitly provided
        # (snake from the validator, or legacy camel); otherwise preserve the
        # existing value so ordinary edits never wipe an orchestrator's children.
        if "sub_agents" in assistant_details:
            sub_agents_value = _normalize_assistant_id_list(
                assistant_details.get("sub_agents")
            )
        elif "subAgents" in assistant_details:
            sub_agents_value = _normalize_assistant_id_list(
                assistant_details.get("subAgents")
            )
        else:
            sub_agents_value = current_assistant.get("sub_agents", [])

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
                    "preload_information"
                ) or current_assistant.get("preload_information"),
                "streaming": assistant_details.get("streaming", False),
                "agent_type": assistant_details.get(
                    "agent_type", current_assistant.get("agent_type", "agent")
                ),
                "sub_agents": sub_agents_value,
                "guardrail_id": assistant_details.get(
                    "guardrail_id", current_assistant.get("guardrail_id", None)
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
            }
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

            # Create board embedding jobs if board_ids are provided and different from current
            new_board_ids = assistant_details.get("board_ids", [])
            current_board_ids = current_assistant.get("board_ids", [])

            if new_board_ids and set(new_board_ids) != set(current_board_ids):
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
                except Exception as job_error:
                    log.error(
                        f"Error creating board embedding jobs for updated assistant {assistant_id}: {job_error}"
                    )
                    # Don't fail the assistant update if job creation fails

            # team_leads (parents) — reverse-write into each parent's sub_agents.
            if "team_leads" in assistant_details:
                await link_assistant_to_team_leads(
                    organization_id,
                    assistant_id,
                    assistant_details.get("team_leads"),
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

        # Update workflow name if it's used in assistant
        if (assistant.get("metadata") or {}).get("workflow_id"):
            assistant_workflow_id = assistant["metadata"]["workflow_id"]
            current_workflow = None

            if not workflows:
                workflows = await get_all(organization_id) or []

            # Prepare workflow mappings
            workflow_name_to_ids = {}
            workflow_id_to_name = {}

            for workflow in workflows:
                workflow_name = workflow.get("name", "")
                workflow_id = workflow.get("id")

                if workflow_id == assistant_workflow_id:
                    current_workflow = workflow

                # Remove number suffix from workflow name
                import re

                base_name = re.sub(r"- \[\d+\]$", "", workflow_name)

                if base_name and workflow_id:
                    if base_name not in workflow_name_to_ids:
                        workflow_name_to_ids[base_name] = []
                    workflow_name_to_ids[base_name].append(workflow_id)

                    if workflow_id == assistant_workflow_id:
                        workflow_id_to_name[workflow_id] = base_name

            if assistant_workflow_id in workflow_id_to_name:
                from datetime import datetime

                date_str = datetime.now().strftime("%m/%d/%y")
                base_name = workflow_id_to_name[assistant_workflow_id]

                new_name = (
                    f"{base_name} - {date_str}"
                    if base_name.startswith("Disconnected")
                    else f"Disconnected - {base_name} - {date_str}"
                )

                if new_name in workflow_name_to_ids:
                    new_name += f" - [{len(workflow_name_to_ids[new_name])}]"

                if current_workflow:
                    current_workflow["name"] = new_name
                    current_workflow["active"] = False
                    await update_workflow_name(organization_id, current_workflow)

                    await update(
                        assistant_workflow_id,
                        {
                            "workflow_id": assistant_workflow_id,
                            "workflow_name": new_name,
                            "assistant_id": assistant_id,
                            "organization_id": organization_id,
                        },
                    )

        return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}
    except Exception as error:
        raise error


async def bulk_remove(
    organization_id: str = "", api_key: str = "", assistant_ids: List[str] = None
):
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        assistants = await assistant_repo.get_by_assistant_ids_and_organization_id(
            organization_id, assistant_ids
        )

        if len(assistants) != len(assistant_ids):
            raise ValueError("Invalid assistant id")

        workflows = await get_all(organization_id) or []

        for assistant_id in assistant_ids:
            await remove(organization_id, api_key, assistant_id, True, workflows)

        return assistant_ids
    except Exception as error:
        raise error


async def update_assistant_instructions(
    organization_id: str = "",
    api_key: str = "",
    assistant_id: str = "",
    core_task: str = "",
):
    try:
        if not api_key:
            api_key = OPENAI_API_KEY

        api_key = await api_key_repo.check_key_is_paid(organization_id, api_key)

        # Verify assistant exists and belongs to organization
        existing_assistant = (
            await assistant_repo.get_by_assistant_id_and_organization_id(
                organization_id=organization_id, assistant_id=assistant_id
            )
        )

        if not existing_assistant:
            raise ValueError("Assistant not found")

        # Generate new instructions using the core_task and existing assistant data
        instructions = generate_ai_assistant_instruction(
            personality_role=existing_assistant.get("personality_role", ""),
            core_task=core_task,
            tone_and_style=existing_assistant.get("tone_and_style", ""),
            response_length=existing_assistant.get("response_length", ""),
            banned_words=existing_assistant.get("banned_words", ""),
            other_requirements=(existing_assistant.get("metadata") or {}).get(
                "other_requirements", []
            ),
        )

        # Update instructions using repository method
        updated_assistant = await assistant_repo.update_assistant_instructions(
            assistant_id=assistant_id, instructions=instructions, core_task=core_task
        )

        if updated_assistant:
            # Clean up response by removing internal fields
            if "_id" in updated_assistant:
                updated_assistant["_id"] = str(updated_assistant["_id"])

            # Remove internal fields that shouldn't be exposed
            for key in ["updated_at", "deleted_at"]:
                if key in updated_assistant:
                    del updated_assistant[key]

        log.info(f"Successfully updated instructions for assistant {assistant_id}")
        return updated_assistant

    except Exception as error:
        log.error(f"Error updating assistant instructions: {error}")
        raise error
