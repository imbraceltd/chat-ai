from typing import Dict, Any

# It's good practice to define custom exceptions for better error handling.
import json
import logging

from open_webui.llm.agent import get_llm_provider


logger = logging.getLogger(__name__)


class InvalidAssistantInputError(ValueError):
    """Base exception for invalid assistant input."""

    pass


class InvalidAssistantNameError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's name."""

    pass


class InvalidAssistantDescriptionError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's description."""

    pass


class InvalidAssistantInstructionsError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's instructions."""

    pass


class InvalidAssistantFileIdsError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's file IDs."""

    pass


class InvalidAssistantFolderIdsError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's folder IDs."""

    pass


class InvalidAssistantMetadataError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's metadata."""

    pass


class InvalidAssistantModeError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's mode."""

    pass


class InvalidAssistantChannelError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's channel."""

    pass


class InvalidAssistantPersonalityAndRoleError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's personality and role."""

    pass


class InvalidAssistantCoreTaskError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's core task."""

    pass


class InvalidAssistantToneAndStyleError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's tone and style."""

    pass


class InvalidAssistantResponseLengthError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's response length."""

    pass


class InvalidAssistantBannedWordsError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's banned words."""

    pass


class InvalidAssistantIdsError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant IDs for bulk removal."""

    pass


class InvalidProviderInputError(ValueError):
    """Base exception for invalid provider input."""

    pass


class InvalidProviderNameError(InvalidProviderInputError):
    """Exception raised for errors in the provider's name."""

    pass


class InvalidProviderTypeError(InvalidProviderInputError):
    """Exception raised for errors in the provider's type."""

    pass


class InvalidProviderConfigError(InvalidProviderInputError):
    """Exception raised for errors in the provider's config."""

    pass

class InvalidAssistantTemperatureError(InvalidAssistantInputError):
    """Exception raised for errors in the assistant's temperature setting."""

    pass


def validate_create_assistant_input(body: dict) -> dict:
    """
    Validates the input for creating a new assistant.

    Args:
        body: A dictionary containing the assistant's configuration.

    Returns:
        A dictionary with the validated assistant data.

    Raises:
        InvalidAssistantInputError: If any of the validation checks fail.
    """
    if body is None:
        raise InvalidAssistantInputError("Request body cannot be None.")

    if not isinstance(body, dict):
        raise InvalidAssistantInputError("Request body must be a dictionary.")

    name = body.get("name")
    description = body.get("description")
    instructions = body.get("instructions")
    file_ids = body.get("file_ids")
    folder_ids = body.get("folder_ids")
    board_ids = body.get("board_ids")
    metadata = body.get("metadata")
    provider_id = body.get("provider_id")
    mode = body.get("mode")
    channel = body.get("channel")
    category = body.get("category")
    personality_role = body.get("personality_role")
    core_task = body.get("core_task")
    tone_and_style = body.get("tone_and_style")
    response_length = body.get("response_length")
    banned_words = body.get("banned_words")
    temperature = body.get("temperature", 0.1)
    use_memory = body.get("use_memory", True)
    default_folder_id = body.get("default_folder_id")
    vibe_coding = body.get("vibe_code", False)

    if not name or not isinstance(name, str) or len(name) > 256:
        raise InvalidAssistantNameError("Invalid assistant name.")

    if description and (not isinstance(description, str) or len(description) > 512):
        raise InvalidAssistantDescriptionError("Invalid assistant description.")

    if instructions and (
        not isinstance(instructions, str) or len(instructions) > 32768
    ):
        raise InvalidAssistantInstructionsError("Invalid assistant instructions.")

    if file_ids and not isinstance(file_ids, list):
        raise InvalidAssistantFileIdsError("Invalid assistant file IDs.")

    if folder_ids and not isinstance(folder_ids, list):
        raise InvalidAssistantFolderIdsError("Invalid assistant folder IDs.")

    if board_ids and not isinstance(board_ids, list):
        raise InvalidAssistantInputError("Invalid assistant board IDs.")

    if metadata and not isinstance(metadata, dict):
        raise InvalidAssistantMetadataError("Invalid assistant metadata.")

    if provider_id is not None and not isinstance(provider_id, str):
        raise InvalidAssistantInputError("Invalid provider_id - must be a string.")

    if mode and not isinstance(mode, str):
        raise InvalidAssistantModeError("Invalid assistant mode.")

    if channel and not isinstance(channel, str):
        raise InvalidAssistantChannelError("Invalid assistant channel.")

    if personality_role and not isinstance(personality_role, str):
        raise InvalidAssistantPersonalityAndRoleError(
            "Invalid assistant personality and role."
        )

    if core_task and not isinstance(core_task, str):
        raise InvalidAssistantCoreTaskError("Invalid assistant core task.")

    if tone_and_style and not isinstance(tone_and_style, str):
        raise InvalidAssistantToneAndStyleError("Invalid assistant tone and style.")

    if response_length and not isinstance(response_length, str):
        raise InvalidAssistantResponseLengthError("Invalid assistant response length.")

    if banned_words and not isinstance(banned_words, str):
        raise InvalidAssistantBannedWordsError("Invalid assistant banned words.")

    if temperature and not (
        isinstance(temperature, (int, float)) and (0 <= temperature <= 2)
    ):
        raise InvalidAssistantTemperatureError("Invalid assistant temperature.")

    if vibe_coding is not None and not isinstance(vibe_coding, bool):
        raise InvalidAssistantInputError("Invalid vibe_coding - must be a boolean.")

    result = {
        "name": name,
        "description": description,
        "instructions": instructions,
        "file_ids": file_ids,
        "folder_ids": folder_ids,
        "board_ids": board_ids,
        "metadata": metadata,
        "provider_id": provider_id,
        "mode": mode,
        "channel": channel,
        "category": category,
        "personality_role": personality_role,
        "core_task": core_task,
        "tone_and_style": tone_and_style,
        "response_length": response_length,
        "banned_words": banned_words,
        "temperature": temperature,
        "use_memory": use_memory,
        "default_folder_id": default_folder_id,
        "vibe_code": vibe_coding,
    }

    # Orchestrator hierarchy fields — pass through ONLY when explicitly present
    # in the body so ordinary edits (which omit them) keep their existing
    # preserve-on-absent behaviour. Mirrors how the webapp/marketplace model
    # multi-agent hierarchy via `sub_agents` + `agent_type`; `team_leads` is the
    # inverse relation, persisted server-side by reverse-writing `sub_agents`.
    if "sub_agents" in body or "subAgents" in body:
        sub_agents = body.get("sub_agents", body.get("subAgents"))
        if sub_agents is not None and not isinstance(sub_agents, list):
            raise InvalidAssistantInputError("Invalid sub_agents - must be a list.")
        result["sub_agents"] = sub_agents
    if "agent_type" in body:
        agent_type = body.get("agent_type")
        if agent_type is not None and not isinstance(agent_type, str):
            raise InvalidAssistantInputError("Invalid agent_type - must be a string.")
        result["agent_type"] = agent_type
    if "team_leads" in body:
        team_leads = body.get("team_leads")
        if team_leads is not None and not isinstance(team_leads, list):
            raise InvalidAssistantInputError("Invalid team_leads - must be a list.")
        result["team_leads"] = team_leads

    return result


def validate_create_assistant_app_input(body: dict) -> dict:
    """
    Validates the input for creating a new assistant app.
    """
    # Check if body is None or not a dictionary
    if body is None:
        raise InvalidAssistantInputError("Request body cannot be None.")

    if not isinstance(body, dict):
        raise InvalidAssistantInputError("Request body must be a dictionary.")

    name = body.get("name")
    description = body.get("description")
    instructions = body.get("instructions")
    file_ids = body.get("file_ids")
    folder_ids = body.get("folder_ids")
    board_ids = body.get("board_ids")
    metadata = body.get("metadata")
    provider_id = body.get("provider_id")
    workflow_name = body.get("workflow_name")
    credential_id = body.get("credential_id")
    credential_name = body.get("credential_name")
    mode = body.get("mode")
    channel = body.get("channel")
    category = body.get("category")
    personality_role = body.get("personality_role")
    core_task = body.get("core_task")
    tone_and_style = body.get("tone_and_style")
    response_length = body.get("response_length")
    banned_words = body.get("banned_words")
    knowledge_hubs = body.get("knowledge_hubs")
    workflow_function_call = body.get("workflow_function_call")
    model_id = body.get("model_id")
    show_thinking_process = body.get("show_thinking_process")
    preload_information = body.get("preload_information")
    preload_information_type = body.get("preload_information_type")
    guardrail_id = body.get("guardrail_id")
    agent_type = body.get("agent_type", "agent")
    sub_agents = body.get("sub_agents", [])
    streaming = body.get("streaming", False)
    temperature = body.get("temperature", 0.1)
    use_memory = body.get("use_memory", True)
    version = body.get("version", 1)
    document_ai = body.get("document_ai")
    vibe_coding = body.get("vibe_code", False)

    if not name or not isinstance(name, str) or len(name) > 256:
        raise InvalidAssistantNameError("Invalid assistant name.")

    if description and (not isinstance(description, str) or len(description) > 512):
        raise InvalidAssistantDescriptionError("Invalid assistant description.")

    if file_ids and not isinstance(file_ids, list):
        raise InvalidAssistantFileIdsError("Invalid assistant file IDs.")

    if folder_ids and not isinstance(folder_ids, list):
        raise InvalidAssistantFolderIdsError("Invalid assistant folder IDs.")

    if board_ids and not isinstance(board_ids, list):
        raise InvalidAssistantInputError("Invalid assistant board IDs.")

    if sub_agents and not isinstance(sub_agents, list):
        raise InvalidAssistantInputError("Invalid sub-agents input.")

    if metadata and not isinstance(metadata, dict):
        raise InvalidAssistantMetadataError("Invalid assistant metadata.")

    if provider_id is not None and not isinstance(provider_id, str):
        raise InvalidAssistantInputError("Invalid provider_id.")

    if (
        not workflow_name
        or not isinstance(workflow_name, str)
        or len(workflow_name) > 256
    ):
        raise InvalidAssistantNameError("Invalid workflow name.")

    if credential_id and not isinstance(credential_id, int):
        raise InvalidAssistantInputError("Invalid credential ID.")
    if credential_name and not isinstance(credential_name, str):
        raise InvalidAssistantInputError("Invalid credential name.")
    if mode and not isinstance(mode, str):
        raise InvalidAssistantModeError("Invalid assistant mode.")
    if channel and not isinstance(channel, str):
        raise InvalidAssistantChannelError("Invalid assistant channel.")
    if personality_role and not isinstance(personality_role, str):
        raise InvalidAssistantPersonalityAndRoleError(
            "Invalid assistant personality and role."
        )
    if core_task and not isinstance(core_task, str):
        raise InvalidAssistantCoreTaskError("Invalid assistant core task.")
    if tone_and_style and not isinstance(tone_and_style, str):
        raise InvalidAssistantToneAndStyleError("Invalid assistant tone and style.")
    if response_length and not isinstance(response_length, str):
        raise InvalidAssistantResponseLengthError("Invalid assistant response length.")
    if banned_words and not isinstance(banned_words, str):
        raise InvalidAssistantBannedWordsError("Invalid assistant banned words.")
    if temperature and not (
        isinstance(temperature, (int, float)) and (0 <= temperature <= 2)
    ):
        raise InvalidAssistantTemperatureError("Invalid assistant temperature.")
    if guardrail_id and not isinstance(guardrail_id, str):
        raise InvalidAssistantInputError("Invalid guardrail id.")

    if vibe_coding is not None and not isinstance(vibe_coding, bool):
        raise InvalidAssistantInputError("Invalid vibe_coding - must be a boolean.")

    if document_ai is not None:
        if not isinstance(document_ai, dict):
            raise InvalidAssistantInputError("Invalid document_ai configuration.")
        # board_id is intentionally NOT required: a document_ai assistant can be
        # created before its data board exists (the board is provisioned later via
        # the databoard _link-assistant flow, which needs the assistant_id first).
        required_doc_ai_fields = ["vlm_provider_id", "vlm_model", "source_languages"]
        for field in required_doc_ai_fields:
            if field not in document_ai:
                raise InvalidAssistantInputError(f"document_ai missing required field: {field}")
        if not isinstance(document_ai.get("source_languages"), list):
            raise InvalidAssistantInputError("document_ai.source_languages must be a list.")
        # Per-schema board settings are optional; when present they must be well-typed.
        # board_category_id = a databoard category id (or empty for the default);
        # board_deployment_access = a list of team ids (empty = all teams).
        for s in document_ai.get("schemas") or []:
            if not isinstance(s, dict):
                continue
            if "board_category_id" in s and not isinstance(s["board_category_id"], str):
                raise InvalidAssistantInputError(
                    "document_ai schema board_category_id must be a string."
                )
            if "board_deployment_access" in s and not isinstance(
                s["board_deployment_access"], list
            ):
                raise InvalidAssistantInputError(
                    "document_ai schema board_deployment_access must be a list."
                )

    return {
        "name": name,
        "description": description,
        "instructions": instructions,
        "fileIds": file_ids,
        "folder_ids": folder_ids,
        "board_ids": board_ids,
        "metadata": metadata,
        "provider_id": provider_id,
        "workflow_name": workflow_name,
        "credential_id": credential_id,
        "credential_name": credential_name,
        "mode": mode,
        "channel": channel,
        "category": category,
        "personality_role": personality_role,
        "core_task": core_task,
        "tone_and_style": tone_and_style,
        "response_length": response_length,
        "banned_words": banned_words,
        "knowledge_hubs": knowledge_hubs or [],
        "workflow_function_call": workflow_function_call or [],
        "model_id": model_id or "",
        "show_thinking_process": show_thinking_process or False,
        "preload_information": preload_information or "",
        "preload_information_type": preload_information_type or "",
        "guardrail_id": guardrail_id or "",
        "agent_type": agent_type,
        "subAgents": sub_agents,
        "streaming": streaming,
        "temperature": temperature,
        "use_memory": use_memory,
        "version": version,
        "default_folder_id": body.get("default_folder_id"),
        "document_ai": document_ai,
        "vibe_code": vibe_coding,
    }


def validate_bulk_remove_assistants_input(body: list) -> list:
    """
    Validates the input for bulk removing assistants.

    Args:
        body: A list of assistant IDs to be removed.

    Returns:
        The validated list of assistant IDs.

    Raises:
        InvalidAssistantIdsError: If the input is not a non-empty list.
    """
    if not body or not isinstance(body, list) or len(body) == 0:
        raise InvalidAssistantIdsError("Invalid assistant IDs for bulk removal.")

    return body


def validate_create_provider_input(body: dict) -> dict:
    """
    Validate input payload for creating a new LLM provider configuration.

    Expected payload:
    {
        "name": str,                 # display name
        "type": str,                 # provider type: openai|ollama|lmstudio|bedrock|vllm|...
        "config": dict,              # provider-specific configuration
        "metadata": dict | None      # optional extra data
    }
    """
    if body is None:
        raise InvalidProviderInputError("Request body cannot be None.")

    if not isinstance(body, dict):
        raise InvalidProviderInputError("Request body must be a dictionary.")

    name = body.get("name")
    provider_type = body.get("type")
    config = body.get("config")
    metadata = body.get("metadata", {})
    source = body.get("source", "custom")
    is_shown = body.get("is_shown", True)
    models = body.get("models", [])

    if not name or not isinstance(name, str) or len(name) > 256:
        raise InvalidProviderNameError("Invalid provider name.")

    if not provider_type or not isinstance(provider_type, str):
        raise InvalidProviderTypeError("Invalid provider type.")

    if config is None or not isinstance(config, dict):
        raise InvalidProviderConfigError("Invalid provider config.")

    # Optional fields
    if source not in ("custom", "system"):
        # Default to "custom" if invalid value is provided
        source = "custom"

    if not isinstance(is_shown, bool):
        # Default to True if invalid value is provided
        is_shown = True

    if models is None:
        models = []
    if not isinstance(models, list) or any(not isinstance(m, dict) for m in models):
        # Keep it simple for now – enforce list[dict] shape
        models = []

    return {
        "name": name,
        "type": provider_type,
        "config": config,
        "metadata": metadata if isinstance(metadata, dict) else {},
        # New fields for D75a
        "source": source,
        "is_shown": is_shown,
        "models": models,
    }


async def validate_provider_runtime_config(
    provider_type: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate provider configuration by attempting to fetch models from the
    underlying LLM provider implementation.

    If instantiation or fetching models fails, an InvalidProviderConfigError
    is raised. If it succeeds, the models dictionary is returned.
    """
    try:
        provider = get_llm_provider(provider_type, config)
        provider_models = await provider.get_models_from_provider()
        logger.info(f"[DEBUG-LLM] Provider models: {provider_models}")
        return provider_models
    except Exception as e:
        raise InvalidProviderConfigError(
            f"Unable to connect to LLM provider '{provider_type}' with the given config: {e}"
        )


def set_workflow_input(
    organization_id: str = "",
    workflow_name: str = "",
    text_expression: str = "",
    thread_id_expression: str = "",
    assistant_id: str = "",
    credential_id: str = "",
    credential_name: str = "",
    tags: list = None,
    method: str = "",
    send_message: bool = False,
) -> dict:
    """
    Constructs a workflow configuration dictionary.

    Args:
        organization_id: The ID of the organization.
        workflow_name: The name of the workflow.
        text_expression: The text expression.
        thread_id_expression: The thread ID expression.
        assistant_id: The ID of the assistant.
        credential_id: The ID of the credential.
        credential_name: The name of the credential.
        tags: A list of tags for the workflow.
        method: The HTTP method for the workflow.
        send_message: A boolean indicating whether to send a message.

    Returns:
        A dictionary representing the workflow configuration.
    """
    if tags is None:
        tags = []

    return {
        "name": workflow_name,
        "nodes": [
            {
                "parameters": {"icsTitle": "Start"},
                "name": "Start",
                "type": "n8n-nodes-base.start",
                "typeVersion": 1,
                "position": [60, 240],
            },
            {
                "parameters": {
                    "icsTitle": "Manage AI-Assistants",
                    "assistant_id": assistant_id,
                    "advanced": send_message,
                    "sendMessage": send_message,
                },
                "name": "Manage AI-Assistants",
                "type": "n8n-nodes-base.assistantRequest",
                "typeVersion": 1,
                "position": [240, 240],
            },
        ],
        "connections": {
            "Start": {
                "main": [[{"node": "Manage AI-Assistants", "type": "main", "index": 0}]]
            },
        },
        "settings": {},
        "tags": tags,
        "active": method == "POST",
    }


# Example usage:
workflow_details = {
    "workflow_name": "My New Assistant Workflow",
    "assistant_id": "asst_12345",
    "tags": ["testing", "ai"],
    "method": "POST",
    "send_message": True,
}

new_workflow = set_workflow_input(**workflow_details)
print(json.dumps(new_workflow, indent=2))
