import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from open_webui.config import LLM_CONFIG
from open_webui.repository.assistant import AssistantRepository

logger = logging.getLogger(__name__)

assistant_repo = AssistantRepository()

# Default system prompt - equivalent to ragSystemPrompt from constants/prompt module
RAG_SYSTEM_PROMPT = """You are an AI assistant answering questions based on retrieved context from knowledge_hub.
    {context}
    You MUST provide the best answer based on the context
    Date & Time: {datetime}     
    Current Conversation ID: {thread_id}
     Also you have access to many other tools to serve many other purposes."""


def build_assistant_prompt(base_prompt: str, instructions: Optional[str] = None) -> str:
    """
    Build assistant prompt by combining base prompt with instructions
    Python equivalent of buildAssistantPrompt

    Args:
        base_prompt: Base system prompt
        instructions: Additional instructions to append

    Returns:
        Combined prompt string
    """
    return f"{base_prompt}{instructions or ''}"


async def enrich_assistant_inputs(
    organization_id: str, model: str, inputs: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Enrich assistant inputs with assistant details and defaults
    Python equivalent of enrichAssistantInputs

    Args:
        organization_id: Organization ID
        model: Model name
        inputs: Input parameters

    Returns:
        Tuple of (enriched_inputs, final_model)
    """
    assistant_prompt = RAG_SYSTEM_PROMPT or ""

    # Default enriched inputs structure
    enriched_inputs = {
        "question": "",
        "assistant_id": "",
        "file_ids": [],
        "name": "",
        "instructions": "",
        "thread_id": str(uuid.uuid4()),
        "board_id": "",
        "board_ids": [],
        "conversation_id": "",
        "message_id": "",
        "message_created_at": "",
        "created_at": "",
        "knowledge_hubs": [],
        "workflow_function_call": [],
        "tools": [],
        "show_thinking_process": False,
        "preload_information": "",
        "guardrail_id": "",
        "streaming": False,
        "sub_agents": [],
        "temperature": 0.1,
        "use_memory": True,
        "agent_type": "",
        "version": 1,
        **(inputs or {}),  # Overwrite defaults if provided
    }

    # Ensure thread_id exists
    if inputs and inputs.get("thread_id"):
        enriched_inputs["thread_id"] = inputs["thread_id"]

    # Early return if no question
    if not enriched_inputs.get("question"):
        return {"thread_id": enriched_inputs["thread_id"], "message": ""}, model

    # Enrich with assistant details if assistant_id is provided
    if enriched_inputs.get("assistant_id"):
        try:
            # This would typically call a repository service
            # assistant_details = await repository.assistantV2.getByAssistantIdAndOrganizationId(
            #     organization_id, enriched_inputs["assistant_id"]
            # )

            # Placeholder for assistant details - replace with actual repository call
            assistant_details = await _get_assistant_details(
                organization_id, enriched_inputs["assistant_id"]
            )

            logging.debug(
                f"Enriching inputs with assistant details: {assistant_details}"
            )

            if assistant_details:
                # Update enriched inputs with assistant details
                if assistant_details.get("instructions"):
                    enriched_inputs["instructions"] = assistant_details["instructions"]

                # if assistant_details.get("created_at"):
                #     created_at = assistant_details["created_at"]
                #     logging.info(f"{created_at}")
                #     enriched_inputs["created_at"] = assistant_details["created_at"]

                if assistant_details.get("workflow_function_call"):
                    enriched_inputs["workflow_function_call"] = assistant_details[
                        "workflow_function_call"
                    ]

                if assistant_details.get("preload_information"):
                    enriched_inputs["preload_information"] = assistant_details[
                        "preload_information"
                    ]

                if assistant_details.get("name"):
                    enriched_inputs["name"] = assistant_details["name"]

                # Set file_ids if not already provided
                if (
                    assistant_details.get("file_ids")
                    and len(assistant_details["file_ids"]) > 0
                    and (
                        not enriched_inputs.get("file_ids")
                        or len(enriched_inputs["file_ids"]) == 0
                    )
                ):
                    enriched_inputs["file_ids"] = assistant_details["file_ids"]

                # Set knowledge_hubs if not already provided
                if (
                    assistant_details.get("knowledge_hubs")
                    and len(assistant_details["knowledge_hubs"]) > 0
                    and (
                        not enriched_inputs.get("knowledge_hubs")
                        or len(enriched_inputs["knowledge_hubs"]) == 0
                    )
                ):
                    enriched_inputs["knowledge_hubs"] = assistant_details[
                        "knowledge_hubs"
                    ]

                # Set board_ids if not already provided
                if (assistant_details.get("board_ids")
                        and len(assistant_details["board_ids"]) > 0
                        and (not enriched_inputs.get("board_ids")
                             or len(enriched_inputs["board_ids"]) == 0)
                        ):
                    enriched_inputs["board_ids"] = assistant_details["board_ids"]

                # Set folder_ids if not already provided
                if (
                    assistant_details.get("folder_ids")
                    and len(assistant_details["folder_ids"]) > 0
                    and (
                        not enriched_inputs.get("folder_ids")
                        or len(enriched_inputs["folder_ids"]) == 0
                    )
                ):
                    enriched_inputs["folder_ids"] = assistant_details["folder_ids"]

                if assistant_details.get("show_thinking_process"):
                    enriched_inputs["show_thinking_process"] = assistant_details[
                        "show_thinking_process"
                    ]

                if assistant_details.get("guardrail_id"):
                    enriched_inputs["guardrail_id"] = assistant_details["guardrail_id"]

                if assistant_details.get("streaming"):
                    enriched_inputs["streaming"] = assistant_details["streaming"]

                if assistant_details.get("sub_agents"):
                    logging.info(
                        f"Sub-agents found: {assistant_details['sub_agents']}")
                    enriched_inputs["sub_agents"] = assistant_details["sub_agents"]

                if assistant_details.get("agent_type"):
                    enriched_inputs["agent_type"] = assistant_details["agent_type"]

                if assistant_details.get("temperature"):
                    enriched_inputs["temperature"] = assistant_details["temperature"]

                if assistant_details.get("use_memory"):
                    enriched_inputs["use_memory"] = assistant_details["use_memory"]

                if assistant_details.get("visualization"):
                    enriched_inputs["visualization"] = assistant_details[
                        "visualization"
                    ]
                if assistant_details.get("version"):
                    enriched_inputs["version"] = assistant_details["version"]

                # Copy provider_id from assistant if not already provided in inputs
                if assistant_details.get("provider_id") and not enriched_inputs.get(
                    "provider_id"
                ):
                    enriched_inputs["provider_id"] = assistant_details["provider_id"]

                # Copy metadata from assistant if not already provided in inputs
                if assistant_details.get("metadata") and not enriched_inputs.get(
                    "meta_data"
                ):
                    enriched_inputs["meta_data"] = assistant_details["metadata"]

                # Handle model selection
                default_model = _get_default_model()
                if (
                    not model
                    or model == "Default"
                    or (isinstance(model, list) and len(model) == 0)
                ):
                    model = assistant_details.get("model_id") or default_model

                if (
                    model == "Default"
                    and assistant_details.get("model_id") == "Default"
                ):
                    model = default_model

        except Exception as e:
            logger.error(f"Error enriching assistant inputs: {e}")

    # Ensure instructions have fallback
    if not enriched_inputs.get("instructions"):
        enriched_inputs["instructions"] = assistant_prompt

    return enriched_inputs, model


def build_filters(inputs: Dict[str, Any], organization_id: str) -> Dict[str, Any]:
    """
    Build MongoDB filters for search queries
    Python equivalent of buildFilters

    Args:
        inputs: Input parameters
        organization_id: Organization ID

    Returns:
        MongoDB filter dictionary
    """
    filters = {"org_id": {"$eq": organization_id}}
    or_conditions = []

    # Scope precedence: the NARROWEST explicit scope wins. When the caller selects
    # specific file_ids (or folder_ids), we filter ONLY by that scope instead of
    # OR-ing it with the broad assistant_id branch (which matches the assistant's
    # entire knowledge base and would widen the candidate set right back). Keeping
    # the candidate set small lets Postgres use the file_id/folder_id btree for an
    # exact KNN scan rather than an approximate-index post-filter — much faster and
    # 100% recall for the "few files, many pages" case.
    if inputs.get("file_ids") and len(inputs["file_ids"]) > 0:
        or_conditions.append({"file_id": {"$in": inputs["file_ids"]}})
    elif inputs.get("folder_ids") and len(inputs["folder_ids"]) > 0:
        or_conditions.append({"folder_id": {"$in": inputs["folder_ids"]}})
    else:
        # No explicit file/folder scope: fall back to the broad assistant + board scope.
        if inputs.get("assistant_id"):
            or_conditions.append({"assistant_id": {"$eq": inputs["assistant_id"]}})

        if inputs.get("knowledge_hubs") and len(inputs["knowledge_hubs"]) > 0:
            or_conditions.append({"board_id": {"$in": inputs["knowledge_hubs"]}})
        elif inputs.get("board_ids") and len(inputs["board_ids"]) > 0:
            or_conditions.append({"board_id": {"$in": inputs["board_ids"]}})
        elif inputs.get("board_id"):
            or_conditions.append({"board_id": {"$eq": inputs["board_id"]}})

    if len(or_conditions) > 0:
        filters["$or"] = or_conditions

    return filters


def build_search_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build search filter clauses for Atlas Search
    Python equivalent of buildSearchFilters

    Args:
        filters: MongoDB filter dictionary

    Returns:
        List of search filter clauses
    """
    filter_clauses = []

    for key, value in filters.items():
        if key == "$or" and isinstance(value, list):
            # Translate $or to should clauses
            should_clauses = []

            for condition in value:
                for field, cond in condition.items():
                    if isinstance(cond, dict) and "$eq" in cond:
                        should_clauses.append(
                            {"equals": {"path": field, "value": cond["$eq"]}}
                        )
                    elif isinstance(cond, dict) and "$in" in cond:
                        should_clauses.append(
                            {"in": {"path": field, "value": cond["$in"]}}
                        )

            if should_clauses:
                filter_clauses.append({"compound": {"should": should_clauses}})
        else:
            # Normal filters like org_id: { $eq: ... }
            if isinstance(value, dict) and "$eq" in value:
                filter_clauses.append(
                    {"equals": {"path": key, "value": value["$eq"]}})

    return filter_clauses


def merge_results(
    vector_docs: List[Dict[str, Any]], full_text_docs: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """
    Merge vector search and full-text search results using Reciprocal Rank Fusion (RRF)
    Python equivalent of mergeResults

    Args:
        vector_docs: Vector search results
        full_text_docs: Full-text search results
        k: Number of top results to return

    Returns:
        List of merged and ranked documents
    """
    rank_constant = 60  # Smoothing constant for RRF
    doc_scores = {}

    # Process vector search results
    for index, doc in enumerate(vector_docs):
        doc_id = str(doc["metadata"]["_id"])
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {**doc, "combined_score": 0}
        doc_scores[doc_id]["combined_score"] += 1 / (index + 1 + rank_constant)

    # Process full-text search results
    for index, doc in enumerate(full_text_docs):
        doc_id = str(doc["metadata"]["_id"])
        if doc_id not in doc_scores:
            doc_scores[doc_id] = {**doc, "combined_score": 0}
        doc_scores[doc_id]["combined_score"] += 1 / (index + 1 + rank_constant)

    # Convert to list and sort by combined score
    merged_results = list(doc_scores.values())
    merged_results.sort(key=lambda x: x["combined_score"], reverse=True)

    return merged_results[:k]


async def get_assistants(assistant_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get sub-agent details for a list of assistant IDs

    Args:
        assistant_ids: List of assistant IDs

    Returns:
        List of sub-agent details
    """
    try:
        return await assistant_repo.get_by_assistant_ids(assistant_ids)
    except Exception as e:
        logger.error(f"Error fetching sub-agents: {e}")
        return []


# Helper functions (placeholders for actual implementations)
async def _get_assistant_details(
    organization_id: str, assistant_id: str
) -> Optional[Dict[str, Any]]:
    """
    Get assistant details from repository
    Placeholder for actual repository call

    Args:
        organization_id: Organization ID
        assistant_id: Assistant ID

    Returns:
        Assistant details dictionary or None
    """
    try:
        # This would typically call a repository service
        return await assistant_repo.get_by_assistant_id_and_organization_id(
            organization_id, assistant_id
        )
    except Exception as e:
        logger.error(f"Error fetching assistant details: {e}")
        return None


def _get_default_model() -> str:
    """
    Get default model configuration
    Placeholder for actual config retrieval

    Returns:
        Default model name
    """
    # This would typically come from config.defaultModel
    return LLM_CONFIG.get("defaultModel", "gpt-4o") or "gpt-4o"


# Utility functions for data validation and transformation
def validate_inputs(inputs: Dict[str, Any]) -> bool:
    """
    Validate input parameters

    Args:
        inputs: Input parameters to validate

    Returns:
        True if inputs are valid, False otherwise
    """
    if not inputs:
        return False

    # Basic validation
    if not inputs.get("question", "").strip():
        return False

    return True


def normalize_file_ids(file_ids: Any) -> List[str]:
    """
    Normalize file IDs to ensure they're a list of strings

    Args:
        file_ids: File IDs in various formats

    Returns:
        List of file ID strings
    """
    if not file_ids:
        return []

    if isinstance(file_ids, str):
        return [file_ids]

    if isinstance(file_ids, list):
        return [str(fid) for fid in file_ids if fid]

    return []


def normalize_knowledge_hubs(knowledge_hubs: Any) -> List[str]:
    """
    Normalize knowledge hubs to ensure they're a list of strings

    Args:
        knowledge_hubs: Knowledge hubs in various formats

    Returns:
        List of knowledge hub strings
    """
    if not knowledge_hubs:
        return []

    if isinstance(knowledge_hubs, str):
        return [knowledge_hubs]

    if isinstance(knowledge_hubs, list):
        return [str(hub) for hub in knowledge_hubs if hub]

    return []


def extract_metadata_for_sources(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract metadata for source references

    Args:
        docs: List of document objects

    Returns:
        List of source metadata
    """
    sources = []

    for doc in docs:
        metadata = doc.get("metadata", {})
        source = {
            "filename": metadata.get("original_name", "Unknown"),
            "url": metadata.get("url", ""),
            "score": doc.get("combined_score", 0.0),
        }

        # Avoid duplicate sources
        if not any(s["url"] == source["url"] for s in sources):
            sources.append(source)

    return sources
