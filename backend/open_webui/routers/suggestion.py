import logging
import json
import re

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mongodb import MongoDBChatMessageHistory

from open_webui.config import MONGODB_CONFIG
from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.auth import auth_imbrace
from open_webui.utils.assistant import get_by_id as assistant_service_get_by_id
from open_webui.utils.provider import get_by_id as provider_service_get_by_id
from open_webui.llm.agent import (
    get_llm_provider,
    get_all_available_models,
    get_all_available_models_from_custom_provider,
    CONFIG,
    PROVIDER,
)
from open_webui.llm.utils.models import detect_provider
from open_webui.llm.utils.assistant import _get_default_model

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("SUGGESTIONS", logging.INFO))

router = APIRouter()

MAX_HISTORY_MESSAGES = 10

SUGGESTION_SYSTEM_PROMPT = """You are a helpful assistant that analyzes conversation history and the AI assistant's instructions to generate follow-up suggestions for the user.

IMPORTANT: Suggestions must be written from the USER's perspective — they are things the user would say or request TO the AI agent. They should sound like natural user messages: requests, instructions, or questions directed at the AI assistant. Do NOT generate questions that the AI would ask the user.

You will be provided with:
- Custom instructions (if available) — additional guidelines or constraints for generating suggestions. Always follow these when provided.
- The AI assistant's instructions (if available) — these describe the assistant's role, capabilities, and domain expertise. Use them to generate suggestions that are relevant to what the assistant can actually do.
- The conversation history — use this to understand the current context, what has already been discussed, and what logical next steps the user might take.

Based on these inputs, do the following:
1. Write a brief follow-up message (1-2 sentences) that summarizes the conversation so far and suggests a next action or direction.
2. Generate 3 to 5 concise follow-up suggestions. Each suggestion must be phrased as a request or question the USER would send to the AI agent to continue the conversation.

Guidelines for suggestions:
- Write them as direct user requests or commands (e.g., "Show me...", "Explain how...", "Help me with...", "Generate a...", "List the...")
- They should be actionable tasks or questions that the AI agent can fulfill based on its capabilities and instructions
- Base them on the conversation context — suggest logical next steps, deeper dives, or related topics from what was already discussed
- Do NOT write questions that ask the user for their preferences or input — the AI agent cannot ask the user questions as suggestions

Return your response as a JSON object with exactly these keys:
- "follow_up_message": a string with the summary and suggested next action
- "suggestions": an array of strings

Example:
{"follow_up_message": "We discussed the quarterly sales report and identified key trends. You might want to explore the underperforming regions next.", "suggestions": ["Show me a breakdown of sales by region for this quarter", "Compare Q1 and Q2 performance side by side", "Analyze the factors behind the decline in the Western region", "Generate a summary report of the top-performing product categories"]}"""


############################
# Models
############################


class SuggestionRequest(BaseModel):
    provider_id: Optional[str] = Field(
        None, description="Provider ID ('system' or custom UUID). Defaults to system provider from env."
    )
    model_id: Optional[str] = Field(
        None, description="Model ID to use for generation. Defaults to system default model from env."
    )
    assistant_id: Optional[str] = Field(
        None, description="AI assistant ID (reserved for future use)"
    )
    thread_id: str = Field(..., description="Thread/session ID for chat history")
    custom_instructions: Optional[str] = Field(
        None, description="Custom instructions to guide suggestion generation"
    )


class SuggestionResponse(BaseModel):
    success: bool
    follow_up_message: str
    suggestions: List[str]


############################
# Helpers
############################


def format_messages_for_prompt(messages) -> str:
    """Convert LangChain message objects into readable conversation text."""
    formatted = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if not content:
            continue
        if role == "human":
            formatted.append(f"User: {content}")
        elif role == "ai":
            formatted.append(f"Assistant: {content}")
        else:
            formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


async def resolve_llm(provider_id: str, model_id: str, organization_id: str):
    """Resolve a LangChain LLM instance from provider_id and model_id."""
    if provider_id and provider_id != "system":
        custom_provider = await provider_service_get_by_id(
            organization_id=organization_id, provider_id=provider_id
        )
        if not custom_provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider {provider_id} not found",
            )
        provider_config = custom_provider.get("config") or {}
        all_models = get_all_available_models_from_custom_provider(custom_provider)
    else:
        provider_config = CONFIG
        all_models = await get_all_available_models(config=CONFIG)

    detected = detect_provider(
        model_id=model_id, models=all_models, default_provider=PROVIDER
    )
    log.info(f"Resolved provider type: {detected} for model: {model_id}")
    provider = get_llm_provider(detected, provider_config)
    return provider.create_llm(model_id, all_models, streaming=False, temperature=0.7)


def parse_llm_json(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code fences."""
    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code fences or embedded JSON
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("Failed to parse JSON from LLM response")


############################
# Get Suggestions
############################


@router.post("", response_model=SuggestionResponse)
async def get_suggestions(
    body: SuggestionRequest,
    userContext=Depends(auth_imbrace),
):
    try:
        organization_id = userContext.get("organization_id", "")

        # 1. Fetch chat history (last N messages)
        chat_history = MongoDBChatMessageHistory(
            connection_string=MONGODB_CONFIG.get("openai_host"),
            database_name=MONGODB_CONFIG.get("openai_db_name", "openai_db"),
            collection_name="history",
            session_id=body.thread_id,
        )
        messages = await chat_history.aget_messages()
        last_messages = messages

        if not last_messages:
            return SuggestionResponse(
                success=True, follow_up_message="", suggestions=[]
            )

        # 2. Fetch assistant instructions if assistant_id is provided
        assistant_instructions = ""
        if body.assistant_id:
            try:
                api_key = userContext.get("api_key", "")
                assistant = await assistant_service_get_by_id(
                    organization_id=organization_id,
                    api_key=api_key,
                    assistant_id=body.assistant_id,
                )
                if assistant:
                    assistant_instructions = assistant.get("instructions", "")
            except Exception as e:
                log.warning(f"Failed to fetch assistant {body.assistant_id}: {e}")

        # 3. Resolve LLM (fall back to system defaults if not provided)
        provider_id = body.provider_id or "system"
        model_id = body.model_id or _get_default_model()
        llm = await resolve_llm(provider_id, model_id, organization_id)

        # 4. Build prompt
        conversation_text = format_messages_for_prompt(last_messages)
        context_parts = [f"Conversation history:\n{conversation_text}"]
        if assistant_instructions:
            context_parts.insert(
                0,
                f"AI Assistant instructions:\n{assistant_instructions}\n",
            )
        if body.custom_instructions:
            context_parts.insert(
                0,
                f"Custom instructions:\n{body.custom_instructions}\n",
            )
        context_parts.append(
            "\nGenerate a follow-up message and suggestions as JSON."
        )
        user_prompt = "\n".join(context_parts)

        # 5. Invoke LLM (non-streaming)
        response = await llm.ainvoke(
            [
                SystemMessage(content=SUGGESTION_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        # 5. Parse response
        parsed = parse_llm_json(response.content)
        follow_up_message = parsed.get("follow_up_message", "")
        suggestions = parsed.get("suggestions", [])

        return SuggestionResponse(
            success=True,
            follow_up_message=follow_up_message,
            suggestions=suggestions,
        )

    except HTTPException:
        raise
    except ValueError as e:
        log.warning(f"Failed to parse LLM response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse LLM response",
        )
    except Exception as e:
        log.exception("Error generating suggestions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
