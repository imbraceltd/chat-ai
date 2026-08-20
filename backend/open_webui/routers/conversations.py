import logging
import json

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.auth import auth_imbrace

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("CONVERSATIONS", logging.INFO))

router = APIRouter()


############################
# Models
############################


class ConversationSummaryResponse(BaseModel):
    summary: str
    conversation_id: str
    importance: str
    sentiment: str
    organization_id: str


class SummarizeRequest(BaseModel):
    data: List[Dict[str, Any]]


############################
# Summarize Conversations
############################


SUMMARY_PROMPT_TEMPLATE = """
You are given a conversation history between a customer and an agent.

**Task:**
1. Write a concise summary (max 20 words) of the given conversation history between the customer and agent. Please use customer's name in the summary
2. Use the provided dynamic **settings** to assign the following:
   - **Importance** level: based on urgency, intent, and keywords
   - **Sentiment** level: based on user tone and emotional state

**Keyword Settings:**
{settings}

**Conversation History:**
{messages}

**Customer Name:**
{customer}

**Organization ID:**
{organization_id}

**Notes:**
- Please focus on the customer's needs and how the agent addressed them.
- If the conversation requires immediate attention, please suggest an action for the agent.
- Use the **settings** object to determine valid importance and sentiment levels and their associated keywords.
- Classify the conversation based on keyword matches and contextual understanding.

Respond with ONLY a JSON object (no markdown, no prose) with exactly these keys:
"summary", "conversation_id", "importance", "sentiment", "organization_id".
"""


# JSON schema for the structured summary output. Passed to
# invoke_with_structured_output so capable providers (OpenAI, recent Bedrock
# models) can enforce it natively; others fall back to prompt-based JSON.
SUMMARY_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "conversation_id": {"type": "string"},
        "importance": {"type": "string"},
        "sentiment": {"type": "string"},
        "organization_id": {"type": "string"},
    },
    "required": [
        "summary",
        "conversation_id",
        "importance",
        "sentiment",
        "organization_id",
    ],
    "additionalProperties": False,
}


def _build_summary_llm(api_key: Optional[str] = None):
    """Build the summary LLM through the shared provider factory so it follows
    the same OpenAI->Bedrock migration as the rest of the app (see echart.py).

    Provider follows LLM_DEFAULT_PROVIDER (PROVIDER); the openai branch is kept
    for the legacy per-request api_key path. Models come from existing app-wide
    env only — no summary-specific vars:
      - bedrock model -> BEDROCK_MODEL (CONFIG["bedrock"]["model"], via create_llm("")).
      - vllm model    -> VLLM_RETRIEVER_MODEL (provider default, via create_llm("")).
      - openai model  -> CHAT_SUMMARIES_MODEL.

    Returns (llm, provider_name).
    """
    # Lazy import to avoid a circular import with llm.agent at module load.
    # PROVIDER follows LLM_DEFAULT_PROVIDER, so the summary path uses whatever
    # provider the deployment's AI config points at (bedrock, vllm, openai, ...).
    from open_webui.llm.agent import get_llm_provider, CONFIG, PROVIDER

    provider_name = PROVIDER

    if provider_name == "openai":
        # Legacy OpenAI path (honours a per-request api_key + the OpenAI proxy).
        from langchain_openai import ChatOpenAI
        from open_webui.config import OPENAI_API_KEY, OPENAI_API_BASE_URLS
        from open_webui.env import CHAT_SUMMARIES_MODEL

        key = api_key or OPENAI_API_KEY
        base_url = (
            OPENAI_API_BASE_URLS.value[0] if OPENAI_API_BASE_URLS.value else None
        )
        if not key:
            raise ValueError("OpenAI API key is required")
        llm = ChatOpenAI(
            model=CHAT_SUMMARIES_MODEL,
            api_key=key,
            base_url=base_url,
            temperature=0.2,
        )
    else:
        # Bedrock / other providers: build via the provider factory.
        # Empty model -> the provider's own default (Bedrock: BEDROCK_MODEL env).
        provider = get_llm_provider(provider_name, CONFIG)
        llm = provider.create_llm("", {}, streaming=False, temperature=0.2)
    return llm, provider_name


async def summarize_single_conversation(
    api_key: str, messages: Dict[str, Any], organization_id: str
) -> ConversationSummaryResponse:
    settings = messages.get("settings", {})
    customer = messages.get("customer", "")

    # Remove settings and customer from messages before sending to LLM
    conversation_messages = {
        k: v for k, v in messages.items() if k not in ("settings", "customer")
    }

    if not conversation_messages:
        return ConversationSummaryResponse(
            summary="",
            conversation_id="",
            importance="",
            sentiment="",
            organization_id=organization_id,
        )

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        settings=json.dumps(settings),
        messages=json.dumps(conversation_messages),
        customer=customer,
        organization_id=organization_id,
    )

    llm, provider_name = _build_summary_llm(api_key)

    # invoke_with_structured_output enforces the schema natively where the
    # provider supports it (OpenAI, recent Bedrock) and falls back to the
    # prompt-based JSON instruction otherwise.
    from open_webui.utils.document_assistant import invoke_with_structured_output
    from open_webui.routers.suggestion import parse_llm_json

    response = await invoke_with_structured_output(
        llm,
        [{"role": "user", "content": prompt}],
        model_provider=provider_name,
        output_schema=SUMMARY_OUTPUT_SCHEMA,
        schema_name="conversation_summary",
    )

    # Bedrock Converse may return content as a list of blocks; normalise to text.
    raw_content = getattr(response, "content", response)
    if isinstance(raw_content, list):
        raw_content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in raw_content
        )

    data = parse_llm_json(raw_content or "")

    # Trust our own ids over whatever the model echoed back.
    data["conversation_id"] = conversation_messages.get(
        "conversation_id", data.get("conversation_id", "")
    )
    data["organization_id"] = organization_id
    data.setdefault("summary", "")
    data.setdefault("importance", "")
    data.setdefault("sentiment", "")
    return ConversationSummaryResponse(**data)


@router.post("/summary", response_model=Dict[str, Any])
async def summarize_conversations(
    body: SummarizeRequest,
    userContext=Depends(auth_imbrace),
):
    try:
        organization_id = userContext.get("organization_id", "")
        # Per-request key only used by the OpenAI branch; Bedrock (the default)
        # authenticates via AWS credentials in CONFIG and ignores it.
        api_key = userContext.get("api_key")

        import asyncio

        summaries = await asyncio.gather(
            *[
                summarize_single_conversation(api_key, conversation, organization_id)
                for conversation in body.data
            ]
        )

        return {
            "success": True,
            "data": [s.model_dump() for s in summaries],
        }
    except Exception as e:
        log.exception("Error summarizing conversations")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
