import aiohttp
import json
from typing import Dict, Any, Optional
import os

import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


def _message_text(response) -> str:
    """Normalise a LangChain chat response to plain text.

    Bedrock Converse may return content as a list of blocks; flatten to text.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


class WebSearch:
    def __init__(self):
        self.session = None

    async def __aenter__(self):
        """Initialize aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Close aiohttp session."""
        await self.session.close()

    async def search_company_details(
        self, context: Dict[str, Any], prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Trigger a web search for company details, then extract structured info.

        The LLM follows LLM_DEFAULT_PROVIDER (override with COMPANY_SEARCH_PROVIDER
        / COMPANY_SEARCH_MODEL), so this runs on bedrock / vLLM / openai alike.
        Tavily is used for the actual web search.
        """
        try:
            logger.info(f"Searching company details with context: {context}")

            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if not tavily_api_key:
                logger.error("Missing TAVILY_API_KEY for web search")
                return None

            # Get company name from context
            data = context.get("data", "")
            if not data:
                logger.error("No company name provided in context")
                return None

            # Build the LLM via the shared provider factory (echart-style routing).
            from open_webui.llm.utils.task_llm import build_task_llm

            llm = build_task_llm(
                "COMPANY_SEARCH", default_openai_model="gpt-4o", temperature=0
            )
            tavily_tool = TavilySearchResults(api_key=tavily_api_key, max_results=5)

            # Phase 1: derive a concise web search query from the context.
            # (Replaces the OpenAI-only forced tool call, which was provider-locked.)
            query_response = await llm.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "You generate concise web search queries. Reply with "
                            "ONLY the search query text — no quotes, no explanation."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Company context: {data}\n"
                            "Search query to find detailed information about this company:"
                        )
                    ),
                ]
            )
            query = _message_text(query_response).strip() or str(data)

            # Execute Tavily search
            search_results = await tavily_tool.ainvoke({"query": query})
            logger.info(f"Search results: {search_results}")

            # Phase 2: extract structured company details as JSON.
            default_prompt = (
                "Extract relevant company information from the search results. "
                "Return as JSON with appropriate keys based on the available "
                "information. Use empty strings for missing data."
            )
            system_content = (prompt or default_prompt) + (
                "\n\nRespond with ONLY a valid JSON object (no markdown, no prose)."
            )
            logger.info(f"Using prompt: {system_content}")

            extraction_response = await llm.ainvoke(
                [
                    SystemMessage(content=system_content),
                    HumanMessage(
                        content=(
                            f"Search results for {data}:\n"
                            f"{json.dumps(search_results, indent=2)}"
                        )
                    ),
                ]
            )
            extracted_text = _message_text(extraction_response)

            # Tolerant JSON parse (handles markdown fences / embedded JSON).
            from open_webui.routers.suggestion import parse_llm_json

            try:
                return parse_llm_json(extracted_text)
            except Exception:
                logger.error("Failed to parse extracted company details")
                return None

        except Exception as error:
            logger.error(f"Error performing web search: {error}")
            return None


# Factory function
def create_web_search_service() -> WebSearch:
    return WebSearch()


# Module-level async function for convenience
async def search_company_details(
    context: Dict[str, Any], prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with WebSearch() as service:
        return await service.search_company_details(context, prompt)
