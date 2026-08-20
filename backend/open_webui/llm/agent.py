import json
import uuid
import re
import ast
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, AsyncGenerator, Optional, cast
import logging
import asyncio

from open_webui.utils import provider as provider_service
from open_webui.llm.utils.models import detect_provider
from langgraph_supervisor import create_supervisor, create_handoff_tool
from pymongo.collection import Collection
from langchain_mongodb import MongoDBChatMessageHistory
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers.hybrid_search import MongoDBAtlasHybridSearchRetriever
from open_webui.internal.vector_store import get_vector_store as _get_vector_store, DB_TYPE as _DB_TYPE
from open_webui.llm.utils.pg_adapters import PgVectorStore as _PgVectorStore
from langgraph.types import Command, Send
from strands.tools.executors import SequentialToolExecutor


from open_webui.llm.utils.chat_history import custom_trim_messages
from open_webui.llm.utils.multi_agent import (
    State,
    create_custom_handoff_tool,
    route_tools,
    create_task_description_handoff_tool,
    pretty_print_messages,
)
from open_webui.internal.mongo_db import mongodb_client
from open_webui.repository.file_content import file_content_repo
from open_webui.repository.echart import echart_repo
from pymongo.collection import Collection
from open_webui.config import (
    LLM_CONFIG,
    LLM_STUDIO_CONFIG,
    MONGODB_CONFIG,
    OLLAMA_CONFIG,
    OPENAI_API_KEY,
    RAG_EMBEDDING_CONFIG,
    KIMI_CONFIG,
    BEDROCK_CONFIG,
    VLLM_CONFIG,
)
from open_webui.models.ai_agents.board_message import (
    BoardMessageForm,
    board_message_model,
)
from open_webui.models.ai_agents.checkpoint import CheckpointWritesAio

# Import the interface from main.py
from .main import AgentChunk, AnswerQuestionInputs, AnswerChunk, ToolCallChunk
from .utils import vercel_ai_stream as vai

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.cache.memory import InMemoryCache


def safe_json_dumps(obj: Any) -> str:
    """
    Safely convert an object to JSON string, handling non-serializable objects.

    Args:
        obj: Object to serialize

    Returns:
        JSON string representation or string representation for non-serializable objects
    """
    try:
        return json.dumps(obj)
    except (TypeError, ValueError) as e:
        # Handle non-serializable objects
        if "not JSON serializable" in str(e):
            # Log the non-serializable object type for debugging
            obj_type = type(obj).__name__
            logger.warning(
                f"Object of type {obj_type} is not JSON serializable, converting to string representation"
            )

            # Convert to string representation for non-serializable objects
            if hasattr(obj, "__dict__"):
                # For objects with __dict__, try to serialize just the dict
                try:
                    return json.dumps(obj.__dict__)
                except (TypeError, ValueError):
                    return str(obj)
            else:
                return str(obj)
        else:
            # Re-raise other JSON errors
            raise


def extract_text_from_content(content: Any) -> str:
    """
    Extract text from content that can be either a string or a list of content blocks.

    Args:
        content: Content from LLM response - can be str or list of dicts

    Returns:
        Extracted text as string
    """
    if not content:
        return ""

    # Handle string content
    if isinstance(content, str):
        return content

    # Handle list of content blocks (e.g., Bedrock format)
    if isinstance(content, list):
        text_parts = []
        for content_block in content:
            if isinstance(content_block, dict):
                # Check for 'text' field in content block
                if content_block.get("type") == "text" and "text" in content_block:
                    text_parts.append(content_block["text"])
                # Skip tool_use blocks as they're not text content
            elif isinstance(content_block, str):
                text_parts.append(content_block)
        return "".join(text_parts)

    # Fallback to string representation
    return str(content)


from .utils.databoard import create_databoard_service
from .utils.workflow import (
    get_workflow_settings,
    get_workflow_settings_v2,
    trigger,
    trigger_v2,
)
from .utils.validator import validate_tool_calls
from .utils.tools import (
    convert_openai_tool_to_bedrock_tools,
    convert_openai_tools_to_bedrock_tools,
    get_reset_memory_tool,
    get_retriever_tool,
    get_aggregation_tool,
    get_parquet_query_tool,
    get_folder_info_tool,
)
from .utils.assistant import (
    _get_default_model,
    enrich_assistant_inputs,
    build_filters,
    build_assistant_prompt,
    RAG_SYSTEM_PROMPT,
    get_assistants,
    validate_inputs,
    merge_results,
)
from open_webui.utils.tools import execute_tool_server
from .utils.safety import check_safety

logger = logging.getLogger(__name__)


def clean_thinking_tags(content: str) -> str:
    """
    Remove thinking tags from content.

    Thinking tags can be:
    - <think> and </think> (HTML/XML style tags)
    - ◁think▷ or \u25c1think\u25b7 (Unicode characters)

    This function removes:
    1. All tags
    2. All content between matching open/close tags
    3. Single open or close tags
    """
    if not content:
        return content

    import re

    # Remove HTML/XML style thinking tags and their content
    # Pattern matches <think>content</think> and removes everything including the tags
    content = re.sub(
        r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove any remaining <think> or </think> tags (unmatched)
    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)

    # Remove Unicode thinking tags and their content
    # Pattern matches ◁think▷content◁think▷ and removes everything including the tags
    content = re.sub(r"◁think▷.*?◁think▷", "", content, flags=re.DOTALL)

    # Remove any remaining ◁think▷ tags (unmatched)
    content = re.sub(r"◁think▷", "", content)

    return content.strip()


PROVIDER = LLM_CONFIG.get("default_provider", "openai").lower()
# Embedding/retriever inherit the default provider when their own env
# (LLM_EMBEDDING_PROVIDER / LLM_RETRIEVER_PROVIDER) is left blank, so a
# single-provider deployment only needs to set LLM_DEFAULT_PROVIDER. Set the
# specific var to override (e.g. chat on bedrock, embedding on vllm).
EMBEDDING_PROVIDER = (LLM_CONFIG.get("embedding_provider") or PROVIDER).strip().lower()
RETRIEVER_PROVIDER = (LLM_CONFIG.get("retriever_provider") or PROVIDER).strip().lower()
SUPPORTED_PROVIDERS = LLM_CONFIG.get("provider", "openai").lower()
DB_URI = MONGODB_CONFIG.get("openai_host", "mongodb://localhost:27017")
DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")
MAX_RESPONSE_TOKENS = LLM_CONFIG.get("maxResponseTokens", 4000)
SAFETY_MAX_ITERATIONS = LLM_CONFIG.get("safetyMaxIterations", 50)
MAX_TOOL_CALLS = LLM_CONFIG.get("maxToolCalls", 5)
FULL_TEXT_SEARCH_INDEX_NAME = MONGODB_CONFIG.get(
    "full_text_search_index_name", "fulltext_search"
)
CONFIG = {
    "database": MONGODB_CONFIG.get("openai_db_name", "openai_db"),
    "search_k_value": RAG_EMBEDDING_CONFIG.get("searchKValue", 10),
    "chunkSize": RAG_EMBEDDING_CONFIG.get("chunkSize", 512),
    "chunkOverlap": RAG_EMBEDDING_CONFIG.get("chunkOverlap", 150),
    "openApi": {
        "api_key": OPENAI_API_KEY,
        "api_type": "gpt-4o",
    },
    "ollama": {
        "host": OLLAMA_CONFIG.get("host", "http://localhost:11434"),
        "retriever_model": OLLAMA_CONFIG.get("retrieverModel", "llama3.2"),
        "embedding_model": OLLAMA_CONFIG.get("embeddingModel", "llama3.2"),
        "supported_tool_call_models": OLLAMA_CONFIG.get("supported_tool_call_models", ["llama3.1", "mistral"]),
    },
    "lmstudio": {
        "host": LLM_STUDIO_CONFIG.get("host", "http://localhost:8000"),
        "api_type": LLM_STUDIO_CONFIG.get("retrieverModel", "llama3.2"),
        "embedding_model": LLM_STUDIO_CONFIG.get("embeddingModel", "llama3.2"),
    },
    "kimi": {
        "host": KIMI_CONFIG.get("host", "http://localhost:8000"),
        "model": KIMI_CONFIG.get("model", "lama3.2"),
    },
    "bedrock": {
        "region": BEDROCK_CONFIG.get("region", "us-west-2"),
        "access_key": BEDROCK_CONFIG.get("access_key", ""),
        "secret_key": BEDROCK_CONFIG.get("secret_key", ""),
        "model": BEDROCK_CONFIG.get(
            "model", "qwen.qwen3-32b-v1:0"
        ),
        # Embedding model id (env RAG_BEDROCK_EMBEDDING_MODEL). Default Titan v1
        # is 1536-dim, matching the rag.embedding/embedding_v2 columns and the
        # backfill, so the query path stays consistent with the ingest path.
        "embedding_model": RAG_EMBEDDING_CONFIG.get(
            "bedrockEmbeddingModel", "amazon.titan-embed-text-v1"
        ),
    },
    "vllm": {
        "host": VLLM_CONFIG.get("host", "http://localhost:8000"),
        "model": VLLM_CONFIG.get("model", "lama3.2"),
        "api_key": VLLM_CONFIG.get("api_key", ""),
        "embedding_model": VLLM_CONFIG.get("embeddingModel", "llama3.2"),
        # Dedicated embedding host (falls back to `host` when empty); used when
        # chat and embedding live on different vLLM instances.
        "embedding_host": VLLM_CONFIG.get("embeddingHost", ""),
        "retriever_model": VLLM_CONFIG.get("retrieverModel", "llama3.2"),
    },
}


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_name = config.get("database", "openai_db")
        self.rag_collection_name = "rag"
        self.history_collection_name = "history"
        self.search_k_value = int(config.get("search_k_value", 10))
        self.chunk_size = int(config.get("chunkSize", 512))
        self.chunk_overlap = int(config.get("chunkOverlap", 150))

    @abstractmethod
    def create_embedding_model(self):
        """Create embedding model specific to provider"""
        pass

    @abstractmethod
    def create_retriever_model(self, model: str, temperature: float = 0.1):
        """Create retriever model specific to provider"""
        pass

    @abstractmethod
    def create_llm(
        self,
        model: str,
        available_models: Dict[str, List[Dict]],
        streaming: bool = True,
        temperature: float = 0.1,
    ):
        """Create main LLM specific to provider"""
        pass

    @abstractmethod
    async def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get available models for this provider"""
        pass

    async def create_vector_store(self, embedding_model, collection_name: str):
        """Create vector store (common across providers)"""
        if _DB_TYPE == "postgresql":
            return _get_vector_store(embedding_model)
        db_client = await mongodb_client.get_client()
        db = db_client[self.db_name]
        rag_collection = db[collection_name]
        return MongoDBAtlasVectorSearch(
            embedding=embedding_model,
            collection=rag_collection,
            index_name="vector_index",
        )

    async def create_hybrid_retriever(
        self, vector_store, top_k, pre_filter, post_filter
    ):
        """Create hybrid retriever (provider-agnostic)"""
        if _DB_TYPE == "postgresql" and isinstance(vector_store, _PgVectorStore):
            # Return the PgVectorStore itself; _execute_hybrid_retrieve_context
            # will call vector_store.hybrid_search() directly in PG mode.
            return vector_store
        return MongoDBAtlasHybridSearchRetriever(
            vectorstore=vector_store,
            k=top_k,
            search_index_name=FULL_TEXT_SEARCH_INDEX_NAME,
            post_filter=post_filter,
            pre_filter=pre_filter,
        )

    async def get_chat_history(
        self, thread_id: str
    ) -> tuple[List[Any], Any]:
        """Get chat history (common across providers)"""
        if _DB_TYPE == "postgresql":
            from open_webui.llm.utils.pg_adapters import PgChatMessageHistory
            chat_history = PgChatMessageHistory(session_id=thread_id)
        else:
            chat_history = MongoDBChatMessageHistory(
                connection_string=MONGODB_CONFIG.get("openai_host"),
                database_name=self.db_name,
                collection_name=self.history_collection_name,
                session_id=thread_id,
            )
        return await chat_history.aget_messages(), chat_history

    async def get_file_content_from_history(self, thread_id: str) -> Optional[str]:
        """Get file content from file_content table"""
        try:
            result = await file_content_repo.get_by_thread_id(thread_id)
            if result and "content" in result:
                logger.info(f"Retrieved file content from history for thread {thread_id}")
                return result["content"]
            logger.info(f"No file content found in history for thread {thread_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving file content from history: {e}")
            return None

    async def save_file_content_to_history(
        self, thread_id: str, content: str, organization_id: str = ""
    ) -> bool:
        """Save file content to file_content table"""
        try:
            await file_content_repo.upsert(thread_id, content, organization_id)
            logger.info(f"Saved file content to history for thread {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving file content to history: {e}")
            return False

    async def save_echart_to_history(
        self, thread_id: str, echart_data: Dict[str, Any], organization_id: str = ""
    ) -> Optional[str]:
        """Save echart data to echarts table and return the ID"""
        try:
            echart_id = await echart_repo.save(thread_id, echart_data, organization_id)
            logger.info(f"Saved echart with ID {echart_id} for thread {thread_id}")
            return echart_id
        except Exception as e:
            logger.error(f"Error saving echart to history: {e}")
            return None

    def bind_tools_to_model(
        self,
        llm,
        workflow_settings: List[Dict],
        tools: List[Dict],
        detected_provider: Optional[str] = None,
    ):
        """Bind tools to model (common logic)"""
        # Combine tools
        all_tools = []
        all_tools.extend(workflow_settings or [])
        all_tools.extend(tools or [])

        # Add reset memory tool
        all_tools.append(get_reset_memory_tool())

        # Keep only OpenAI-compatible keys: type and function
        sanitized = []
        for t in all_tools:
            if isinstance(t, dict) and "function" in t and "type" in t:
                sanitized.append({"type": t.get("type"), "function": t.get("function")})
            else:
                sanitized.append(t)

        try:
            tool_names = []
            for t in all_tools:
                if isinstance(t, dict) and t.get("function", {}).get("name"):
                    tool_names.append(t.get("function", {}).get("name"))
            logger.info(
                f"[DEBUG-MCP] [MCP] bind_tools_to_model total_tools={len(all_tools)} sanitized={len(sanitized)} names={tool_names[:10]}"
            )
        except Exception:
            pass

        logger.info(f"llm details {llm}")

        # Check if llm has bind method
        if hasattr(llm, "bind") and callable(getattr(llm, "bind")):
            logger.info(f"Using llm.bind() method to bind tools {detected_provider}")
            if detected_provider == "bedrock":
                return llm.bind(tools=convert_openai_tools_to_bedrock_tools(sanitized))
            return llm.bind(tools=sanitized)
        else:
            # Fallback: directly set tools attribute (for Strands Agent)
            logger.info("llm does not have bind() method, setting llm.tools directly")
            processed_tools = convert_openai_tools_to_bedrock_tools(sanitized)
            llm.tool_registry.process_tools(processed_tools)
            llm.tool_executor = SequentialToolExecutor()
            return llm

    def is_tool_call_available(self, llm) -> bool:
        """Check if model supports tool calling"""
        return True

    async def answer_question(
        self,
        organization_id: str = "",
        model: str = "",
        inputs: AnswerQuestionInputs = None,
    ) -> AsyncGenerator[AnswerChunk, None]:
        """
        Main answer question function - common logic across providers
        Provider-specific models are created via abstract methods
        """
        try:
            temperature = 0.1
            use_memory = True
            response_mode = getattr(inputs, "response_mode", "standard") if inputs else "standard"
            logger.info(f"Answer question called with response_mode={response_mode}")
            # Initialize variables matching JS structure
            tools = inputs.tools if inputs else []
            assistant_prompt = RAG_SYSTEM_PROMPT
            thread_id = (
                inputs.thread_id if inputs and inputs.thread_id else str(uuid.uuid4())
            )
            message_id = str(uuid.uuid4())

            # Validate inputs
            if not validate_inputs(inputs.dict() if inputs else {}):
                yield AnswerChunk(
                    thread_id=thread_id, message="", message_id=message_id
                )
                return

            # Enrich inputs with assistant details if assistantId is provided
            enriched_inputs_dict = inputs.dict() if inputs else {}
            assistant_name = "superagent"
            version = 1
            agent_type = ""
            if inputs and inputs.assistant_id:
                enriched_inputs_dict, final_model = await enrich_assistant_inputs(
                    organization_id, model, enriched_inputs_dict
                )
                # Convert back to AnswerQuestionInputs
                inputs = AnswerQuestionInputs(**enriched_inputs_dict)
                model = final_model
                temperature = enriched_inputs_dict.get("temperature", 0.1)
                use_memory = enriched_inputs_dict.get("use_memory", True)
                assistant_name = enriched_inputs_dict.get("name", "superagent")
                agent_type = enriched_inputs_dict.get("agent_type", "")
                version = enriched_inputs_dict.get("version", 1)

            # Resolve provider_config from enriched_inputs_dict.provider_id
            provider_config = CONFIG
            custom_provider_from_agent = None
            enriched_provider_id = enriched_inputs_dict.get("provider_id")
            if enriched_provider_id and enriched_provider_id != "system":
                custom_provider_from_agent = await provider_service.get_by_id(
                    organization_id=organization_id,
                    provider_id=enriched_provider_id,
                )
                if custom_provider_from_agent:
                    provider_config = custom_provider_from_agent.get("config") or {}

            # Inject MCP/OpenAPI tools from enriched_inputs_dict.meta_data.tool_server
            tool_server = (enriched_inputs_dict.get("meta_data") or {}).get(
                "tool_server"
            ) or {}
            if tool_server.get("enabled") and tool_server.get("url"):
                try:
                    from open_webui.utils.tools import get_tool_server_data

                    raw_url = str(tool_server.get("url", ""))
                    raw_path = str(tool_server.get("path", "openapi.json"))
                    auth_type = str(tool_server.get("auth_type", "bearer")).lower()
                    server_type = str(tool_server.get("type", "openapi")).lower()

                    base_url = raw_url if server_type == "mcp" else raw_url.rstrip("/")
                    normalized_path = raw_path.lstrip("/") if raw_path else ""
                    fetch_url = (
                        base_url
                        if server_type == "mcp"
                        else (
                            f"{base_url}/{normalized_path}"
                            if normalized_path
                            else base_url
                        )
                    )

                    token = ""
                    if auth_type == "bearer":
                        token = tool_server.get("key", "") or ""

                    server_data = await get_tool_server_data(
                        token, fetch_url, server_type
                    )
                    specs = server_data.get("specs", []) or []
                    logger.info(
                        "[DEBUG-LLM] Fetched %d MCP/OpenAPI specs from %s",
                        len(specs),
                        fetch_url,
                    )

                    server_data.setdefault(
                        "connection_url", server_data.get("url", fetch_url)
                    )
                    execute_url = (
                        base_url
                        if server_type == "openapi"
                        else server_data["connection_url"]
                    )

                    mcp_tools = []
                    for spec in specs:
                        name = spec.get("name")
                        if not name:
                            continue
                        mcp_tools.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "description": spec.get("description", ""),
                                    "parameters": spec.get("parameters", {}),
                                },
                                "server_context": {
                                    "url": execute_url,
                                    "token": token,
                                    "type": server_data.get("type", server_type),
                                    "headers": server_data.get("headers", {}),
                                    "server_data": server_data,
                                },
                            }
                        )

                    tools = (tools or []) + mcp_tools
                    logger.info(
                        "[DEBUG-LLM] Appended %d MCP tools: %s",
                        len(mcp_tools),
                        [t["function"]["name"] for t in mcp_tools],
                    )
                except Exception as e:
                    logger.error("[DEBUG-LLM] Failed to inject MCP tools: %s", e)

            # Guardrail check
            if inputs and inputs.guardrail_id:
                safety_status = await check_safety(
                    org_id=organization_id,
                    message=inputs.question,
                    guardrails_config_id=inputs.guardrail_id,
                )

                logger.info(f"guardrail check: {safety_status}")

                is_safe = safety_status.is_safe
                safety_categories = safety_status.safety_categories

                if not is_safe:
                    yield AnswerChunk(
                        thread_id=thread_id,
                        message=f"Your request was blocked due to safety violation: {safety_categories}",
                        message_id=message_id,
                    )
                    return  # Stop execution

            # Setup prompts if instructions are provided
            if inputs.instructions:
                assistant_prompt = build_assistant_prompt(
                    RAG_SYSTEM_PROMPT, inputs.instructions
                )

            # Set up the filters for MongoDB queries
            filters = build_filters(inputs.dict(), organization_id)

            # Set up the collections based on cutoff date
            collection_name = self._determine_collection("")

            # Set up chat history (common)
            chat_history, chat_history_instance = await self.get_chat_history(thread_id)
            chat_history_value = []
            if use_memory:
                chat_history_value = validate_tool_calls(chat_history)

            # Set up the tools and workflow settings
            workflow_settings = await self._get_workflow_settings(
                inputs.workflow_function_call, organization_id, version
            )

            # Initialize LLM (provider-specific)
            # If custom provider from agent, use its models; otherwise fetch from system CONFIG
            if custom_provider_from_agent:
                all_models = get_all_available_models_from_custom_provider(
                    custom_provider_from_agent
                )
            else:
                all_models = await get_all_available_models(config=CONFIG)
            detected_provider = detect_provider(
                model_id=model,
                models=all_models,
                default_provider=PROVIDER,
            )
            logger.info(f"Detected provider {detected_provider}")
            detected_llm = get_llm_provider(detected_provider, provider_config)
            llm = detected_llm.create_llm(
                model, all_models, inputs.streaming, temperature
            )
            model_with_tools = detected_llm.bind_tools_to_model(
                llm, workflow_settings, tools, detected_provider
            )

            # Initialize models - use separate embedding provider if configured, default to openai
            _emb_provider_name = EMBEDDING_PROVIDER if EMBEDDING_PROVIDER else "openai"
            _emb_provider = get_llm_provider(_emb_provider_name, CONFIG)
            embedding_model = _emb_provider.create_embedding_model()
            logger.debug(f"Created embedding model using {_emb_provider_name} provider: {embedding_model}")
            # Create retriever model - use separate retriever provider if configured, default to openai
            _ret_provider_name = RETRIEVER_PROVIDER if RETRIEVER_PROVIDER else "openai"
            _ret_provider = get_llm_provider(_ret_provider_name, CONFIG)
            # Pass None as model when retriever provider differs from detected provider,
            # so the provider uses its own default retriever model
            _ret_model = model if _ret_provider_name == detected_provider else None
            retriever_model = _ret_provider.create_retriever_model(_ret_model, temperature)
            logger.info(f"Created retriever model using {_ret_provider_name} provider {_ret_model}, type={type(retriever_model).__name__}, openai_api_base={getattr(retriever_model, 'openai_api_base', 'NOT_SET')}, model={getattr(retriever_model, 'model_name', 'UNKNOWN')}")
            vector_store = await self.create_vector_store(
                embedding_model, collection_name
            )

            # Prepare question with preload information and file content
            new_question = inputs.question
            if inputs.preload_information and inputs.preload_information.strip():
                new_question = f"{inputs.preload_information}\n\n{inputs.question}"
                logger.info(f"Question with preload information: {new_question}")

            # Handle file_content: save if provided, retrieve if not
            file_content = None
            if inputs.file_content and inputs.file_content.strip():
                # Save file_content to MongoDB
                await self.save_file_content_to_history(
                    thread_id, inputs.file_content, organization_id
                )
                file_content = inputs.file_content
                logger.info(f"Saved file content to history for thread {thread_id}")
            else:
                # Try to retrieve file_content from MongoDB
                file_content = await self.get_file_content_from_history(thread_id)
                if file_content:
                    logger.info(
                        f"Retrieved file content from history for thread {thread_id}"
                    )
                else:
                    logger.info(
                        f"No file content found in history for thread {thread_id}"
                    )

            # Combine file_content with question if file_content exists
            if file_content and file_content.strip():
                new_question = f"{new_question}\n\nHere is the specific text or sections from the documents:\n{file_content}"
                logger.info(f"Question with file content: {new_question}")

            # Generate echart if needed (after file_content is retrieved and combined)
            if hasattr(inputs, "visualization") and inputs.visualization:
                try:
                    from .utils.echart import generate_echart

                    echart_result = await generate_echart(inputs.question, file_content)
                    if echart_result and isinstance(echart_result, dict):
                        # Save echart to MongoDB and get ID
                        echart_id = await self.save_echart_to_history(
                            thread_id, echart_result, organization_id
                        )
                        if echart_id:
                            # Yield the echart data with ID
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message="",
                                is_partial=True,
                                message_id=message_id,
                                echart=echart_result,
                                echart_id=echart_id,
                            )
                            logger.info(
                                f"Saved echart with ID {echart_id} for thread {thread_id}"
                            )
                    elif echart_result == "no echart required":
                        logger.info("EChart generation determined not required")
                except Exception as e:
                    logger.error(f"Error generating echart: {e}")
                    # Continue without echart if generation fails

            # Retrieve context if needed
            context_result = ""
            sources = []
            # logger.info(f"Checking if RAG analysis is needed from inputs: {inputs}")
            if (
                inputs.board_ids
                or inputs.folder_ids
                or inputs.default_folder_id
                or inputs.file_ids
            ):
                logger.info("Starting RAG analysis")

                # Get MongoDB collection for full-text search (None in PostgreSQL mode)
                if _DB_TYPE == "postgresql":
                    rag_collection = None
                else:
                    db_client = await mongodb_client.get_client()
                    db = db_client[self.db_name]
                    rag_collection = db[collection_name]

                context_result, sources = await self._retrieve_context(
                    organization_id,
                    inputs,
                    filters,
                    vector_store,
                    retriever_model,
                    rag_collection,
                    chat_history_instance,
                )

                # Update prompt with context
                formatted_prompt = self._build_prompt_with_context(
                    assistant_prompt,
                    chat_history_value,
                    inputs.question,
                    context_result,
                    thread_id,
                )
            else:
                # Build initial prompt without context
                formatted_prompt = self._build_prompt(
                    assistant_prompt, chat_history_value, new_question, thread_id
                )

            # Convert prompt to format expected by model
            prompt_json = self._format_prompt_for_model(formatted_prompt)

            # Stream response
            if (
                getattr(inputs, "sub_agents", None)
                and len(inputs.sub_agents) > 0
                and agent_type == "team_lead"
            ):
                logger.info("Streaming response with sub-agents enabled")
                async for chunk in self._process_multi_agent_response(
                    assistant_name,
                    inputs.instructions,
                    prompt_json,
                    thread_id,
                    message_id,
                    llm,
                    model_with_tools,
                    workflow_settings,
                    tools,
                    sources,
                    chat_history_instance,
                    inputs.sub_agents,
                    use_memory,
                    all_models,
                    response_mode,
                ):
                    yield chunk
            elif inputs.streaming:
                logger.info("Streaming response enabled")
                if detected_provider == "bedrock":
                    logger.info("Using Bedrock streaming response v4")
                    async for chunk in self._process_streaming_response_v4(
                        inputs,
                        prompt_json,
                        thread_id,
                        message_id,
                        model_with_tools,
                        workflow_settings,
                        tools,
                        sources,
                        chat_history_instance,
                        use_memory,
                    ):
                        yield chunk
                else:
                    # Streaming response - yield chunks as they come
                    async for chunk in self._process_streaming_response_v2(
                        inputs,
                        prompt_json,
                        thread_id,
                        message_id,
                        model_with_tools,
                        workflow_settings,
                        tools,
                        sources,
                        chat_history_instance,
                        use_memory,
                    ):
                        yield chunk
            else:
                # Non-streaming response
                if detected_provider == "bedrock":
                    response = await self._process_non_streaming_response_v2(
                        inputs,
                        prompt_json,
                        thread_id,
                        message_id,
                        model_with_tools,
                        workflow_settings,
                        tools,
                        sources,
                        chat_history_instance,
                        use_memory,
                    )
                    yield response
                else:
                    response = await self._process_non_streaming_response(
                        inputs,
                        prompt_json,
                        thread_id,
                        message_id,
                        model_with_tools,
                        workflow_settings,
                        tools,
                        sources,
                        chat_history_instance,
                        use_memory,
                    )
                    yield response

        except Exception as error:
            logger.error(f"Error in answer_question: {error}")
            yield AnswerChunk(
                thread_id=thread_id if "thread_id" in locals() else str(uuid.uuid4()),
                message="An error occurred while processing your request",
                message_id=(
                    message_id if "message_id" in locals() else str(uuid.uuid4())
                ),
            )

    # Common helper methods (shared across providers)
    def _determine_collection(self, created_at: str) -> str:
        """Determine which collection to use based on created_at date"""
        return self.rag_collection_name

    async def _get_workflow_settings(
        self, workflow_calls: List, organization_id: str, version: int = 1
    ) -> List[Dict]:
        """Get workflow settings"""
        if version == 2:
            return get_workflow_settings_v2(workflow_calls, organization_id)
        return get_workflow_settings(workflow_calls, organization_id)

    def _build_prompt(
        self, assistant_prompt: str, history: List[Any], question: str, thread_id: str
    ) -> List[Dict]:
        """Build prompt messages"""
        try:
            messages = []

            # System message with context placeholder
            # Use safe string formatting to avoid issues with JSON examples in prompts
            try:
                system_content = assistant_prompt.format(
                    context="", datetime=datetime.now().isoformat(), thread_id=thread_id
                )
            except (KeyError, ValueError) as format_error:
                logger.warning(
                    f"String formatting error in _build_prompt: {format_error}"
                )
                # Fallback: use template-style replacement to avoid format() issues
                system_content = assistant_prompt.replace("{context}", "")
                system_content = system_content.replace(
                    "{datetime}", datetime.now().isoformat()
                )
                system_content = system_content.replace("{thread_id}", thread_id)

            messages.append({"role": "system", "content": system_content})

            # Add history
            if history:
                messages.extend(history)

            # User question
            messages.append({"role": "user", "content": question})

            logger.debug(f"Built prompt with {len(messages)} messages")
            return messages

        except Exception as e:
            logger.error(f"Error in _build_prompt: {e}")
            # Return minimal prompt as fallback
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ]

    def _build_prompt_with_context(
        self,
        assistant_prompt: str,
        history: List[Any],
        question: str,
        context: str,
        thread_id: str,
    ) -> List[Dict]:
        """Build prompt with RAG context"""
        try:
            messages = []

            # System message with context
            # Use safe string formatting to avoid issues with JSON examples in prompts
            try:
                system_content = assistant_prompt.format(
                    context=context,
                    datetime=datetime.now().isoformat(),
                    thread_id=thread_id,
                )
            except (KeyError, ValueError) as format_error:
                logger.warning(
                    f"String formatting error in _build_prompt_with_context: {format_error}"
                )
                # Fallback: use template-style replacement to avoid format() issues
                system_content = assistant_prompt.replace("{context}", context)
                system_content = system_content.replace(
                    "{datetime}", datetime.now().isoformat()
                )
                system_content = system_content.replace("{thread_id}", thread_id)

            messages.append({"role": "system", "content": system_content})

            # Add history
            if history:
                messages.extend(history)

            # User question
            messages.append({"role": "user", "content": question})

            logger.debug(f"Built prompt with context, {len(messages)} messages")
            return messages

        except Exception as e:
            logger.error(f"Error in _build_prompt_with_context: {e}")
            # Return minimal prompt as fallback
            return [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Context: {context}",
                },
                {"role": "user", "content": question},
            ]

    def _format_prompt_for_model(self, prompt_messages: List[Dict]) -> List[Dict]:
        """Format prompt messages for the model"""
        return prompt_messages

    def _convert_mongodb_to_atlas_filter(
        self, filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert MongoDB query filters to Atlas Search filter format

        Args:
            filters: MongoDB filter dictionary with $eq, $or, $in operators

        Returns:
            Atlas Search filter dictionary
        """
        if not filters:
            return None

        filter_clauses = []

        for key, value in filters.items():
            if key == "$or" and isinstance(value, list):
                # Convert $or to should clauses
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
                # Handle direct filters like org_id: { $eq: ... }
                if isinstance(value, dict) and "$eq" in value:
                    filter_clauses.append(
                        {"equals": {"path": key, "value": value["$eq"]}}
                    )
                elif isinstance(value, dict) and "$in" in value:
                    filter_clauses.append({"in": {"path": key, "value": value["$in"]}})

        # Combine multiple filter clauses with AND logic
        if len(filter_clauses) == 0:
            return None
        elif len(filter_clauses) == 1:
            return filter_clauses[0]
        else:
            return {"compound": {"must": filter_clauses}}

    async def _retrieve_context(
        self,
        orgId: str,
        inputs: AnswerQuestionInputs,
        filters: Dict[str, Any],
        vector_store: MongoDBAtlasVectorSearch,
        retriever_model,
        rag_collection: Collection,
        chat_history: Optional[Any] = None,
    ) -> tuple[str, List[Dict]]:
        """
        Retrieve context using tool-based approach similar to JavaScript version.
        Uses retrieverTool and parquet query tool to let the model decide how to search.
        """
        try:
            # Import MongoDB client for full-text search

            # Initialize databoard service for aggregation
            databoard_service = create_databoard_service()

            # Fetch parquet files based on assistant_id
            parquet_files = []
            assistant_folder_ids = []
            if inputs.assistant_id:
                try:
                    from open_webui.repository.assistant import AssistantRepository
                    from open_webui.routers.parquets import get_by_filters

                    assistant_repo = AssistantRepository()
                    assistant_details = (
                        await assistant_repo.get_by_assistant_id_and_organization_id(
                            orgId, inputs.assistant_id
                        )
                    )

                    if assistant_details:
                        # Set folder_ids and board_ids from assistant configuration
                        folder_ids = assistant_details.get("folder_ids", [])
                        board_ids = assistant_details.get("board_ids", [])
                        assistant_folder_ids = folder_ids

                        # Fetch parquet files
                        parquet_result = await get_by_filters(
                            organization_id=orgId,
                            folder_ids=folder_ids,
                            board_ids=board_ids,
                        )

                        # Format parquet files for the tool
                        for parquet in parquet_result:
                            parquet_files.append(
                                {
                                    "s3_path": parquet.get("s3_path", ""),
                                    "alias": parquet.get("description", "")
                                    .replace(" ", "_")
                                    .lower()[:20]
                                    or f"table_{len(parquet_files)}",
                                }
                            )

                        logger.info(
                            f"Fetched {len(parquet_files)} parquet files for assistant {inputs.assistant_id}"
                        )

                except Exception as parquet_error:
                    logger.warning(f"Error fetching parquet files: {parquet_error}")

            # Expand assistant_folder_ids to include all nested subfolder IDs
            if assistant_folder_ids:
                try:
                    databoard_svc = create_databoard_service()
                    subfolder_response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        databoard_svc.get_folder_subfolders,
                        assistant_folder_ids,
                        True,   # recursive
                        True,   # ignore_assistant
                    )
                    if subfolder_response and subfolder_response.get("success"):
                        all_folder_ids = set(assistant_folder_ids)
                        for folder in subfolder_response.get("data", {}).get("folders", []):
                            all_folder_ids.add(folder.get("id", ""))
                            for subfolder in folder.get("subfolders", []):
                                all_folder_ids.add(subfolder.get("id", ""))
                        all_folder_ids.discard("")
                        expanded_folder_ids = list(all_folder_ids)
                        logger.info(
                            f"Expanded folder IDs from {len(assistant_folder_ids)} to "
                            f"{len(expanded_folder_ids)} (including subfolders)"
                        )
                        assistant_folder_ids = expanded_folder_ids

                        # Update filters to use expanded folder IDs
                        if filters.get("$or"):
                            for i, condition in enumerate(filters["$or"]):
                                if "folder_id" in condition:
                                    filters["$or"][i] = {
                                        "folder_id": {"$in": expanded_folder_ids}
                                    }
                                    break
                            else:
                                # No existing folder_id filter, add one
                                filters["$or"].append(
                                    {"folder_id": {"$in": expanded_folder_ids}}
                                )
                        elif expanded_folder_ids:
                            filters.setdefault("$or", []).append(
                                {"folder_id": {"$in": expanded_folder_ids}}
                            )

                        # Update inputs.folder_ids so full-text search also benefits
                        inputs.folder_ids = expanded_folder_ids
                except Exception as subfolder_error:
                    logger.warning(
                        f"Error expanding subfolder IDs, using original folder IDs: {subfolder_error}"
                    )

            # Check if retriever model is from Bedrock
            is_bedrock_model = (
                hasattr(retriever_model, "client")
                and hasattr(retriever_model.client, "_service_model")
                and "bedrock" in str(retriever_model.client._service_model).lower()
            ) or (
                hasattr(retriever_model, "model_id")
                and str(type(retriever_model).__module__).startswith("langchain_aws")
            )

            logger.info(
                f"Retriever model type: {type(retriever_model).__name__}, Is Bedrock: {is_bedrock_model}"
            )

            # Set up tools for retriever model only if not Bedrock
            if not is_bedrock_model:
                tools_list = [get_retriever_tool()]

                # Add parquet query tool if parquet files are available
                if parquet_files:
                    tools_list.append(get_parquet_query_tool())

                # Add folder info tool if assistant has folder_ids configured
                if assistant_folder_ids:
                    tools_list.append(get_folder_info_tool())

                logger.info(
                    f"Using tools: {[tool.get('function', {}).get('name', 'unknown') for tool in tools_list]}"
                )
                # Bind tools to retriever model
                retriever_model_with_tools = retriever_model.bind(tools=tools_list)
            else:
                logger.info("Skipping tool binding for Bedrock model")
                retriever_model_with_tools = retriever_model

            # Build retriever prompt with parquet schema information
            parquet_schema_info = ""
            if parquet_files:
                parquet_schema_info = "\n\nAvailable Parquet Data Sources:\n"
                for i, parquet in enumerate(parquet_files):
                    parquet_schema_info += f"{i+1}. Table alias: {parquet['alias']}\n"
                    parquet_schema_info += f"   S3 Path: {parquet['s3_path']}\n"
                    # Note: In a full implementation, you'd fetch actual column schema here
                    parquet_schema_info += f"   Description: Contains structured data that can be queried with SQL\n"

            # Build folder info hint for the retriever prompt
            folder_info_hint = ""
            if assistant_folder_ids:
                folder_info_hint = f"""
- Use query_folder_info to get folder metadata such as file counts, file names, subfolder hierarchy, and folder structure.
  This tool connects to {len(assistant_folder_ids)} configured folder(s) and returns their contents.
  You MUST use query_folder_info (instead of retrieve_context) when the user's question matches any of these patterns:
    * Asking how many files or documents are in a folder (e.g. "How many documents do you store?", "How many files are in the folders?")
    * Asking what files or documents exist (e.g. "What files do you have?", "List all documents", "Show me the files")
    * Asking about folder structure or subfolders (e.g. "What folders are there?", "Show folder hierarchy")
    * Asking about file types or file names (e.g. "Do you have any PDF files?", "What kind of files are stored?")
    * Any question about inventory, storage, or counting of documents/files
  Do NOT use retrieve_context for these questions — retrieve_context searches inside document content, not folder metadata."""

            retriever_prompt_content = f"""You are a retrieval assistant that helps extract relevant information from knowledge bases.
Your task is to analyze the user's question and determine the best search strategy.

IMPORTANT: Read the user's question carefully and pick the RIGHT tool:

Available tools:
- retrieve_context: Search inside document content using vector and full-text search. Use this when the user asks a question about the CONTENT of documents (e.g. "What does the report say about revenue?", "Find information about X").{parquet_schema_info}
- query_parquet: Execute SQL queries on structured parquet data. Use this when the question requires analytical queries, aggregations, or specific data filtering on tabular data.{folder_info_hint}

Choose the most appropriate tool based on the user's question. If unsure, prefer the tool whose description best matches the intent."""

            retriever_messages = [
                {"role": "system", "content": retriever_prompt_content},
                {"role": "user", "content": inputs.question},
            ]

            # Execute retriever model
            logger.info("Starting RAG analysis")
            start_time = datetime.now()

            # For Bedrock models, skip tool-based approach and use direct retrieval
            if is_bedrock_model:
                logger.info("Using direct retrieval for Bedrock model")
                result, sources = await self._execute_retrieve_context(
                    {
                        "vector_search_query": inputs.question,
                        "full_text_search_query": inputs.question,
                    },
                    inputs,
                    filters,
                    vector_store,
                    rag_collection,
                )
            else:
                # Use tool-based approach for non-Bedrock models
                # Debug: log the actual model client info before invocation
                _bound_model = getattr(retriever_model_with_tools, 'bound', retriever_model_with_tools)
                _base_url = (
                    getattr(_bound_model, 'base_url', None)  # ChatOllama
                    or getattr(_bound_model, 'openai_api_base', None)  # ChatOpenAI
                    or getattr(getattr(_bound_model, 'client', None), '_base_url', None)  # OpenAI client
                )
                logger.info(f"Using tool-based approach for retrieval, model type: {type(_bound_model).__name__}, base_url: {_base_url}")
                retriever_response = await retriever_model_with_tools.ainvoke(
                    retriever_messages
                )

                # Process tool calls
                result = ""
                sources = []

                if (
                    hasattr(retriever_response, "tool_calls")
                    and retriever_response.tool_calls
                ):
                    for tool_call in retriever_response.tool_calls:
                        function_name = tool_call.get("name", "")
                        function_args = tool_call.get("args", {})

                        logger.info(
                            f"Executing tool: {function_name} with args: {function_args}"
                        )

                        if function_name == "retrieve_context":
                            result, sources = await self._execute_retrieve_context(
                                function_args,
                                inputs,
                                filters,
                                vector_store,
                                rag_collection,
                            )
                        elif function_name == "query_parquet":
                            result = await self._execute_parquet_query(
                                function_args, parquet_files
                            )

                        elif function_name == "query_folder_info":
                            result = await self._execute_folder_info(
                                function_args, assistant_folder_ids
                            )

                        elif function_name == "reset_memory" and chat_history:
                            result = await self._execute_reset_memory(
                                function_args, chat_history
                            )

                        else:
                            result = "No valid tool was called"

                        logger.info(f"Tool call result: {result}")
                else:
                    # Fallback: direct retrieve_context call if no tools were used
                    logger.info("No tool calls detected, using fallback retrieval")
                    result, sources = await self._execute_retrieve_context(
                        {
                            "vector_search_query": inputs.question,
                            "full_text_search_query": inputs.question,
                        },
                        inputs,
                        filters,
                        vector_store,
                        rag_collection,
                    )

            end_time = datetime.now()
            logger.info(
                f"RAG analysis completed in {(end_time - start_time).total_seconds():.2f} seconds"
            )

            # Parse result and extract sources
            if isinstance(result, str) and result.strip():
                # For now, return the result as context
                # In a full implementation, you'd parse the result to extract sources
                logger.info(f"RAG analysis result: {len(sources)}")
                return result, sources
            else:
                return "No relevant documents found.", []

        except Exception as e:
            logger.error(f"Error in tool-based retrieval: {e}")
            return "", []

    async def _execute_retrieve_context(
        self,
        function_args: Dict[str, Any],
        inputs: AnswerQuestionInputs,
        filters: Dict[str, Any],
        vector_store: MongoDBAtlasVectorSearch,
        rag_collection: Collection,
        reranking_function: Any = None,
        top_k_reranker: int = 3,
        relevance_threshold: float = 0.0,
    ) -> tuple[str, List[Dict]]:
        """Execute the retrieve_context tool function"""
        try:
            vector_search_query = function_args.get("vector_search_query", "")
            full_text_search_query = function_args.get("full_text_search_query", "")
            sources = []

            if not vector_search_query.strip() and not full_text_search_query.strip():
                return "There is no valid query to retrieve", sources

            # Build full-text search pipeline (similar to JavaScript version)
            full_text_pipeline = [
                {
                    "$search": {
                        "index": FULL_TEXT_SEARCH_INDEX_NAME,
                        "compound": {
                            "must": [
                                {
                                    "text": {
                                        "query": full_text_search_query,
                                        "path": "text",
                                    },
                                },
                                {
                                    "equals": {
                                        "path": "org_id",
                                        "value": (
                                            inputs.organization_id
                                            if hasattr(inputs, "organization_id")
                                            else ""
                                        ),
                                    },
                                },
                            ],
                            "should": [],
                            "minimumShouldMatch": 0,
                        },
                    },
                },
                {"$limit": self.search_k_value},
                {
                    "$project": {
                        "text": 1,
                        "score": {"$meta": "searchScore"},
                        "_id": 1,
                        "original_name": 1,
                        "url": 1,
                    },
                },
            ]

            # Add dynamic OR conditions based on inputs
            if inputs.assistant_id:
                full_text_pipeline[0]["$search"]["compound"]["should"].append(
                    {
                        "equals": {
                            "path": "assistant_id",
                            "value": inputs.assistant_id,
                        },
                    }
                )

            if inputs.knowledge_hubs:
                full_text_pipeline[0]["$search"]["compound"]["should"].append(
                    {
                        "in": {
                            "path": "board_id",
                            "value": inputs.knowledge_hubs,
                        },
                    }
                )
            elif hasattr(inputs, "board_id") and inputs.board_id:
                full_text_pipeline[0]["$search"]["compound"]["should"].append(
                    {
                        "equals": {
                            "path": "board_id",
                            "value": inputs.board_id,
                        },
                    }
                )

            if inputs.file_ids:
                full_text_pipeline[0]["$search"]["compound"]["should"].append(
                    {
                        "in": {
                            "path": "file_id",
                            "value": inputs.file_ids,
                        },
                    }
                )

            if inputs.folder_ids:
                full_text_pipeline[0]["$search"]["compound"]["should"].append(
                    {
                        "in": {
                            "path": "folder_id",
                            "value": inputs.folder_ids,
                        },
                    }
                )

            # Set minimumShouldMatch if there are should conditions
            if full_text_pipeline[0]["$search"]["compound"]["should"]:
                full_text_pipeline[0]["$search"]["compound"]["minimumShouldMatch"] = 1

            # Define async functions for concurrent execution
            async def vector_search():
                """Execute vector search"""
                if vector_store is None or not vector_search_query:
                    return []

                try:
                    logging.info(
                        f"Vector search query: {vector_search_query}, filters: {filters}"
                    )

                    # Convert MongoDB filters to Atlas Search filter format
                    atlas_filter = (
                        self._convert_mongodb_to_atlas_filter(filters)
                        if filters
                        else None
                    )
                    logging.info(f"Converted Atlas filter: {atlas_filter}")

                    vector_results = await vector_store.asimilarity_search_with_score(
                        query=vector_search_query,
                        k=self.search_k_value,
                        pre_filter=filters,
                    )

                    return [
                        {
                            "pageContent": doc.page_content,
                            "metadata": doc.metadata,
                            "vectorScore": score,
                        }
                        for doc, score in vector_results
                    ]
                except Exception as e:
                    logger.error(f"Vector search error: {e}")
                    return []

            async def full_text_search():
                """Execute full-text search"""
                if not full_text_search_query:
                    return []

                # PostgreSQL mode: use PgVectorStore.full_text_search
                if _DB_TYPE == "postgresql" and isinstance(vector_store, _PgVectorStore):
                    try:
                        return await vector_store.full_text_search(
                            query=full_text_search_query,
                            k=self.search_k_value,
                            pre_filter=filters,
                        )
                    except Exception as e:
                        logger.error(f"PG full-text search error: {e}")
                        return []

                # MongoDB mode
                if rag_collection is None:
                    return []
                try:
                    loop = asyncio.get_event_loop()
                    full_text_results = await loop.run_in_executor(
                        None,
                        lambda: rag_collection.aggregate(full_text_pipeline).to_list(
                            None
                        ),
                    )
                    return [
                        {
                            "pageContent": doc["text"],
                            "metadata": {
                                "_id": doc["_id"],
                                "url": doc.get("url"),
                                "fileName": doc.get("original_name"),
                            },
                            "fullTextScore": min(doc["score"], 10) / 10,
                        }
                        for doc in full_text_results
                    ]
                except Exception as e:
                    logger.error(f"Full-text search error: {e}")
                    return []

            # Execute both searches concurrently
            vector_docs, full_text_docs = await asyncio.gather(
                vector_search(), full_text_search(), return_exceptions=True
            )

            # Handle exceptions from gather
            if isinstance(vector_docs, Exception):
                logger.error(f"Vector search failed: {vector_docs}")
                vector_docs = []

            if isinstance(full_text_docs, Exception):
                logger.error(f"Full-text search failed: {full_text_docs}")
                full_text_docs = []

            # Merge results using RRF (Reciprocal Rank Fusion)
            merged_results = merge_results(
                vector_docs, full_text_docs, self.search_k_value
            )

            # Rerank merged results if reranking function is available
            if reranking_function is not None and merged_results:
                from open_webui.llm.utils.rerank import rerank_documents

                merged_results = rerank_documents(
                    query=vector_search_query or full_text_search_query,
                    documents=merged_results,
                    reranking_function=reranking_function,
                    top_n=top_k_reranker,
                    relevance_threshold=relevance_threshold,
                )
                results = []
                for doc in merged_results:
                    page_content = doc.get("pageContent", "")
                    metadata = doc.get("metadata", {})
                    if page_content:
                        results.append(page_content)
                    if metadata and not any(
                        item.get("url") == metadata.get("url") for item in sources
                    ):
                        sources.append(metadata)
                logger.info(f"Reranked results: {len(results)}")
                return (
                    "\n\n".join(results)
                    if results
                    else "No relevant documents found."
                ), sources

            # Apply dynamic threshold (fallback when no reranker is configured)
            if merged_results:
                dynamic_threshold = merged_results[int(len(merged_results) * 0.75)].get(
                    "combined_score", 0.01
                )
                filtered_results = [
                    doc
                    for doc in merged_results
                    if doc.get("combined_score", 0) >= dynamic_threshold
                ]

                # Extract page content and build result
                results = []
                for doc in filtered_results:
                    page_content = doc.get("pageContent", "")
                    metadata = doc.get("metadata", {})
                    if page_content:
                        results.append(page_content)
                    if metadata:
                        # logger.info(f"Metadata: {metadata}")
                        if not any(
                            item.get("url") == metadata.get("url") for item in sources
                        ):
                            sources.append(metadata)
                logger.info(f"Filtered results: {len(results)}")
                return (
                    "\n\n".join(results) if results else "No relevant documents found."
                ), sources
            else:
                return "No relevant documents found.", sources

        except Exception as error:
            logger.error(f"Retrieve context error: {error}")
            return f"Error retrieving context: {str(error)}", sources

    async def _execute_hybrid_retrieve_context(
        self,
        function_args: Dict[str, Any],
        inputs: AnswerQuestionInputs,
        filters: Dict[str, Any],
        vector_store: MongoDBAtlasVectorSearch,
        rag_collection: Collection,
        reranking_function: Any = None,
        top_k_reranker: int = 3,
        relevance_threshold: float = 0.0,
    ):
        """Execute the hybrid_retrieve_context tool function"""
        try:
            vector_search_query = function_args.get("vector_search_query", "")
            full_text_search_query = function_args.get("full_text_search_query", "")
            sources = []

            if not vector_search_query.strip() and not full_text_search_query.strip():
                return "There is no valid query to retrieve", sources

            # post_filter for the hybrid retriever. It runs AFTER reciprocal-rank
            # fusion (final_hybrid_stage), where there is no $search/$vectorSearch
            # stage immediately preceding, so `{"$meta": "searchScore"}` is invalid
            # here (MongoDB error 40218). The fused score is already exposed as the
            # `score` field, so project it directly.
            full_text_pipeline = [
                {
                    "$project": {
                        "text": 1,
                        "score": 1,
                        "_id": 1,
                        "original_name": 1,
                        "url": 1,
                    },
                },
            ]

            logger.info(
                f"Hybrid search query: {vector_search_query}, full text query: {full_text_search_query}, pre_filters: {filters} post_filter: {full_text_pipeline}"
            )
            retriever = await self.create_hybrid_retriever(
                vector_store=vector_store,
                top_k=self.search_k_value,
                pre_filter=filters,
                post_filter=full_text_pipeline,
            )
            logger.info("Starting hybrid retrieval")

            # PostgreSQL: call hybrid_search directly (vector + FTS + RRF in one call)
            if _DB_TYPE == "postgresql" and isinstance(vector_store, _PgVectorStore):
                documents = await vector_store.hybrid_search(
                    query=vector_search_query or full_text_search_query,
                    k=self.search_k_value,
                    pre_filter=filters,
                )
            else:
                result = await retriever.ainvoke(vector_search_query)
                documents = []
                for doc in result:
                    if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                        documents.append(
                            {
                                "pageContent": doc.page_content,
                                "metadata": {
                                    "_id": doc.metadata.get("_id"),
                                    "url": doc.metadata.get("url"),
                                    "fileName": doc.metadata.get("original_name"),
                                },
                                "fullTextScore": 0.8,
                                "vectorScore": 0.8,
                            }
                        )
                    elif isinstance(doc, dict):
                        documents.append(
                            {
                                "pageContent": doc.get("text", ""),
                                "metadata": {
                                    "_id": doc.get("_id"),
                                    "url": doc.get("url"),
                                    "fileName": doc.get("original_name"),
                                },
                                "fullTextScore": min(doc.get("fulltext_score", 1), 10) / 10,
                                "vectorScore": doc.get("vector_score", 0),
                            }
                        )
                    else:
                        logger.warning(f"Unexpected document format: {type(doc)}")
                        continue

            logger.info(f"Hybrid retrieval results: {len(documents)}")

            # Rerank documents if reranking function is available
            if reranking_function is not None and documents:
                from open_webui.llm.utils.rerank import rerank_documents

                documents = rerank_documents(
                    query=function_args.get("vector_search_query", ""),
                    documents=documents,
                    reranking_function=reranking_function,
                    top_n=top_k_reranker,
                    relevance_threshold=relevance_threshold,
                )
                results = []
                for doc in documents:
                    page_content = doc.get("pageContent", "")
                    metadata = doc.get("metadata", {})
                    if page_content:
                        results.append(page_content)
                    if metadata and not any(
                        item.get("url") == metadata.get("url") for item in sources
                    ):
                        sources.append(metadata)
                logger.info(f"Reranked results: {len(results)}")
                return (
                    "\n\n".join(results)
                    if results
                    else "No relevant documents found."
                ), sources

            # No reranker configured: cap context deterministically by relevance.
            # `documents` is already sorted by combined_score (RRF) descending, so we
            # take the strongest chunks up to a hard count cap and a byte ceiling.
            # This replaces the old "75th-percentile vectorScore" heuristic, which
            # kept a relative fraction with no absolute cap and could overflow the
            # model context when many files matched.
            if documents:
                import os

                max_chunks = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "8"))
                max_bytes = int(os.getenv("RAG_MAX_CONTEXT_BYTES", "60000"))

                results = []
                total_bytes = 0
                for doc in documents[:max_chunks]:
                    page_content = doc.get("pageContent", "")
                    if not page_content:
                        continue
                    chunk_bytes = len(page_content.encode("utf-8"))
                    # Always allow the single top chunk; otherwise stop before the
                    # byte ceiling is exceeded.
                    if results and total_bytes + chunk_bytes > max_bytes:
                        break
                    results.append(page_content)
                    total_bytes += chunk_bytes
                    metadata = doc.get("metadata", {})
                    if metadata and not any(
                        item.get("url") == metadata.get("url") for item in sources
                    ):
                        sources.append(metadata)
                logger.info(
                    f"Context-capped results: {len(results)} chunk(s), {total_bytes} bytes "
                    f"(max_chunks={max_chunks}, max_bytes={max_bytes})"
                )
                return (
                    "\n\n".join(results) if results else "No relevant documents found."
                ), sources
            else:
                return "No relevant documents found.", sources
        except Exception as e:
            logger.error(f"Hybrid retrieval error: {e}")
            return f"Error in hybrid retrieval: {str(e)}", sources

    async def _execute_aggregate_context(
        self,
        orgId: str,
        function_args: Dict[str, Any],
        inputs: AnswerQuestionInputs,
        databoard_service,
        retriever_model,
    ) -> str:
        """Execute the aggregate_context tool function"""
        try:
            query = function_args.get("query", "")
            logger.info(f"Aggregate prompt: {query}")

            if not inputs.knowledge_hubs or len(inputs.knowledge_hubs) == 0:
                return "There is no knowledge hub to aggregate"

            board_id = inputs.knowledge_hubs[0]
            organization_id = orgId

            # Get board schema from databoard service
            board_schema_data = databoard_service.get_board_schema(
                organization_id, board_id
            )

            if not board_schema_data:
                return "There is no knowledge hub to aggregate"

            board_items = board_schema_data.get("boardData")
            board_schema = board_schema_data.get("boardSchema")

            if not board_schema or not board_items:
                return "There is no knowledge hub to aggregate"

            # Build prompt for generating aggregation pipeline
            pipeline_prompt = f"""You are a helpful assistant that can generate MongoDB aggregation pipelines based on user queries.

You have access to the following board schema: {json.dumps(board_schema)}

You have access to the following board items: {json.dumps(board_items)}

You have access to the following query context: "{query}"

Generate a MongoDB aggregation pipeline as a JSON array for board_id "{board_id}".
The pipeline should be a valid MongoDB aggregation pipeline that can be executed on the board_item collection.
Please don't include any explanation, just return the JSON array."""

            pipeline_messages = [{"role": "user", "content": pipeline_prompt}]

            # Generate pipeline using retriever model
            try:
                response = await retriever_model.ainvoke(pipeline_messages)
                pipeline_content = (
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                logger.error(f"Pipeline generation error: {e}")
                return "Error generating aggregation pipeline."

            # Parse the generated pipeline
            try:
                # Clean the response to extract JSON
                cleaned_json = re.sub(r"```json\s*", "", pipeline_content)
                cleaned_json = re.sub(r"```\s*", "", cleaned_json)
                cleaned_json = cleaned_json.strip()

                pipeline = json.loads(cleaned_json)
            except json.JSONDecodeError as error:
                logger.error(f"Pipeline parsing error: {error}")
                return "Error parsing the aggregation pipeline."

            logger.info(f"Aggregate pipeline: {pipeline}")

            # Execute pipeline using databoard service
            data = databoard_service.execute_pipeline(
                organization_id, board_id, {"query": pipeline}
            )

            if not data:
                return "Error executing the aggregation pipeline."

            logger.info(f"Aggregate data: {data}")

            return f"User query: {query}\nresponse from retriever: {json.dumps(data)}\n"

        except Exception as error:
            logger.error(f"Aggregate context error: {error}")
            return f"Error in aggregation: {str(error)}"

    async def _execute_parquet_query(
        self, function_args: Dict[str, Any], parquet_files: List[Dict[str, Any]]
    ) -> str:
        """Execute the query_parquet tool function using DuckDB"""
        try:
            query = function_args.get("query", "")
            query_parquet_files = function_args.get("parquet_files", [])

            if not query.strip():
                return "No query provided for parquet execution"

            if not query_parquet_files:
                return "No parquet files specified for query"

            logger.info(f"Executing parquet query: {query}")
            logger.info(f"Available parquet files: {parquet_files}")

            # Validate that requested files are available
            available_files = {pf["alias"]: pf["s3_path"] for pf in parquet_files}
            requested_aliases = {pf.get("alias", "") for pf in query_parquet_files}

            missing_files = requested_aliases - set(available_files.keys())
            if missing_files:
                return f"Requested parquet files not available: {missing_files}"

            # Execute query using DuckDB
            try:
                import duckdb

                from open_webui.config import (
                    S3_ACCESS_KEY_ID,
                    S3_SECRET_ACCESS_KEY,
                    S3_REGION_NAME,
                    S3_ENDPOINT_URL,
                    S3_ADDRESSING_STYLE,
                )

                # Create DuckDB connection
                con = duckdb.connect()

                # httpfs is required to read parquet over http(s)/s3 — e.g.
                # file-service download URLs when tabular storage is local.
                try:
                    con.execute("INSTALL httpfs; LOAD httpfs;")
                except Exception as _httpfs_err:
                    logger.warning(f"Could not load httpfs extension: {_httpfs_err}")

                # Set up S3 credentials/settings from app config
                s3_access_key = (S3_ACCESS_KEY_ID or "").strip()
                s3_secret_key = (S3_SECRET_ACCESS_KEY or "").strip()
                s3_region = ((S3_REGION_NAME or "").strip() or "us-east-1")
                s3_endpoint = (S3_ENDPOINT_URL or "").strip()
                s3_url_style = (S3_ADDRESSING_STYLE or "").strip()

                if s3_access_key and s3_secret_key:
                    settings = [
                        f"SET s3_access_key_id='{s3_access_key}';",
                        f"SET s3_secret_access_key='{s3_secret_key}';",
                        f"SET s3_region='{s3_region}';",
                    ]
                    if s3_endpoint:
                        settings.append(f"SET s3_endpoint='{s3_endpoint}';")
                    if s3_url_style:
                        settings.append(f"SET s3_url_style='{s3_url_style}';")

                    con.execute("\n".join(settings))
                    logger.info("Configured DuckDB with S3 credentials from open_webui.config")
                else:
                    logger.warning(
                        "No S3 credentials configured (S3_ACCESS_KEY_ID/S3_SECRET_ACCESS_KEY); DuckDB may not be able to access S3 files"
                    )

                # Create table references for each requested parquet file
                # Resolve stored paths to local disk when tabular storage is local.
                from open_webui.retrieval.tabular.duckdb import resolve_tabular_path

                table_creations = []
                for pf in query_parquet_files:
                    alias = pf.get("alias", "")
                    s3_path = resolve_tabular_path(available_files.get(alias, ""))

                    if s3_path and alias:
                        # Create a table reference to the parquet file
                        table_creations.append(
                            f"""
                            CREATE OR REPLACE TABLE {alias}
                            AS SELECT * FROM read_parquet('{s3_path}')
                        """
                        )

                # Execute table creations
                for table_sql in table_creations:
                    logger.info(f"Creating table: {table_sql.strip()}")
                    con.execute(table_sql)

                # Execute the user's query
                logger.info(f"Executing user query: {query}")
                result = con.execute(query)

                # Fetch results (limit to prevent huge responses)
                rows = result.fetchmany(1000)  # Limit to 1000 rows
                columns = [desc[0] for desc in result.description]

                # Format results as readable text
                if rows:
                    # Create a table-like output
                    output_lines = []

                    # Add column headers
                    output_lines.append(" | ".join(columns))
                    output_lines.append("-" * len(output_lines[0]))

                    # Add data rows (limit individual cell content)
                    for row in rows[:50]:  # Limit display to 50 rows
                        formatted_row = []
                        for cell in row:
                            cell_str = str(cell) if cell is not None else "NULL"
                            # Truncate long cells
                            if len(cell_str) > 50:
                                cell_str = cell_str[:47] + "..."
                            formatted_row.append(cell_str)
                        output_lines.append(" | ".join(formatted_row))

                    # Add summary if truncated
                    total_rows = len(rows)
                    if total_rows > 50:
                        output_lines.append(f"\n... and {total_rows - 50} more rows")
                    elif (
                        len(result.fetchall()) > 0
                    ):  # Check if there are more rows we didn't fetch
                        output_lines.append("\n... (results truncated)")

                    result_text = "\n".join(output_lines)
                else:
                    result_text = "Query executed successfully but returned no results."

                # Close connection
                con.close()

                logger.info(
                    f"Parquet query completed successfully, result length: {len(result_text)}"
                )
                return f"Query Results:\n{result_text}"

            except ImportError as e:
                logger.error(f"DuckDB not available: {e}")
                return "DuckDB is not available for parquet query execution"

            except Exception as query_error:
                logger.error(f"Parquet query execution error: {query_error}")
                return f"Error executing parquet query: {str(query_error)}"

        except Exception as error:
            logger.error(f"Parquet query error: {error}")
            return f"Error in parquet query execution: {str(error)}"

    async def _execute_folder_info(
        self,
        function_args: Dict[str, Any],
        folder_ids: List[str],
    ) -> str:
        """
        Execute the query_folder_info tool function.
        Calls the data-board API to retrieve folder structure, file counts and file names
        for the AI assistant's configured folder IDs.
        """
        try:
            query = function_args.get("query", "")
            recursive = function_args.get("recursive", True)

            if not folder_ids:
                return "No folder IDs available to query folder information."

            logger.info(
                f"Executing folder info query for {len(folder_ids)} folders, "
                f"recursive={recursive}, query='{query}'"
            )

            # Call the data-board API via DataboardService
            databoard_service = create_databoard_service()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                databoard_service.get_folder_subfolders,
                folder_ids,
                recursive,
                True,  # ignore_assistant
            )

            if not response or not response.get("success"):
                error_msg = (
                    response.get("message", "Unknown error")
                    if response
                    else "No response from folder API"
                )
                logger.error(f"Folder info API error: {error_msg}")
                return f"Error retrieving folder information: {error_msg}"

            data = response.get("data", {})
            folders = data.get("folders", [])
            total_folders = data.get("total_folders", 0)
            total_files = data.get("total_files", 0)

            # Build a human-readable summary for the LLM context
            summary_parts = [
                f"User query: {query}",
                f"Total folders: {total_folders}",
                f"Total files across all folders: {total_files}",
                f"Recursive search: {data.get('recursive', recursive)}",
                "",
                "Folder Details:",
            ]

            for folder in folders:
                folder_name = folder.get("name", "unknown")
                folder_path = folder.get("path", "")
                file_count = folder.get("file_count", 0)
                files = folder.get("files", [])
                subfolders = folder.get("subfolders", [])

                summary_parts.append(f"\n  Folder: {folder_name}")
                summary_parts.append(f"  Path: {folder_path}")
                summary_parts.append(f"  File count: {file_count}")

                if files:
                    # Deduplicate and list unique file names
                    unique_files = sorted(set(files))
                    summary_parts.append(
                        f"  Unique files ({len(unique_files)}): {', '.join(unique_files[:50])}"
                    )
                    if len(unique_files) > 50:
                        summary_parts.append(
                            f"  ... and {len(unique_files) - 50} more unique files"
                        )

                if subfolders:
                    subfolder_names = [
                        sf.get("name", "unknown") for sf in subfolders
                    ]
                    summary_parts.append(
                        f"  Subfolders ({len(subfolders)}): {', '.join(subfolder_names)}"
                    )

            result_text = "\n".join(summary_parts)
            logger.info(
                f"Folder info query completed: {total_folders} folders, {total_files} files"
            )
            return result_text

        except Exception as error:
            logger.error(f"Folder info query error: {error}")
            return f"Error retrieving folder information: {str(error)}"

    async def _execute_reset_memory(
        self, function_args: Dict[str, Any], chat_history: MongoDBChatMessageHistory
    ) -> str:
        """Execute the reset_memory tool function"""
        try:
            # Use conversation_id from function_args, required parameter
            conversation_id = function_args.get("conversation_id", "")

            # Clear chat history
            if chat_history and hasattr(chat_history, "clear"):
                if hasattr(chat_history, "aclear"):
                    await chat_history.aclear()
                else:
                    chat_history.clear()

            # Delete checkpoints for the conversation
            if conversation_id:
                try:
                    deleted_count = await CheckpointWritesAio.delete_by_thread(
                        conversation_id
                    )
                    logger.info(
                        f"Deleted {deleted_count} checkpoint records for conversation_id: {conversation_id}"
                    )
                except Exception as checkpoint_error:
                    logger.error(
                        f"Error deleting checkpoints for conversation {conversation_id}: {checkpoint_error}"
                    )
                    # Don't fail the entire operation if checkpoint deletion fails

            logger.info(
                f"Memory and checkpoints reset for conversation: {conversation_id}"
            )
            return "Memory and checkpoints have been reset. Please end the conversation and start a new one."

        except Exception as error:
            logger.error(f"Reset memory error: {error}")
            return f"Error resetting memory: {str(error)}"

    async def _execute_workflow_trigger(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        workflow_settings: List[Dict],
        thread_id: str,
    ) -> str:
        """Execute workflow trigger based on tool call"""
        try:
            # Find the workflow configuration
            workflow_config = next(
                (
                    ws
                    for ws in workflow_settings
                    if ws.get("function", {}).get("name") == tool_name
                ),
                None,
            )

            if not workflow_config:
                return f"Workflow configuration not found for tool: {tool_name}"

            # Extract workflow ID from tool name (format: workflowId_workflowName)
            workflow_id = tool_name.split("_")[0] if "_" in tool_name else tool_name

            # Get organization ID from tool args or workflow config
            organization_id = tool_args.get("organization_id")
            if not organization_id:
                logger.info(f"workflow_settings {workflow_config}")
                # Try to get organization_id from workflow config function
                organization_id = workflow_config.get("function", {}).get(
                    "organization_id", ""
                )
                logger.info(
                    f"Extracted organization_id from workflow config: {organization_id}"
                )

                # If still empty, log a warning but continue
                if not organization_id:
                    logger.warning(
                        f"No organization_id found in tool_args or workflow_config for workflow: {tool_name}"
                    )

            # Extract HTTP method from workflow configuration
            method = workflow_config.get("function", {}).get("method", "POST").upper()

            # Validate HTTP method
            valid_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
            if method not in valid_methods:
                logger.warning(
                    f"Invalid HTTP method '{method}' for workflow '{tool_name}'. Defaulting to POST."
                )
                method = "POST"

            logger.info(f"Using HTTP method: {method} for workflow: {tool_name}")

            # Prepare workflow data
            workflow_data = {
                "thread_id": thread_id,
                "input_data": tool_args,
                "timestamp": datetime.now().isoformat(),
                **tool_args,  # Include all tool arguments as workflow data
            }

            # Execute the workflow trigger
            logger.info(
                f"Triggering workflow: {workflow_id} for organization: {organization_id} with method: {method}"
            )

            result = await self._async_trigger_workflow(
                organization_id=organization_id,
                workflow_id=workflow_id,
                method=method,
                data=workflow_data,
            )

            if isinstance(result, str) and result == "can not execute workflow":
                return f"Failed to execute tool {tool_name}"

            return f"Tool {tool_name} was executed successfully \n Here is the result {result}"

        except Exception as error:
            logger.error(f"Error executing workflow trigger: {error}")
            return f"Error executing workflow: {str(error)}"

    async def _async_trigger_workflow(
        self, organization_id: str, workflow_id: str, method: str, data: Dict[str, Any]
    ) -> Any:
        """Async wrapper for workflow trigger function"""

        # Run the synchronous trigger function in a thread pool
        loop = asyncio.get_event_loop()

        # Check if workflow_id is not a number, use trigger_v2
        try:
            # Try to convert workflow_id to a number
            int(workflow_id)
            is_number = True
        except ValueError:
            is_number = False

        if not is_number:
            logger.info(f"Using trigger_v2 for non-numeric workflow_id: {workflow_id}")
            return await loop.run_in_executor(
                None, trigger_v2, organization_id, workflow_id, method, data
            )
        else:
            logger.info(f"Using trigger for numeric workflow_id: {workflow_id}")
            return await loop.run_in_executor(
                None, trigger, organization_id, workflow_id, method, data
            )

    async def _process_tool_call(
        self,
        tool_call: Dict[str, Any],
        workflow_settings: List[Dict],
        tools: List[Dict],
        chat_history,
        thread_id: str,
    ) -> str:
        """Process individual tool call - shared logic between streaming and non-streaming"""
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})

        logger.info("finding tool: %s with args: %s", tool_name, tool_args)

        # Handle workflow triggers
        workflow_tool = next(
            (
                ws
                for ws in workflow_settings
                if ws.get("function", {}).get("name") == tool_name
            ),
            None,
        )

        logger.info(
            f"Processing workflow_tool call: {workflow_tool} with args: {tool_args}"
        )

        if workflow_tool:
            result = await self._execute_workflow_trigger(
                tool_name, tool_args, workflow_settings, thread_id
            )
            return f"{result}"

        # Handle reset memory tool
        elif tool_name == "reset_memory":
            await self._execute_reset_memory(tool_args, chat_history)
            return "Memory and checkpoints have been reset successfully. Please end the conversation and start a new one."

        # Handle other custom tools
        elif any(tool.get("function", {}).get("name") == tool_name for tool in tools):
            return f"Custom tool '{tool_name}' executed."

        return f"Unknown tool '{tool_name}' called."

    async def _process_multi_agent_response(
        self,
        assistant_name: str,
        instructions: str,
        prompt_json: List[Dict],
        thread_id: str,
        message_id: str,
        llm,
        model_with_tools,
        workflow_settings: List[Dict],
        tools: List[Dict],
        sources: List[Dict],
        chat_history,
        sub_agents: List[Dict] = [],
        use_memory: bool = True,
        all_models: Dict[str, List[Dict]] = None,
        response_mode: str = "standard"
    ) -> AsyncGenerator[AnswerChunk, None]:
        """Process streaming response from LLM with workflow trigger integration"""
        try:
            accumulated_content = ""
            assistant_name = self._normalize_agent_name(assistant_name)
            is_advanced = response_mode == "advanced"
            text_block_id = vai.generate_id() if is_advanced else None

            supervisor = model_with_tools
            if sub_agents and len(sub_agents) > 0:
                # Build sub-agents if provided
                agents = await self.build_sub_agents(sub_agents, all_models)
                if agents:
                    async with AsyncMongoDBSaver.from_conn_string(
                        DB_URI, DB_NAME
                    ) as checkpointer:
                        logger.info(
                            f"Built {len(agents)} sub-agents for streaming for supervisor {assistant_name}"
                        )
                        # Extract only the .agent field from each agent_config for supervisor
                        handoffs = [
                            agent_config.get("handoff")
                            for agent_config in agents
                            if agent_config.get("handoff")
                        ]
                        # agentList = [
                        #     agent_config.get("agent")
                        #     for agent_config in agents
                        #     if agent_config.get("agent")
                        # ]
                        sub_agent_workflow_settings = [
                            agent_config.get("workflow_settings")
                            for agent_config in agents
                            if agent_config.get("workflow_settings")
                        ]

                        logger.info(
                            f"Found {len(handoffs)} handoffs for supervisor {assistant_name}"
                        )
                        combined_workflow_settings = []
                        if sub_agent_workflow_settings:
                            # Flatten the list of lists - each sub-agent has a list of workflow settings
                            for agent_workflow_settings in sub_agent_workflow_settings:
                                if agent_workflow_settings:
                                    combined_workflow_settings.extend(
                                        agent_workflow_settings
                                    )
                        if workflow_settings:
                            combined_workflow_settings.extend(workflow_settings)

                        if agents and len(agents) > 0:
                            supervisor_tools = []
                            supervisor_tools.extend(workflow_settings or [])
                            supervisor_tools.extend(tools or [])
                            supervisor_tools.extend(handoffs or [])
                            if len(workflow_settings) > 0:
                                supervisor_tools.append(get_reset_memory_tool())
                            logger.info(
                                f"Creating supervisor with {len(supervisor_tools)} tools and {len(agents)} agents"
                            )

                            supervisor_with_tools = llm.bind_tools(
                                tools=supervisor_tools,
                                parallel_tool_calls=False,
                            )

                            # supervisor_agent = create_react_agent(
                            #     model=supervisor_with_tools,
                            #     tools=supervisor_tools,
                            #     name=assistant_name,
                            #     prompt=instructions,
                            # )
                            graph = StateGraph(State)

                            supervisor_agent = self.build_supervisor(
                                llm=supervisor_with_tools,
                                name=assistant_name,
                                combined_workflow_settings=combined_workflow_settings,
                                tools=supervisor_tools,
                                chat_history=chat_history,
                                thread_id=thread_id,
                            )
                            graph.add_node(assistant_name, supervisor_agent)
                            for agent in agents:
                                graph.add_node(
                                    node=agent.get("agent"),
                                    metadata=agent,
                                )

                            graph.add_edge(START, assistant_name)
                            graph.add_edge(assistant_name, END)
                            # graph.add_conditional_edges(assistant_name, route_tools)
                            for agent in agents:
                                graph.add_edge(agent.get("name"), assistant_name)

                            if use_memory:
                                supervisor = graph.compile(
                                    checkpointer=checkpointer,
                                    cache=InMemoryCache(),
                                )
                            else:
                                supervisor = graph.compile()

                            # supervisor = create_supervisor(
                            #     model=llm,
                            #     agents=agentList,
                            #     prompt=instructions,
                            #     supervisor_name=assistant_name,
                            #     output_mode="full_history",
                            #     add_handoff_messages=True,
                            #     add_handoff_back_messages=True,
                            #     parallel_tool_calls=False,
                            # ).compile(
                            #     checkpointer=checkpointer,
                            #     cache=InMemoryCache()
                            # )

                            logger.info(
                                f"Supervisor created with {len(agents)} sub-agents"
                            )

                            # Emit Vercel AI SDK preamble for advanced mode
                            if is_advanced:
                                yield vai.message_start(message_id)
                                yield vai.start_step()
                                yield vai.text_start(text_block_id)
                                # Emit sources as source-document parts
                                if sources:
                                    yield vai.emit_sources(sources)

                            langgraphConfig = RunnableConfig(thread_id=thread_id)
                            async for chunk in supervisor.astream(
                                {"messages": prompt_json, "value": "start"},
                                stream_mode="messages",
                                config=langgraphConfig,
                                subgraphs=True,
                            ):
                                # Handle LangGraph supervisor chunks
                                # logger.info(f"Received chunk: {chunk}")

                                # New format: (('agent_name:id',), (AIMessageChunk(...), metadata))
                                if isinstance(chunk, tuple) and len(chunk) >= 2:
                                    agent_info = chunk[0]  # ('agent_name:id',)
                                    message_data = chunk[
                                        1
                                    ]  # (AIMessageChunk, metadata)

                                    if (
                                        isinstance(message_data, tuple)
                                        and len(message_data) >= 1
                                    ):
                                        message = message_data[0]  # AIMessageChunk

                                        # Handle AIMessageChunk content
                                        if (
                                            hasattr(message, "content")
                                            and message.content
                                            and hasattr(message, "__class__")
                                            and "AIMessageChunk"
                                            in str(message.__class__)
                                        ):
                                            accumulated_content += message.content
                                            if is_advanced:
                                                yield vai.text_delta(
                                                    text_block_id,
                                                    message.content,
                                                )
                                            else:
                                                yield AnswerChunk(
                                                    thread_id=thread_id,
                                                    message=message.content,
                                                    is_partial=True,
                                                    message_id=message_id,
                                                    sources=(
                                                        sources
                                                        if not accumulated_content
                                                        else []
                                                    ),
                                                )

                                        # Handle tool calls in AIMessageChunk
                                        if (
                                            hasattr(message, "tool_calls")
                                            and message.tool_calls
                                        ):
                                            for tool_call in message.tool_calls:
                                                # Only process complete tool calls (with valid name and args)
                                                if (
                                                    tool_call.get("name")
                                                    and tool_call.get("id")
                                                    and isinstance(
                                                        tool_call.get("args"), dict
                                                    )
                                                ):

                                                    tool_name = tool_call.get(
                                                        "name", ""
                                                    )
                                                    tool_args = tool_call.get(
                                                        "args", {}
                                                    )
                                                    tool_id = tool_call.get(
                                                        "id", ""
                                                    )

                                                    if tool_name.startswith(
                                                        "transfer_to"
                                                    ):
                                                        if is_advanced:
                                                            yield vai.data_part(
                                                                "agent-handoff",
                                                                {
                                                                    "agent": tool_name.replace(
                                                                        "transfer_to_", ""
                                                                    ),
                                                                    "args": tool_args,
                                                                    "threadId": thread_id,
                                                                    "messageId": message_id,
                                                                },
                                                            )
                                                        else:
                                                            yield AgentChunk(
                                                                thread_id=thread_id,
                                                                agent=tool_name.replace(
                                                                    "transfer_to_", ""
                                                                ),
                                                                args=tool_args,
                                                                is_partial=True,
                                                                message_id=message_id,
                                                            )
                                                    elif not tool_name.startswith(
                                                        "retrieve_context"
                                                    ) and not tool_name.startswith(
                                                        "aggregate_context"
                                                    ):
                                                        logger.info(
                                                            f"Processing tool call: {tool_name} with args: {tool_args}"
                                                        )
                                                        # if is_advanced:
                                                        #     # Emit Vercel AI SDK tool call events
                                                        #     yield vai.tool_input_start(
                                                        #         tool_id, tool_name
                                                        #     )
                                                        #     yield vai.tool_input_available(
                                                        #         tool_id,
                                                        #         tool_name,
                                                        #         tool_args,
                                                        #     )

                        else:
                            raise ValueError(
                                "No valid agents found for multi-agent building"
                            )
                else:
                    raise ValueError("Multi agent building failed")

            # Final chunk with complete response
            if accumulated_content and accumulated_content.strip():
                if is_advanced:
                    # Close the text block and emit finish events
                    yield vai.text_end(text_block_id)
                    yield vai.finish_step()
                    yield vai.finish_message()
                    yield vai.stream_done()

                # Save to chat history
                if chat_history and prompt_json and use_memory:
                    chat_history.add_user_message(prompt_json[-1].get("content", ""))
                    chat_history.add_ai_message(accumulated_content.strip())
            elif is_advanced:
                # Even if no content, close the stream properly
                yield vai.text_end(text_block_id)
                yield vai.finish_step()
                yield vai.finish_message()
                yield vai.stream_done()

        except Exception as error:
            logger.error(f"Streaming error: {error}")
            if response_mode == "advanced":
                yield vai.error_part(
                    "Something went wrong. Please try again later."
                )
                yield vai.finish_message()
                yield vai.stream_done()
            else:
                yield AnswerChunk(
                    thread_id=thread_id,
                    message=f"Something went wrong. Please try again later.",
                    is_partial=True,
                    message_id=message_id,
                )
                yield AnswerChunk(
                    thread_id=thread_id,
                    message=f"Something went wrong. Please try again later.",
                    is_partial=False,
                    message_id=message_id,
                )

    async def _process_streaming_response(
        self,
        prompt_json: List[Dict],
        thread_id: str,
        message_id: str,
        model_with_tools,
        workflow_settings: List[Dict],
        tools: List[Dict],
        sources: List[Dict],
        chat_history: MongoDBChatMessageHistory,
        use_memory: bool = True,
    ) -> AsyncGenerator[AnswerChunk, None]:
        """Process streaming response from LLM with workflow trigger integration"""
        try:
            # Create working copy of prompt messages
            working_messages = prompt_json
            accumulated_content = ""

            # Safety counter to prevent infinite loops
            safety_counter = 0

            logger.info(
                f"Starting streaming response processing for thread {thread_id} with message ID {message_id}"
            )

            # Continue processing until the AI naturally completes or stops
            while safety_counter < SAFETY_MAX_ITERATIONS:
                safety_counter += 1

                # Time the model invocation
                start_time = time.time()

                # Stream the response from the model
                current_response = None
                current_content = ""

                async for chunk in model_with_tools.astream(working_messages):
                    # Handle content chunks
                    if hasattr(chunk, "content") and chunk.content:
                        current_content += chunk.content
                        accumulated_content += chunk.content

                        # Yield content chunks immediately
                        yield AnswerChunk(
                            thread_id=thread_id,
                            message=chunk.content,
                            is_partial=True,
                            message_id=message_id,
                            sources=sources if not accumulated_content else [],
                        )

                    # Store the complete response for tool call processing
                    if hasattr(chunk, "tool_calls") or hasattr(
                        chunk, "response_metadata"
                    ):
                        current_response = chunk

                end_time = time.time()
                logger.info(
                    f"Streaming eval duration: {(end_time - start_time) * 1000:.2f}ms"
                )

                if not current_response:
                    # If no response, break and return final chunk
                    break

                # Check if we should stop (no tool calls or stop reason)
                should_stop = False

                # First, check if there are tool calls to process
                has_tool_calls = (
                    hasattr(current_response, "tool_calls")
                    and current_response.tool_calls
                    and len(current_response.tool_calls) > 0
                )

                # If no tool calls, check for stop conditions
                if not has_tool_calls:
                    # Check various finish reasons
                    finish_reason = None
                    if (
                        hasattr(current_response, "response_metadata")
                        and current_response.response_metadata
                    ):
                        finish_reason = current_response.response_metadata.get(
                            "finish_reason"
                        ) or current_response.response_metadata.get("done_reason")

                    if (
                        hasattr(current_response, "additional_kwargs")
                        and current_response.additional_kwargs
                    ):
                        finish_reason = (
                            finish_reason
                            or current_response.additional_kwargs.get(
                                "finishReason", ""
                            ).lower()
                        )

                    # Stop if we have a stop reason, or if there's simply no more tool calls and we have content
                    if finish_reason == "stop" or (
                        current_content and current_content.strip()
                    ):
                        should_stop = True

                if should_stop:
                    # Add final AI message and save to history
                    if current_content:
                        working_messages.append(AIMessage(content=current_content))
                    final_prompt = custom_trim_messages(working_messages, max_tokens=50)
                    new_history = validate_tool_calls(final_prompt)
                    # Save to chat history
                    if chat_history and use_memory:
                        chat_history.clear()
                        for msg in new_history:
                            if isinstance(msg, dict):
                                if msg.get("role") == "user":
                                    chat_history.add_user_message(
                                        msg.get("content", "")
                                    )
                                elif msg.get("role") == "assistant":
                                    chat_history.add_ai_message(msg.get("content", ""))
                            elif isinstance(msg, HumanMessage):
                                chat_history.add_user_message(msg.content)
                            elif isinstance(msg, AIMessage):
                                chat_history.add_ai_message(msg.content)

                    # Final chunk with complete response
                    # yield AnswerChunk(
                    #     thread_id=thread_id,
                    #     message=accumulated_content,
                    #     is_partial=False,
                    #     message_id=message_id,
                    #     sources=sources,
                    # )
                    return

                # Process tool calls
                if has_tool_calls:
                    # Add the AI message with tool calls
                    working_messages.append(
                        AIMessage(
                            content=current_content or "",
                            tool_calls=current_response.tool_calls,
                        )
                    )

                    for tool_call in current_response.tool_calls:
                        is_error = False
                        # Handle both dict and object formats for tool_call
                        if hasattr(tool_call, "name"):
                            function_name = tool_call.name
                            function_args = (
                                tool_call.args if hasattr(tool_call, "args") else {}
                            )
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                        else:
                            function_name = tool_call.get("name", "")
                            function_args = tool_call.get("args", {})
                            tool_call_id = tool_call.get("id", "")

                        logger.info(f"Processing tool call: {tool_call}")

                        # Extract workflow ID from function name
                        workflow_id = None
                        if "_" in function_name:
                            try:
                                workflow_id = function_name.split("_")[0]
                            except ValueError:
                                pass

                        logger.info(
                            f"Extracted workflow_id: {workflow_id} from function_name: {function_name}"
                        )
                        if not workflow_id:
                            is_error = True

                        # Find workflow settings
                        settings = None
                        if workflow_settings:
                            settings = next(
                                (
                                    ws
                                    for ws in workflow_settings
                                    if ws.get("function", {}).get("name")
                                    == function_name
                                ),
                                None,
                            )

                        organization_id = None
                        if settings:
                            organization_id = settings.get("function", {}).get(
                                "organization_id"
                            )

                        if not organization_id:
                            is_error = True

                        method = "POST"
                        is_direct_response = False
                        if settings:
                            method = settings.get("function", {}).get("method", "POST")
                            is_direct_response = settings.get("function", {}).get(
                                "is_direct_response", False
                            )

                        # Handle specific tool calls
                        result = None

                        if function_name == "reset_memory":
                            result = await self._execute_reset_memory(
                                function_args, chat_history
                            )
                            # For reset memory, yield the result and return
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=result,
                                is_partial=True,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Handle custom tools (including MCP server tools)
                        tool_names = [
                            tool.get("function", {}).get("name", "") for tool in tools
                        ]
                        if function_name in tool_names:
                            # Try executing via MCP server if server_context is provided
                            try:
                                matched_tool = next(
                                    (
                                        t
                                        for t in tools
                                        if t.get("function", {}).get("name")
                                        == function_name
                                    ),
                                    None,
                                )
                                server_ctx = (
                                    (matched_tool or {}).get("server_context")
                                    if matched_tool
                                    else None
                                )
                                if server_ctx:
                                    logger.debug(
                                        f"[DEBUG-MCP] [MCP] streaming call name={function_name} url={server_ctx.get('url','')} token_present={bool(server_ctx.get('token'))}"
                                    )
                                    result = await execute_tool_server(
                                        token=server_ctx.get("token", ""),
                                        url=server_ctx.get("url", ""),
                                        name=function_name,
                                        params=function_args,
                                        server_data=server_ctx.get("server_data", {}),
                                    )

                                    tool_result_message = (
                                        safe_json_dumps(result)
                                        if not isinstance(result, str)
                                        else result
                                    )
                                    yield AnswerChunk(
                                        thread_id=thread_id,
                                        message=tool_result_message,
                                        is_partial=True,
                                        message_id=message_id,
                                    )

                                    tool_message = ToolMessage(
                                        content=tool_result_message,
                                        name=function_name,
                                        tool_call_id=tool_call_id,
                                    )
                                    working_messages.append(tool_message)
                                    # Continue to next tool call
                                    continue
                            except Exception as _:
                                pass

                            # Fallback: original behavior (UI-driven custom tools)
                            logger.debug(
                                f"[DEBUG-MCP] [MCP] streaming fallback (no server_context) for tool={function_name}"
                            )
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=current_content or "",
                                is_partial=True,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Handle workflow triggers
                        if is_error:
                            result = f"missing required data {function_args} to execute the function {function_name}"
                        else:
                            result = await self._async_trigger_workflow(
                                organization_id=organization_id,
                                workflow_id=str(workflow_id),
                                method=method,
                                data=function_args,
                            )

                        logger.info(f"Tool call result: {result}")

                        # Yield the tool result
                        tool_result_message = (
                            safe_json_dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                        yield AnswerChunk(
                            thread_id=thread_id,
                            message=tool_result_message,
                            is_partial=True,
                            message_id=message_id,
                        )

                        # Add tool message to conversation
                        tool_message = ToolMessage(
                            content=tool_result_message,
                            name=function_name,
                            status="error" if is_error else "success",
                            tool_call_id=tool_call_id,
                        )
                        working_messages.append(tool_message)

                        # Check if response is too long
                        if len(tool_result_message) > MAX_RESPONSE_TOKENS:
                            # In a real implementation, you'd upload to file storage here
                            # For now, we'll truncate and add a notice
                            truncated_response = tool_result_message[
                                :MAX_RESPONSE_TOKENS
                            ]
                            file_notice = f"\n\n[Note: Response truncated due to length. Full response was {len(tool_result_message)} characters.]"
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=truncated_response + file_notice,
                                is_partial=True,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Check if this is a direct response
                        if is_direct_response and len(current_response.tool_calls) == 1:
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=tool_result_message,
                                is_partial=True,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                    # After processing all tool calls, continue the loop to get the next response
                    continue

                # If we get here without tool calls and without stopping, break
                break

            # Safety net: if we somehow exit the loop without returning
            logger.warning(
                f"Safety limit reached ({SAFETY_MAX_ITERATIONS} iterations) - returning final streaming response"
            )

            # Final chunk with complete response
            # yield AnswerChunk(
            #     thread_id=thread_id,
            #     message=(
            #         accumulated_content
            #         if accumulated_content
            #         else "Processing completed"
            #     ),
            #     is_partial=False,
            #     message_id=message_id,
            #     sources=sources,
            # )

        except Exception as error:
            logger.error(f"Streaming error: {error}")
            yield AnswerChunk(
                thread_id=thread_id,
                message=f"Something went wrong. Please try again later.",
                is_partial=False,
                message_id=message_id,
            )
            yield AnswerChunk(
                thread_id=thread_id,
                message=f"Something went wrong. Please try again later.",
                is_partial=True,
                message_id=message_id,
            )

    async def _process_non_streaming_response(
        self,
        inputs: AnswerQuestionInputs,
        prompt_json: List[Dict],
        thread_id: str,
        message_id: str,
        model_with_tools,
        workflow_settings: List[Dict],
        tools: List[Dict],
        sources: List[Dict],
        chat_history: MongoDBChatMessageHistory,
        use_memory: bool = True,
    ) -> AnswerChunk:
        """Process non-streaming response from LLM with workflow trigger integration"""
        try:

            # Create working copy of prompt messages
            working_messages = prompt_json
            tool_call_success_counts = {}

            # Safety counter to prevent infinite loops
            safety_counter = 0
            logger.info(
                f"Starting non-streaming response processing for thread {thread_id} with message ID {message_id}"
            )
            # Continue processing until the AI naturally completes or stops
            while safety_counter < SAFETY_MAX_ITERATIONS:
                safety_counter += 1

                # Time the model invocation
                start_time = time.time()
                response = await model_with_tools.ainvoke(working_messages)
                end_time = time.time()

                logger.info(f"Assistant response: {response}")
                logger.info(f"Eval duration: {(end_time - start_time) * 1000:.2f}ms")

                if not response:
                    return AnswerChunk(
                        thread_id=thread_id,
                        message="",
                        is_partial=True,
                        message_id=message_id,
                        sources=sources,
                    )

                # Check if we should stop (no tool calls or stop reason)
                should_stop = False

                # First, check if there are tool calls to process
                has_tool_calls = (
                    hasattr(response, "tool_calls")
                    and response.tool_calls
                    and len(response.tool_calls) > 0
                )

                # If no tool calls, check for stop conditions
                if not has_tool_calls:
                    # Check various finish reasons
                    finish_reason = None
                    if (
                        hasattr(response, "response_metadata")
                        and response.response_metadata
                    ):
                        finish_reason = response.response_metadata.get(
                            "finish_reason"
                        ) or response.response_metadata.get("done_reason")

                    if (
                        hasattr(response, "additional_kwargs")
                        and response.additional_kwargs
                    ):
                        finish_reason = (
                            finish_reason
                            or response.additional_kwargs.get(
                                "finishReason", ""
                            ).lower()
                        )

                    # Stop if we have a stop reason, or if there's simply no more tool calls and we have content
                    if finish_reason == "stop" or (
                        hasattr(response, "content")
                        and response.content
                        and response.content.strip()
                    ):
                        should_stop = True

                if should_stop:
                    # Add final AI message and save to history
                    if hasattr(response, "content") and response.content:
                        working_messages.append(AIMessage(content=response.content))
                    final_prompt = custom_trim_messages(working_messages, max_tokens=50)
                    new_history = validate_tool_calls(final_prompt)
                    # Save to chat history
                    if chat_history and use_memory:
                        chat_history.clear()
                        for msg in new_history:
                            if isinstance(msg, dict):
                                if msg.get("role") == "user":
                                    chat_history.add_user_message(
                                        msg.get("content", "")
                                    )
                                elif msg.get("role") == "assistant":
                                    chat_history.add_ai_message(msg.get("content", ""))
                            elif isinstance(msg, HumanMessage):
                                chat_history.add_user_message(msg.content)
                            elif isinstance(msg, AIMessage):
                                chat_history.add_ai_message(msg.content)
                    if (
                        inputs.message_id
                        and inputs.conversation_id
                        and inputs.message_created_at
                        and response.content.startswith("bi_")
                    ):
                        board_message_model.create(
                            organization_id=inputs.organization_id,
                            details=BoardMessageForm(
                                conversation_id=inputs.conversation_id,
                                message_id=inputs.message_id,
                                board_item_id=inputs.board_item_id,
                                created_at=inputs.message_created_at,
                            ),
                        )

                    return AnswerChunk(
                        thread_id=thread_id,
                        message=clean_thinking_tags(response.content or ""),
                        is_partial=True,
                        message_id=message_id,
                        sources=sources,
                    )

                # Process tool calls
                if has_tool_calls:
                    # Add the AI message with tool calls
                    working_messages.append(
                        AIMessage(
                            content=response.content or "",
                            tool_calls=response.tool_calls,
                        )
                    )

                    for tool_call in response.tool_calls:
                        is_error = False
                        # Handle both dict and object formats for tool_call
                        if hasattr(tool_call, "name"):
                            function_name = tool_call.name
                            function_args = (
                                tool_call.args if hasattr(tool_call, "args") else {}
                            )
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                        else:
                            function_name = tool_call.get("name", "")
                            function_args = tool_call.get("args", {})
                            tool_call_id = tool_call.get("id", "")

                        logger.info(f"Processing tool call: {tool_call}")

                        # Extract workflow ID from function name
                        workflow_id = None
                        if "_" in function_name:
                            try:
                                workflow_id = function_name.split("_")[0]
                            except ValueError:
                                pass

                        logger.info(
                            f"Extracted workflow_id: {workflow_id} from function_name: {function_name}"
                        )
                        if not workflow_id:
                            is_error = True

                        # Find workflow settings
                        settings = None
                        if workflow_settings:
                            settings = next(
                                (
                                    ws
                                    for ws in workflow_settings
                                    if ws.get("function", {}).get("name")
                                    == function_name
                                ),
                                None,
                            )

                        organization_id = None
                        if settings:
                            organization_id = settings.get("function", {}).get(
                                "organization_id"
                            )

                        if not organization_id:
                            is_error = True

                        method = "POST"
                        is_direct_response = False
                        if settings:
                            method = settings.get("function", {}).get("method", "POST")
                            is_direct_response = settings.get("function", {}).get(
                                "is_direct_response", False
                            )

                        # Handle specific tool calls
                        result = None

                        if function_name == "reset_memory":
                            result = await self._execute_reset_memory(
                                function_args, chat_history
                            )
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=result,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                        current_tool_count = tool_call_success_counts.get(
                            function_name, 0
                        )
                        if current_tool_count >= MAX_TOOL_CALLS:
                            logger.warning(
                                f"Tool call limit reached for {function_name}: {current_tool_count}/{MAX_TOOL_CALLS}"
                            )
                            # Add tool message with error and return immediately
                            tool_message = ToolMessage(
                                content=f"Tool call limit reached for '{function_name}'. This tool has been called {current_tool_count} times, which exceeds the maximum of {MAX_TOOL_CALLS}. Please don't reinvoke it.",
                                name=function_name,
                                tool_call_id=tool_call_id,
                                status="success",
                                additional_kwargs={"input": function_args},
                            )
                            working_messages.append(tool_message)
                            # Continue to next tool call or iteration
                            continue

                        # Handle custom tools (including MCP server tools)
                        tool_names = [
                            tool.get("function", {}).get("name", "") for tool in tools
                        ]
                        if function_name in tool_names:
                            # Try executing via MCP server if server_context is provided
                            try:
                                matched_tool = next(
                                    (
                                        t
                                        for t in tools
                                        if t.get("function", {}).get("name")
                                        == function_name
                                    ),
                                    None,
                                )
                                server_ctx = (
                                    (matched_tool or {}).get("server_context")
                                    if matched_tool
                                    else None
                                )
                                if server_ctx:
                                    logger.debug(
                                        f"[DEBUG-MCP] [MCP] non_streaming call name={function_name} url={server_ctx.get('url','')} token_present={bool(server_ctx.get('token'))}"
                                    )
                                    result = await execute_tool_server(
                                        token=server_ctx.get("token", ""),
                                        url=server_ctx.get("url", ""),
                                        name=function_name,
                                        params=function_args,
                                        server_data=server_ctx.get("server_data", {}),
                                    )

                                    tool_message = ToolMessage(
                                        content=(
                                            safe_json_dumps(result)
                                            if not isinstance(result, str)
                                            else result
                                        ),
                                        name=function_name,
                                        tool_call_id=tool_call_id,
                                    )
                                    working_messages.append(tool_message)

                                    final_response = (
                                        safe_json_dumps(result)
                                        if not isinstance(result, str)
                                        else result
                                    )

                                    # Only return immediately if this tool is marked as direct response
                                    if (
                                        is_direct_response
                                        and len(response.tool_calls) == 1
                                    ):
                                        return AnswerChunk(
                                            thread_id=thread_id,
                                            message=final_response,
                                            is_partial=False,
                                            message_id=message_id,
                                            sources=sources,
                                        )

                                    # Otherwise, continue loop so the model can produce a natural-language reply
                                    continue
                            except Exception as _:
                                pass

                            # Fallback: original behavior (UI-driven custom tools)
                            logger.debug(
                                f"[DEBUG-MCP] [MCP] non_streaming fallback (no server_context) for tool={function_name}"
                            )
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=clean_thinking_tags(response.content or ""),
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                        # Handle workflow triggers
                        if is_error:
                            result = f"missing required data {function_args} to execute the function {function_name}"
                        else:
                            result = await self._async_trigger_workflow(
                                organization_id=organization_id,
                                workflow_id=str(workflow_id),
                                method=method,
                                data=function_args,
                            )

                        logger.info(f"Tool call result: {result}")

                        # Add tool message to conversation
                        tool_message = ToolMessage(
                            content=(
                                safe_json_dumps(result)
                                if not isinstance(result, str)
                                else result
                            ),
                            name=function_name,
                            status="error" if is_error else "success",
                            tool_call_id=tool_call_id,
                        )
                        tool_call_success_counts[function_name] = (
                            tool_call_success_counts.get(function_name, 0) + 1
                        )
                        logger.info(
                            f"Tool '{function_name}' successful call count: {tool_call_success_counts[function_name]}/{MAX_TOOL_CALLS}"
                        )

                        logger.info(f"Appending tool message: {tool_message}")
                        working_messages.append(tool_message)

                        # Check if response is too long
                        final_response = (
                            safe_json_dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                        if len(final_response) > MAX_RESPONSE_TOKENS:
                            # In a real implementation, you'd upload to file storage here
                            # For now, we'll truncate and add a notice
                            truncated_response = final_response[:MAX_RESPONSE_TOKENS]
                            file_notice = f"\n\n[Note: Response truncated due to length. Full response was {len(final_response)} characters.]"
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=truncated_response + file_notice,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                        # Check if this is a direct response
                        if is_direct_response and len(response.tool_calls) == 1:
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=final_response,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                    # After processing all tool calls, continue the loop to get the next response
                    continue

                # Continue the loop if we haven't returned yet
                # The loop will continue until the AI naturally completes or stops

            # Safety net: if we somehow exit the loop without returning
            logger.warning(
                f"Safety limit reached ({SAFETY_MAX_ITERATIONS} iterations) - returning last response"
            )
            # Try to return the last response content if available
            last_ai_message = None
            for msg in reversed(working_messages):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break

            return AnswerChunk(
                thread_id=thread_id,
                message=(
                    last_ai_message.content
                    if last_ai_message
                    else "Processing completed"
                ),
                is_partial=False,
                message_id=message_id,
                sources=sources,
            )

        except Exception as error:
            logger.exception("Non-streaming error")
            error_message = str(error) or repr(error)
            return AnswerChunk(
                thread_id=thread_id,
                message=error_message,
                is_partial=False,
                message_id=message_id,
                sources=sources,
            )

    async def _process_non_streaming_response_v2(
        self,
        inputs: AnswerQuestionInputs,
        prompt_json: List[Dict],
        thread_id: str,
        message_id: str,
        model_with_tools,
        workflow_settings: List[Dict],
        tools: List[Dict],
        sources: List[Dict],
        chat_history: MongoDBChatMessageHistory,
        use_memory: bool = True,
    ) -> AnswerChunk:
        """Process non-streaming response from LLM with workflow trigger integration"""
        try:

            # Create working copy of prompt messages
            working_messages = prompt_json
            tool_call_success_counts = {}

            # Safety counter to prevent infinite loops
            safety_counter = 0
            logger.info(
                f"Starting non-streaming response processing for thread {thread_id} with message ID {message_id}"
            )
            # Continue processing until the AI naturally completes or stops
            while safety_counter < SAFETY_MAX_ITERATIONS:
                safety_counter += 1

                # Time the model invocation
                start_time = time.time()
                response = await model_with_tools.ainvoke(working_messages)
                end_time = time.time()

                logger.info(f"Assistant response: {response}")
                logger.info(f"Eval duration: {(end_time - start_time) * 1000:.2f}ms")

                if not response:
                    return AnswerChunk(
                        thread_id=thread_id,
                        message="",
                        is_partial=True,
                        message_id=message_id,
                        sources=sources,
                    )

                # Check if we should stop (no tool calls or stop reason)
                should_stop = False

                # First, check if there are tool calls to process
                has_tool_calls = (
                    hasattr(response, "tool_calls")
                    and response.tool_calls
                    and len(response.tool_calls) > 0
                )

                # If no tool calls, check for stop conditions
                if not has_tool_calls:
                    # Check various finish reasons
                    finish_reason = None
                    if (
                        hasattr(response, "response_metadata")
                        and response.response_metadata
                    ):
                        finish_reason = response.response_metadata.get(
                            "finish_reason"
                        ) or response.response_metadata.get("done_reason")

                    if (
                        hasattr(response, "additional_kwargs")
                        and response.additional_kwargs
                    ):
                        finish_reason = (
                            finish_reason
                            or response.additional_kwargs.get(
                                "finishReason", ""
                            ).lower()
                        )

                    # Stop if we have a stop reason, or if there's simply no more tool calls and we have content
                    content_text = (
                        extract_text_from_content(response.content)
                        if hasattr(response, "content")
                        else ""
                    )
                    if finish_reason == "stop" or (
                        hasattr(response, "content")
                        and content_text
                        and content_text.strip()
                    ):
                        should_stop = True

                if should_stop:
                    # Add final AI message and save to history
                    content_text = (
                        extract_text_from_content(response.content)
                        if hasattr(response, "content")
                        else ""
                    )
                    if hasattr(response, "content") and content_text:
                        working_messages.append(AIMessage(content=content_text))
                    final_prompt = custom_trim_messages(working_messages, max_tokens=50)
                    new_history = validate_tool_calls(final_prompt)
                    # Save to chat history
                    if chat_history and use_memory:
                        chat_history.clear()
                        for msg in new_history:
                            if isinstance(msg, dict):
                                if msg.get("role") == "user":
                                    chat_history.add_user_message(
                                        msg.get("content", "")
                                    )
                                elif msg.get("role") == "assistant":
                                    chat_history.add_ai_message(msg.get("content", ""))
                            elif isinstance(msg, HumanMessage):
                                chat_history.add_user_message(msg.content)
                            elif isinstance(msg, AIMessage):
                                chat_history.add_ai_message(msg.content)
                    if (
                        inputs.message_id
                        and inputs.conversation_id
                        and inputs.message_created_at
                        and content_text.startswith("bi_")
                    ):
                        board_message_model.create(
                            organization_id=inputs.organization_id,
                            details=BoardMessageForm(
                                conversation_id=inputs.conversation_id,
                                message_id=inputs.message_id,
                                board_item_id=inputs.board_item_id,
                                created_at=inputs.message_created_at,
                            ),
                        )

                    return AnswerChunk(
                        thread_id=thread_id,
                        message=clean_thinking_tags(content_text or ""),
                        is_partial=True,
                        message_id=message_id,
                        sources=sources,
                    )

                # Process tool calls
                if has_tool_calls:
                    # Add the AI message with tool calls
                    working_messages.append(
                        AIMessage(
                            content=response.content or "",
                            tool_calls=response.tool_calls,
                        )
                    )

                    for tool_call in response.tool_calls:
                        is_error = False
                        # Handle both dict and object formats for tool_call
                        if hasattr(tool_call, "name"):
                            function_name = tool_call.name
                            function_args = (
                                tool_call.args if hasattr(tool_call, "args") else {}
                            )
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                        else:
                            function_name = tool_call.get("name", "")
                            function_args = tool_call.get("args", {})
                            tool_call_id = tool_call.get("id", "")

                        logger.info(f"Processing tool call: {tool_call}")

                        # Extract workflow ID from function name
                        workflow_id = None
                        if "_" in function_name:
                            try:
                                workflow_id = function_name.split("_")[0]
                            except ValueError:
                                pass

                        logger.info(
                            f"Extracted workflow_id: {workflow_id} from function_name: {function_name}"
                        )
                        if not workflow_id:
                            is_error = True

                        # Find workflow settings
                        settings = None
                        if workflow_settings:
                            settings = next(
                                (
                                    ws
                                    for ws in workflow_settings
                                    if ws.get("function", {}).get("name")
                                    == function_name
                                ),
                                None,
                            )

                        organization_id = None
                        if settings:
                            organization_id = settings.get("function", {}).get(
                                "organization_id"
                            )

                        if not organization_id:
                            is_error = True

                        method = "POST"
                        is_direct_response = False
                        if settings:
                            method = settings.get("function", {}).get("method", "POST")
                            is_direct_response = settings.get("function", {}).get(
                                "is_direct_response", False
                            )

                        # Handle specific tool calls
                        result = None

                        if function_name == "reset_memory":
                            result = await self._execute_reset_memory(
                                function_args, chat_history
                            )
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=result,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                        current_tool_count = tool_call_success_counts.get(
                            function_name, 0
                        )
                        if current_tool_count >= MAX_TOOL_CALLS:
                            logger.warning(
                                f"Tool call limit reached for {function_name}: {current_tool_count}/{MAX_TOOL_CALLS}"
                            )
                            # Add tool message with error and return immediately
                            tool_message = ToolMessage(
                                content=f"Tool call limit reached for '{function_name}'. This tool has been called {current_tool_count} times, which exceeds the maximum of {MAX_TOOL_CALLS}. Please don't reinvoke it.",
                                name=function_name,
                                tool_call_id=tool_call_id,
                                status="success",
                                additional_kwargs={"input": function_args},
                            )
                            working_messages.append(tool_message)
                            # Continue to next tool call or iteration
                            continue

                        # Handle custom tools (including MCP server tools)
                        tool_names = [
                            tool.get("function", {}).get("name", "") for tool in tools
                        ]
                        if function_name in tool_names:
                            # Try executing via MCP server if server_context is provided
                            try:
                                matched_tool = next(
                                    (
                                        t
                                        for t in tools
                                        if t.get("function", {}).get("name")
                                        == function_name
                                    ),
                                    None,
                                )
                                server_ctx = (
                                    (matched_tool or {}).get("server_context")
                                    if matched_tool
                                    else None
                                )
                                if server_ctx:
                                    logger.debug(
                                        f"[DEBUG-MCP] [MCP] non_streaming call name={function_name} url={server_ctx.get('url','')} token_present={bool(server_ctx.get('token'))}"
                                    )
                                    result = await execute_tool_server(
                                        token=server_ctx.get("token", ""),
                                        url=server_ctx.get("url", ""),
                                        name=function_name,
                                        params=function_args,
                                        server_data=server_ctx.get("server_data", {}),
                                    )

                                    tool_message = ToolMessage(
                                        content=(
                                            safe_json_dumps(result)
                                            if not isinstance(result, str)
                                            else result
                                        ),
                                        name=function_name,
                                        tool_call_id=tool_call_id,
                                    )
                                    working_messages.append(tool_message)

                                    final_response = (
                                        safe_json_dumps(result)
                                        if not isinstance(result, str)
                                        else result
                                    )

                                    # Only return immediately if this tool is marked as direct response
                                    if (
                                        is_direct_response
                                        and len(response.tool_calls) == 1
                                    ):
                                        return AnswerChunk(
                                            thread_id=thread_id,
                                            message=final_response,
                                            is_partial=False,
                                            message_id=message_id,
                                            sources=sources,
                                        )

                                    # Otherwise, continue loop so the model can produce a natural-language reply
                                    continue
                            except Exception as _:
                                pass

                            # Fallback: original behavior (UI-driven custom tools)
                            logger.debug(
                                f"[DEBUG-MCP] [MCP] non_streaming fallback (no server_context) for tool={function_name}"
                            )
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=clean_thinking_tags(response.content or ""),
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                        # Handle workflow triggers
                        if is_error:
                            result = f"missing required data {function_args} to execute the function {function_name}"
                        else:
                            result = await self._async_trigger_workflow(
                                organization_id=organization_id,
                                workflow_id=str(workflow_id),
                                method=method,
                                data=function_args,
                            )

                        logger.info(f"Tool call result: {result}")

                        # Add tool message to conversation
                        tool_message = ToolMessage(
                            content=(
                                safe_json_dumps(result)
                                if not isinstance(result, str)
                                else result
                            ),
                            name=function_name,
                            status="error" if is_error else "success",
                            tool_call_id=tool_call_id,
                        )
                        tool_call_success_counts[function_name] = (
                            tool_call_success_counts.get(function_name, 0) + 1
                        )
                        logger.info(
                            f"Tool '{function_name}' successful call count: {tool_call_success_counts[function_name]}/{MAX_TOOL_CALLS}"
                        )

                        working_messages.append(tool_message)

                        # Check if response is too long
                        final_response = (
                            safe_json_dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                        if len(final_response) > MAX_RESPONSE_TOKENS:
                            # In a real implementation, you'd upload to file storage here
                            # For now, we'll truncate and add a notice
                            truncated_response = final_response[:MAX_RESPONSE_TOKENS]
                            file_notice = f"\n\n[Note: Response truncated due to length. Full response was {len(final_response)} characters.]"
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=truncated_response + file_notice,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                        # Check if this is a direct response
                        if is_direct_response and len(response.tool_calls) == 1:
                            return AnswerChunk(
                                thread_id=thread_id,
                                message=final_response,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )

                    # After processing all tool calls, continue the loop to get the next response
                    continue

                # Continue the loop if we haven't returned yet
                # The loop will continue until the AI naturally completes or stops

            # Safety net: if we somehow exit the loop without returning
            logger.warning(
                f"Safety limit reached ({SAFETY_MAX_ITERATIONS} iterations) - returning last response"
            )
            # Try to return the last response content if available
            last_ai_message = None
            for msg in reversed(working_messages):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break

            return AnswerChunk(
                thread_id=thread_id,
                message=(
                    last_ai_message.content
                    if last_ai_message
                    else "Processing completed"
                ),
                is_partial=False,
                message_id=message_id,
                sources=sources,
            )

        except Exception as error:
            logger.error(f"Non-streaming error: {error}")
            return AnswerChunk(
                thread_id=thread_id,
                message=f"There was an error processing your request with llm provider. Please try again later.",
                is_partial=False,
                message_id=message_id,
                sources=sources,
            )

    async def _process_streaming_response_v2(
        self,
        inputs: AnswerQuestionInputs,
        prompt_json: List[Dict],
        thread_id: str,
        message_id: str,
        model_with_tools,
        workflow_settings: List[Dict],
        tools: List[Dict],
        sources: List[Dict],
        chat_history: MongoDBChatMessageHistory,
        use_memory: bool = True,
    ) -> AsyncGenerator[AnswerChunk, None]:
        """Process streaming response from LLM with workflow trigger integration"""
        try:

            # Create working copy of prompt messages
            working_messages = prompt_json
            tool_call_success_counts = {}

            # Safety counter to prevent infinite loops
            safety_counter = 0
            logger.info(
                f"Starting streaming response processing for thread {thread_id} with message ID {message_id}"
            )
            # Continue processing until the AI naturally completes or stops
            while safety_counter < SAFETY_MAX_ITERATIONS:
                safety_counter += 1

                # Time the model invocation
                start_time = time.time()

                # Stream the response from the model
                # Use LangChain's chunk accumulation pattern
                gathered = None
                current_content = ""
                content_chunks = (
                    []
                )  # Buffer content chunks until we know if there are tool calls

                async for chunk in model_with_tools.astream(working_messages):
                    # Accumulate chunks using the + operator (LangChain pattern)
                    gathered = chunk if gathered is None else gathered + chunk
                    # Handle content chunks - buffer them instead of streaming immediately
                    if hasattr(chunk, "content") and chunk.content:
                        # Extract text from content (handles both string and list formats)
                        chunk_text = extract_text_from_content(chunk.content)
                        if chunk_text:
                            current_content += chunk_text
                            content_chunks.append(chunk_text)
                            logger.debug(f"Buffered chunk content: {chunk_text}")

                # After streaming, use the accumulated response
                current_response = gathered

                # Check if response has tool calls - if not, stream the buffered content
                has_tool_calls_in_response = (
                    current_response
                    and hasattr(current_response, "tool_calls")
                    and current_response.tool_calls
                    and len(current_response.tool_calls) > 0
                )

                # Only stream content if there are NO tool calls
                # If there are tool calls, skip streaming and let the tool execution flow handle it
                if not has_tool_calls_in_response and content_chunks:
                    for chunk_content in content_chunks:
                        yield AnswerChunk(
                            thread_id=thread_id,
                            message=chunk_content,
                            is_partial=True,
                            message_id=message_id,
                            sources=(
                                sources
                                if content_chunks.index(chunk_content) == 0
                                else []
                            ),
                        )
                logger.debug(f"Accumulated response: {current_response}")
                if current_response and hasattr(current_response, "tool_calls"):
                    logger.info(
                        f"Accumulated tool calls: {len(current_response.tool_calls) if current_response.tool_calls else 0}"
                    )
                    if current_response.tool_calls:
                        for i, tc in enumerate(current_response.tool_calls):
                            logger.info(
                                f"Accumulated tool call {i}: name={getattr(tc, 'name', None)}, id={getattr(tc, 'id', None)}"
                            )

                end_time = time.time()
                logger.info(
                    f"Streaming eval duration: {(end_time - start_time) * 1000:.2f}ms"
                )

                if not current_response:
                    logger.warning("No response received from model during streaming")
                    yield AnswerChunk(
                        thread_id=thread_id,
                        message="",
                        is_partial=True,
                        message_id=message_id,
                        sources=sources,
                    )
                    return

                # Check if we should stop (no tool calls or stop reason)
                should_stop = False

                # First, check if there are tool calls to process
                has_tool_calls = (
                    hasattr(current_response, "tool_calls")
                    and current_response.tool_calls
                    and len(current_response.tool_calls) > 0
                )

                logger.debug(f"Tool call detection - has_tool_calls: {has_tool_calls}")
                if has_tool_calls:
                    logger.info(
                        f"Number of tool calls: {len(current_response.tool_calls)}"
                    )
                    for i, tc in enumerate(current_response.tool_calls):
                        # Enhanced debugging to see the actual structure
                        logger.info(f"Tool call {i} raw object: {tc}")
                        logger.info(f"Tool call {i} type: {type(tc)}")
                        if hasattr(tc, "__dict__"):
                            logger.info(f"Tool call {i} attributes: {tc.__dict__}")

                        # Try different ways to get the name
                        name = None
                        if hasattr(tc, "name"):
                            name = tc.name
                        elif hasattr(tc, "function") and hasattr(tc.function, "name"):
                            name = tc.function.name
                        elif isinstance(tc, dict):
                            name = tc.get("name") or tc.get("function", {}).get("name")

                        logger.info(f"Tool call {i} extracted name: {name}")

                # If no tool calls, check for stop conditions
                if not has_tool_calls:
                    # Check various finish reasons
                    finish_reason = None
                    if (
                        hasattr(current_response, "response_metadata")
                        and current_response.response_metadata
                    ):
                        finish_reason = current_response.response_metadata.get(
                            "finish_reason"
                        ) or current_response.response_metadata.get("done_reason")

                    if (
                        hasattr(current_response, "additional_kwargs")
                        and current_response.additional_kwargs
                    ):
                        finish_reason = (
                            finish_reason
                            or current_response.additional_kwargs.get(
                                "finishReason", ""
                            ).lower()
                        )

                    # Stop if we have a stop reason, or if there's simply no more tool calls and we have content
                    if finish_reason == "stop" or (
                        hasattr(current_response, "content")
                        and current_content
                        and current_content.strip()
                    ):
                        should_stop = True

                if should_stop:
                    # Add final AI message and save to history
                    if hasattr(current_response, "content") and current_content:
                        working_messages.append(AIMessage(content=current_content))
                    final_prompt = custom_trim_messages(working_messages, max_tokens=50)
                    new_history = validate_tool_calls(final_prompt)
                    # Save to chat history
                    if chat_history and use_memory:
                        chat_history.clear()
                        for msg in new_history:
                            if isinstance(msg, dict):
                                if msg.get("role") == "user":
                                    chat_history.add_user_message(
                                        msg.get("content", "")
                                    )
                                elif msg.get("role") == "assistant":
                                    chat_history.add_ai_message(msg.get("content", ""))
                            elif isinstance(msg, HumanMessage):
                                chat_history.add_user_message(msg.content)
                            elif isinstance(msg, AIMessage):
                                chat_history.add_ai_message(msg.content)
                    if (
                        inputs.message_id
                        and inputs.conversation_id
                        and inputs.message_created_at
                        and current_content.startswith("bi_")
                    ):
                        board_message_model.create(
                            organization_id=inputs.organization_id,
                            details=BoardMessageForm(
                                conversation_id=inputs.conversation_id,
                                message_id=inputs.message_id,
                                board_item_id=inputs.board_item_id,
                                created_at=inputs.message_created_at,
                            ),
                        )

                    # yield AnswerChunk(
                    #     thread_id=thread_id,
                    #     message=current_content or "",
                    #     is_partial=False,
                    #     message_id=message_id,
                    #     sources=sources,
                    # )
                    return

                # Process tool calls
                if has_tool_calls:
                    # Add the AI message with tool calls
                    working_messages.append(
                        AIMessage(
                            content=current_content or "",
                            tool_calls=current_response.tool_calls,
                        )
                    )

                    for tool_call in current_response.tool_calls:
                        is_error = False
                        function_name = ""
                        function_args = {}
                        tool_call_id = ""

                        # Enhanced tool call parsing to handle various formats
                        logger.info(f"Processing tool call raw: {tool_call}")
                        logger.info(f"Tool call type: {type(tool_call)}")

                        # Method 1: Direct attributes (LangChain format or our reconstructed format)
                        if hasattr(tool_call, "name") and tool_call.name:
                            function_name = tool_call.name
                            function_args = (
                                tool_call.args
                                if hasattr(tool_call, "args") and tool_call.args
                                else {}
                            )
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                            logger.info(
                                f"Method 1 - name: {function_name}, args: {function_args}"
                            )

                        # Method 2: Nested function attribute (OpenAI format)
                        elif hasattr(tool_call, "function") and tool_call.function:
                            func = tool_call.function
                            function_name = getattr(func, "name", "")
                            # Handle args that might be JSON string or dict
                            args = getattr(func, "arguments", {})
                            if isinstance(args, str):
                                try:
                                    function_args = (
                                        json.loads(args) if args.strip() else {}
                                    )
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse JSON arguments: {args}"
                                    )
                                    function_args = {}
                            else:
                                function_args = args
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                            logger.info(
                                f"Method 2 - name: {function_name}, args: {function_args}"
                            )

                        # Method 3: Dictionary format
                        elif isinstance(tool_call, dict):
                            function_name = tool_call.get("name", "")
                            if not function_name and "function" in tool_call:
                                func = tool_call["function"]
                                function_name = func.get("name", "")
                                args = func.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        function_args = (
                                            json.loads(args) if args.strip() else {}
                                        )
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to parse JSON arguments: {args}"
                                        )
                                        function_args = {}
                                else:
                                    function_args = args if args else {}
                            else:
                                # Try multiple possible argument field names
                                function_args = None
                                logger.debug(
                                    f"Checking for args in tool_call keys: {list(tool_call.keys())}"
                                )

                                for arg_field in [
                                    "args",
                                    "arguments",
                                    "parameters",
                                    "input",
                                    "params",
                                ]:
                                    if arg_field in tool_call:
                                        potential_args = tool_call[arg_field]
                                        logger.debug(
                                            f"Found {arg_field}: {potential_args} (type: {type(potential_args)})"
                                        )

                                        if isinstance(potential_args, str):
                                            try:
                                                function_args = (
                                                    json.loads(potential_args)
                                                    if potential_args.strip()
                                                    else {}
                                                )
                                                logger.debug(
                                                    f"Successfully parsed JSON from {arg_field}"
                                                )
                                                break
                                            except json.JSONDecodeError:
                                                logger.debug(
                                                    f"Failed to parse JSON from {arg_field}"
                                                )
                                                continue
                                        elif isinstance(potential_args, dict):
                                            function_args = potential_args
                                            logger.debug(f"Using dict from {arg_field}")
                                            break

                                # If no args found in any field, default to empty dict
                                if function_args is None:
                                    function_args = {}
                                    logger.warning(
                                        f"No arguments found in any field. Available fields: {list(tool_call.keys())}"
                                    )

                            tool_call_id = tool_call.get("id", "")
                            logger.debug(f"Method 3 - final args: {function_args}")

                        # Method 4: Check for type field indicating different structure or incomplete tool call
                        elif (
                            hasattr(tool_call, "type") and tool_call.type == "tool_call"
                        ):
                            # This might be a partially formed tool call, skip it
                            logger.warning(
                                f"Skipping incomplete tool call: {tool_call}"
                            )
                            continue
                        else:
                            # Unknown format, try best effort extraction
                            logger.warning(
                                f"Unknown tool call format: {tool_call}, attempting best effort extraction"
                            )
                            function_name = ""
                            function_args = {}
                            tool_call_id = ""

                            # Try to extract name from common attributes
                            if hasattr(tool_call, "__dict__"):
                                attrs = tool_call.__dict__
                                logger.debug(f"Tool call attributes: {attrs}")
                                function_name = attrs.get("name", "")
                                function_args = attrs.get(
                                    "args", attrs.get("arguments", {})
                                )
                                tool_call_id = attrs.get("id", "")

                            logger.info(
                                f"Best effort extraction - name: {function_name}, args: {function_args}"
                            )

                        logger.info(
                            f"Parsed - name: '{function_name}', args: {function_args}, id: '{tool_call_id}'"
                        )

                        # Skip if we could not extract a function name
                        if not function_name or function_name.strip() == "":
                            logger.warning(
                                f"Skipping tool call with empty name: {tool_call}"
                            )
                            continue
                        # Extract workflow ID from function name
                        workflow_id = None
                        if "_" in function_name:
                            try:
                                workflow_id = function_name.split("_")[0]
                            except ValueError:
                                pass
                        logger.info(
                            f"Extracted workflow_id: {workflow_id} from function_name: {function_name}"
                        )
                        if not workflow_id:
                            is_error = True

                        # Find workflow settings
                        settings = None
                        if workflow_settings:
                            settings = next(
                                (
                                    ws
                                    for ws in workflow_settings
                                    if ws.get("function", {}).get("name")
                                    == function_name
                                ),
                                None,
                            )

                        organization_id = None
                        if settings:
                            organization_id = settings.get("function", {}).get(
                                "organization_id"
                            )

                        if not organization_id:
                            is_error = True

                        method = "POST"
                        is_direct_response = False
                        if settings:
                            method = settings.get("function", {}).get("method", "POST")
                            is_direct_response = settings.get("function", {}).get(
                                "is_direct_response", False
                            )

                        # Handle specific tool calls
                        result = None

                        if function_name == "reset_memory":
                            result = await self._execute_reset_memory(
                                function_args, chat_history
                            )
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=result,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        current_tool_count = tool_call_success_counts.get(
                            function_name, 0
                        )
                        if current_tool_count >= MAX_TOOL_CALLS:
                            logger.warning(
                                f"Tool call limit reached for {function_name}: {current_tool_count}/{MAX_TOOL_CALLS}"
                            )
                            # Add tool message with error and return immediately
                            tool_message = ToolMessage(
                                content=f"Tool call limit reached for '{function_name}'. This tool has been called {current_tool_count} times, which exceeds the maximum of {MAX_TOOL_CALLS}. Please don't reinvoke it.",
                                name=function_name,
                                tool_call_id=tool_call_id,
                                status="success",
                                additional_kwargs={"input": function_args},
                            )
                            working_messages.append(tool_message)
                            # Continue to next tool call or iteration
                            continue

                        # Handle custom tools (including MCP server tools)
                        tool_names = [
                            tool.get("function", {}).get("name", "") for tool in tools
                        ]
                        if function_name in tool_names:
                            # Try executing via MCP server if server_context is provided
                            try:
                                matched_tool = next(
                                    (
                                        t
                                        for t in tools
                                        if t.get("function", {}).get("name")
                                        == function_name
                                    ),
                                    None,
                                )
                                server_ctx = (
                                    (matched_tool or {}).get("server_context")
                                    if matched_tool
                                    else None
                                )
                                if server_ctx:
                                    logger.debug(
                                        f"[DEBUG-MCP] [MCP] non_streaming (alt path) call name={function_name} url={server_ctx.get('url','')} token_present={bool(server_ctx.get('token'))}"
                                    )
                                    result = await execute_tool_server(
                                        token=server_ctx.get("token", ""),
                                        url=server_ctx.get("url", ""),
                                        name=function_name,
                                        params=function_args,
                                        server_data=server_ctx.get("server_data", {}),
                                    )

                                    tool_message = ToolMessage(
                                        content=(
                                            safe_json_dumps(result)
                                            if not isinstance(result, str)
                                            else result
                                        ),
                                        name=function_name,
                                        tool_call_id=tool_call_id,
                                        additional_kwargs={"input": function_args},
                                    )
                                    working_messages.append(tool_message)

                                    final_response = (
                                        safe_json_dumps(result)
                                        if not isinstance(result, str)
                                        else result
                                    )
                                    # Respect direct-response; otherwise let model continue to respond naturally
                                    if (
                                        is_direct_response
                                        and len(current_response.tool_calls) == 1
                                    ):
                                        yield AnswerChunk(
                                            thread_id=thread_id,
                                            message=final_response,
                                            is_partial=False,
                                            message_id=message_id,
                                            sources=sources,
                                        )
                                        return

                                    # Tool executed successfully, continue conversation loop
                                    continue
                            except Exception as _:
                                pass

                            # Fallback: original behavior (UI-driven custom tools)
                            # For custom tools, we need to let the frontend handle them
                            # Return current content and let the UI process the tool call
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=current_content or "",
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Handle workflow triggers
                        if is_error:
                            result = f"missing required data {function_args} to execute the function {function_name}"
                        else:
                            result = await self._async_trigger_workflow(
                                organization_id=organization_id,
                                workflow_id=str(workflow_id),
                                method=method,
                                data=function_args,
                            )

                        logger.info(f"Tool call result: {result}")

                        # Add tool message to conversation
                        tool_message = ToolMessage(
                            content=(
                                safe_json_dumps(result)
                                if not isinstance(result, str)
                                else result
                            ),
                            name=function_name,
                            tool_call_id=tool_call_id,
                            status="error" if is_error else "success",
                        )

                        tool_call_success_counts[function_name] = (
                            tool_call_success_counts.get(function_name, 0) + 1
                        )
                        logger.info(
                            f"Tool '{function_name}' successful call count: {tool_call_success_counts[function_name]}/{MAX_TOOL_CALLS}"
                        )
                        working_messages.append(tool_message)

                        # Check if response is too long
                        final_response = (
                            safe_json_dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                        if len(final_response) > MAX_RESPONSE_TOKENS:
                            # In a real implementation, you'd upload to file storage here
                            # For now, we'll truncate and add a notice
                            truncated_response = final_response[:MAX_RESPONSE_TOKENS]
                            file_notice = f"\n\n[Note: Response truncated due to length. Full response was {len(final_response)} characters.]"
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=truncated_response + file_notice,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Check if this is a direct response
                        if is_direct_response and len(current_response.tool_calls) == 1:
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=final_response,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                    # After processing all tool calls, continue the loop to get the next response
                    continue

                # Continue the loop if we haven't returned yet
                # The loop will continue until the AI naturally completes or stops

            # Safety net: if we somehow exit the loop without returning
            logger.warning(
                f"Safety limit reached ({SAFETY_MAX_ITERATIONS} iterations) - returning last response"
            )
            # Try to return the last response content if available
            last_ai_message = None
            for msg in reversed(working_messages):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break

            # yield AnswerChunk(
            #     thread_id=thread_id,
            #     message=(
            #         last_ai_message.content
            #         if last_ai_message
            #         else "Processing completed"
            #     ),
            #     is_partial=False,
            #     message_id=message_id,
            #     sources=sources,
            # )

        except Exception as error:
            logger.error(f"Streaming error: {error}")
            yield AnswerChunk(
                thread_id=thread_id,
                message=f"Error: {str(error)}",
                is_partial=False,
                message_id=message_id,
                sources=sources,
            )

    async def _process_streaming_response_v3(
        self,
        inputs: AnswerQuestionInputs,
        prompt_json: List[Dict],
        thread_id: str,
        message_id: str,
        model_with_tools,
        workflow_settings: List[Dict],
        tools: List[Dict],
        sources: List[Dict],
        chat_history: MongoDBChatMessageHistory,
        use_memory: bool = True,
    ) -> AsyncGenerator[AnswerChunk, None]:
        """Process streaming response from LLM with workflow trigger integration"""
        try:

            # Create working copy of prompt messages
            working_messages = prompt_json

            # Safety counter to prevent infinite loops
            safety_counter = 0
            # Track successful tool call counts per tool name to enforce MAX_TOOL_CALLS limit
            tool_call_success_counts = {}
            logger.info(
                f"Starting streaming response processing for thread {thread_id} with message ID {message_id}"
            )
            # Continue processing until the AI naturally completes or stops
            while safety_counter < SAFETY_MAX_ITERATIONS:
                safety_counter += 1

                # Time the model invocation
                start_time = time.time()

                # Stream the response from the model
                # Convert working_messages to strands format
                current_content = ""
                # logger.info(f"working messages: {working_messages}")

                # Extract and set system prompt if available
                system_prompt = None
                for msg in working_messages:
                    if isinstance(msg, dict) and msg.get("role") == "system":
                        system_prompt = msg.get("content")
                        break
                    elif isinstance(msg, SystemMessage):
                        system_prompt = msg.content
                        break

                if system_prompt:
                    model_with_tools.system_prompt = system_prompt
                    logger.info(f"Set system prompt on model: {system_prompt[:100]}...")

                # Build conversation history in strands format: list of dicts with role and content
                # IMPORTANT: Omit system messages - they are set via model_with_tools.system_prompt
                # AWS Bedrock and other providers only accept 'user' or 'assistant' roles in conversation
                conversation_history = []
                logger.info(
                    f"{len(working_messages)} working messages to convert to strands format"
                )
                for msg in working_messages:
                    if isinstance(msg, dict):
                        # Skip system messages - already set on model.system_prompt
                        if msg.get("role") == "system":
                            continue
                        # Already in dict format
                        if "role" in msg and "content" in msg:
                            # Check if content is already in strands format [{"text": "..."}]
                            if isinstance(msg["content"], str):
                                conversation_history.append(
                                    {
                                        "role": msg["role"],
                                        "content": [{"text": msg["content"]}],
                                    }
                                )
                            else:
                                # Already in proper format
                                conversation_history.append(msg)
                    elif isinstance(msg, SystemMessage):
                        # Skip system messages - already set on model.system_prompt
                        continue
                    elif isinstance(msg, HumanMessage):
                        logger.info(f"add Human message content: {msg.content}")
                        conversation_history.append(
                            {"role": "user", "content": [{"text": msg.content}]}
                        )
                    elif isinstance(msg, ToolMessage):
                        input = (
                            msg.additional_kwargs.get("input", {})
                            if msg.additional_kwargs
                            else {}
                        )
                        logger.info(
                            f"tool status : {msg.status} for tool call id: {msg.tool_call_id} with name: {input}"
                        )
                        conversation_history.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "toolUse": {
                                            "toolUseId": msg.tool_call_id,
                                            "name": msg.name,
                                            "input": input,
                                        }
                                    }
                                ],
                            }
                        )
                        conversation_history.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "toolResult": {
                                            "toolUseId": msg.tool_call_id,
                                            "status": msg.status,
                                            "content": [{"text": msg.content}],
                                        }
                                    }
                                ],
                            }
                        )
                        conversation_history.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "text": f"agent.tool.{msg.name} was called. If the tool result meets your needs, please don't call tool {msg.name} again."
                                    }
                                ],
                            }
                        )
                    elif isinstance(msg, AIMessage):
                        logger.info(f"add AI message content:")
                        conversation_history.append(
                            {"role": "assistant", "content": [{"text": msg.content}]}
                        )

                # logger.info(f"Converted conversation history (system messages omitted): {conversation_history}")

                # Track tool calls being built incrementally
                tool_calls_in_progress = {}
                # Track reasoning content from thinking models
                accumulated_reasoning = []
                # Track if we should stop (set by complete/force_stop events)
                should_stop = False
                event_count = 0
                logger.debug(
                    f"Beginning to stream events from model with tools... {model_with_tools}"
                )
                # Use the conversation history with the agent
                async for event in model_with_tools.stream_async(conversation_history):
                    event_count += 1
                    logger.debug(f"Received event #{event_count}: {event}")
                    logger.debug(
                        f"Event keys: {event.keys() if isinstance(event, dict) else 'N/A'}"
                    )

                    # Handle lifecycle events
                    if event.get("init_event_loop", False):
                        logger.info("🔄 Event loop initialized")
                        continue

                    elif event.get("start_event_loop", False):
                        logger.info("▶️ Event loop cycle starting")
                        continue

                    elif "message" in event:
                        msg = event["message"]
                        logger.info(
                            f"📬 New message created: {msg.get('role', 'unknown')}"
                        )
                        continue

                    elif event.get("complete", False):
                        should_stop = True
                        logger.info("✅ Cycle completed")
                        continue

                    elif event.get("force_stop", False):
                        should_stop = True
                        reason = event.get("force_stop_reason", "unknown reason")
                        logger.warning(f"🛑 Event loop force-stopped: {reason}")
                        break

                    # Handle tool use events (current_tool_use)
                    if "current_tool_use" in event and event["current_tool_use"].get(
                        "name"
                    ):
                        tool_info = event["current_tool_use"]
                        tool_name = tool_info.get("name")
                        logger.info(f"🔧 Using tool: {tool_info}")

                        # Extract tool call ID and name
                        tool_id = (
                            tool_info.get("id")
                            or tool_info.get("toolUseId")
                            or tool_info.get("tool_call_id")
                        )

                        if tool_id:
                            # Initialize or update tool call in progress
                            if tool_id not in tool_calls_in_progress:
                                tool_calls_in_progress[tool_id] = {
                                    "id": tool_id,
                                    "name": tool_name,
                                    "args": {},
                                }

                            # Accumulate arguments (may come in chunks) - check for 'input' or 'arguments'
                            args_field = (
                                "input" if "input" in tool_info else "arguments"
                            )
                            if args_field in tool_info:
                                args_data = tool_info[args_field]
                                # Handle incremental JSON string chunks
                                if isinstance(args_data, str):
                                    if (
                                        "args_buffer"
                                        not in tool_calls_in_progress[tool_id]
                                    ):
                                        tool_calls_in_progress[tool_id][
                                            "args_buffer"
                                        ] = ""
                                    tool_calls_in_progress[tool_id][
                                        "args_buffer"
                                    ] += args_data
                                    # Try to parse accumulated JSON
                                    try:
                                        parsed_args = json.loads(
                                            tool_calls_in_progress[tool_id][
                                                "args_buffer"
                                            ]
                                        )
                                        tool_calls_in_progress[tool_id][
                                            "args"
                                        ] = parsed_args
                                        logger.debug(
                                            f"✅ Parsed complete tool args: {parsed_args}"
                                        )
                                    except json.JSONDecodeError:
                                        # Not yet complete, keep accumulating
                                        logger.debug(
                                            f"⏳ Accumulating tool args... ({len(tool_calls_in_progress[tool_id]['args_buffer'])} chars)"
                                        )
                                        pass
                                elif isinstance(args_data, dict):
                                    # Merge dict arguments
                                    tool_calls_in_progress[tool_id]["args"].update(
                                        args_data
                                    )
                                    logger.debug(
                                        f"✅ Updated tool args: {tool_calls_in_progress[tool_id]['args']}"
                                    )

                            logger.debug(
                                f"Tool call in progress: {tool_calls_in_progress[tool_id]}"
                            )
                        logger.info(
                            "Breaking stream_async loop - complete tool calls detected"
                        )
                        break
                    # Handle text data events
                    if "data" in event:
                        data_text = event["data"]
                        # Show snippet in logs
                        data_snippet = data_text[:50] + (
                            "..." if len(data_text) > 50 else ""
                        )
                        logger.debug(f"📟 Text: {data_snippet}")

                        # Accumulate content
                        current_content += data_text

                        # Stream content to user (only if there's actual content)
                        if data_text:  # Don't yield if message is empty
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=data_text,
                                is_partial=True,
                                message_id=message_id,
                                sources=sources if not current_content else [],
                            )

                    # Handle tool_stream_event
                    elif "tool_stream_event" in event:
                        tool_stream = event["tool_stream_event"]
                        logger.debug(f"🔧 Tool stream event: {tool_stream}")
                        # Extract data from tool stream if available
                        if "data" in tool_stream:
                            logger.debug(f"Tool streamed data: {tool_stream['data']}")

                    # Handle reasoning events (for thinking/planning models)
                    elif "reasoning" in event and event["reasoning"]:
                        logger.info(f"🤔 Reasoning event received")
                        reasoning_text = event.get("reasoningText", "")
                        if reasoning_text:
                            snippet = reasoning_text[:50] + (
                                "..." if len(reasoning_text) > 50 else ""
                            )
                            logger.debug(f"🤔 Reasoning: {snippet}")
                            # Accumulate reasoning for final response
                            accumulated_reasoning.append(reasoning_text)
                            # Stream reasoning to user in real-time
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message="",  # No regular message content
                                reasoning=reasoning_text,
                                is_partial=True,
                                message_id=message_id,
                                sources=[],
                            )

                    # Handle result events with complete tool calls and/or reasoning
                    elif "result" in event:
                        logger.info(f"📊 Result event received")
                        result_data = event["result"]

                        # Extract reasoning from result if present
                        if isinstance(result_data, dict):
                            result_reasoning = result_data.get(
                                "reasoning"
                            ) or result_data.get("reasoningText")
                            if result_reasoning:
                                logger.info(
                                    f"🤔 Reasoning from result: {result_reasoning[:100]}..."
                                )
                                accumulated_reasoning.append(result_reasoning)
                                # Stream reasoning from result to user
                                yield AnswerChunk(
                                    thread_id=thread_id,
                                    message="",
                                    reasoning=result_reasoning,
                                    is_partial=True,
                                    message_id=message_id,
                                    sources=[],
                                )

                        # Final result may contain complete tool calls
                        if (
                            isinstance(result_data, dict)
                            and "tool_calls" in result_data
                        ):
                            final_tool_calls = result_data["tool_calls"]
                            logger.info(
                                f"Final tool calls from result: {len(final_tool_calls)} tool(s)"
                            )
                            # Clear in-progress and use these instead
                            tool_calls_in_progress.clear()
                            for tc in final_tool_calls:
                                tc_id = (
                                    tc.get("id")
                                    or tc.get("toolUseId")
                                    or tc.get("tool_call_id")
                                    or str(len(tool_calls_in_progress))
                                )
                                tool_calls_in_progress[tc_id] = {
                                    "id": tc_id,
                                    "name": tc.get("name") or tc.get("tool_name"),
                                    "args": tc.get("args")
                                    or tc.get("input")
                                    or tc.get("arguments", {}),
                                }
                            logger.info(
                                "Breaking stream_async loop - complete tool calls detected"
                            )
                            break

                    # Fallback: Handle events with tool_calls directly embedded
                    if (
                        isinstance(event, dict)
                        and "tool_calls" in event
                        and "current_tool_use" not in event
                    ):
                        logger.info(
                            f"Event with embedded tool_calls: {len(event['tool_calls'])} tool(s) - breaking stream to process tools"
                        )
                        # Process tool calls from event
                        for tc in event["tool_calls"]:
                            tc_id = (
                                tc.get("id")
                                or tc.get("toolUseId")
                                or tc.get("tool_call_id")
                                or str(len(tool_calls_in_progress))
                            )
                            if tc_id not in tool_calls_in_progress:
                                tool_calls_in_progress[tc_id] = {
                                    "id": tc_id,
                                    "name": tc.get("name") or tc.get("tool_name"),
                                    "args": tc.get("args")
                                    or tc.get("input")
                                    or tc.get("arguments", {}),
                                }
                            else:
                                # Update existing
                                if tc.get("name") or tc.get("tool_name"):
                                    tool_calls_in_progress[tc_id]["name"] = tc.get(
                                        "name"
                                    ) or tc.get("tool_name")
                                if (
                                    tc.get("args")
                                    or tc.get("input")
                                    or tc.get("arguments")
                                ):
                                    tool_calls_in_progress[tc_id]["args"] = (
                                        tc.get("args")
                                        or tc.get("input")
                                        or tc.get("arguments", {})
                                    )

                        # Break out of streaming loop since we have complete tool calls
                        # Tool execution will happen after the loop exits
                        logger.info(
                            "Breaking stream_async loop - complete tool calls detected"
                        )
                        break

                # After streaming loop, finalize tool calls and create response object
                logger.info(
                    f"Streaming loop completed - events: {event_count}, content_length: {len(current_content)}, tool_calls: {len(tool_calls_in_progress)}, reasoning_chunks: {len(accumulated_reasoning)}"
                )

                # Log reasoning if present
                if accumulated_reasoning:
                    total_reasoning_length = sum(len(r) for r in accumulated_reasoning)
                    logger.info(
                        f"💭 Total reasoning accumulated: {len(accumulated_reasoning)} chunk(s), {total_reasoning_length} chars"
                    )
                    # Show first snippet
                    if accumulated_reasoning[0]:
                        snippet = accumulated_reasoning[0][:100] + (
                            "..." if len(accumulated_reasoning[0]) > 100 else ""
                        )
                        logger.debug(f"💭 First reasoning chunk: {snippet}")

                # Create a simple object to hold tool call data that our parsing logic can handle
                class ToolCallObject:
                    def __init__(self, id, name, args):
                        self.id = id
                        self.name = name
                        self.args = args

                    def __repr__(self):
                        return f"ToolCallObject(id={self.id}, name={self.name}, args={self.args})"

                # Create minimal response object with content, tool calls, and reasoning
                class ResponseWithToolCalls:
                    def __init__(self, content, tool_calls, reasoning=None):
                        self.content = content
                        self.tool_calls = tool_calls
                        self.reasoning = reasoning  # List of reasoning chunks or None

                # Build tool calls list if any exist
                tool_calls_list = []
                if tool_calls_in_progress:
                    tool_calls_list = [
                        ToolCallObject(id=tc["id"], name=tc["name"], args=tc["args"])
                        for tc in tool_calls_in_progress.values()
                    ]
                    logger.info(
                        f"Finalizing {len(tool_calls_list)} tool calls from incremental events"
                    )

                # Always create a response object (even if no content or tool calls)
                # This prevents the "No response received" warning
                # Include reasoning if any was accumulated
                current_response = ResponseWithToolCalls(
                    current_content,
                    tool_calls_list,
                    reasoning=accumulated_reasoning if accumulated_reasoning else None,
                )
                logger.info(
                    f"Created response - content_length: {len(current_content)}, tool_calls: {len(tool_calls_list)}, reasoning_chunks: {len(accumulated_reasoning) if accumulated_reasoning else 0}"
                )
                logger.debug(
                    f"Accumulated response: {json.dumps(current_response.__dict__, default=str)}"
                )
                if current_response and hasattr(current_response, "tool_calls"):
                    logger.info(
                        f"Accumulated tool calls: {len(current_response.tool_calls) if current_response.tool_calls else 0}"
                    )
                    if current_response.tool_calls:
                        for i, tc in enumerate(current_response.tool_calls):
                            logger.info(
                                f"Accumulated tool call {i}: name={getattr(tc, 'name', None)}, id={getattr(tc, 'id', None)}"
                            )

                end_time = time.time()
                logger.info(
                    f"Streaming eval duration: {(end_time - start_time) * 1000:.2f}ms"
                )

                if not current_response:
                    logger.warning("No response received from model during streaming")
                    yield AnswerChunk(
                        thread_id=thread_id,
                        message="",
                        is_partial=True,
                        message_id=message_id,
                        sources=sources,
                    )
                    return

                # First, check if there are tool calls to process
                has_tool_calls = (
                    hasattr(current_response, "tool_calls")
                    and current_response.tool_calls
                    and len(current_response.tool_calls) > 0
                )

                logger.debug(f"Tool call detection - has_tool_calls: {has_tool_calls}")
                if has_tool_calls:
                    logger.info(
                        f"Number of tool calls: {len(current_response.tool_calls)}"
                    )
                    for i, tc in enumerate(current_response.tool_calls):
                        # Enhanced debugging to see the actual structure
                        logger.info(f"Tool call {i} raw object: {tc}")
                        logger.info(f"Tool call {i} type: {type(tc)}")
                        if hasattr(tc, "__dict__"):
                            logger.info(f"Tool call {i} attributes: {tc.__dict__}")

                        # Try different ways to get the name
                        name = None
                        if hasattr(tc, "name"):
                            name = tc.name
                        elif hasattr(tc, "function") and hasattr(tc.function, "name"):
                            name = tc.function.name
                        elif isinstance(tc, dict):
                            name = tc.get("name") or tc.get("function", {}).get("name")

                        logger.info(f"Tool call {i} extracted name: {name}")

                # If no tool calls, check for stop conditions
                if not has_tool_calls:
                    # Check various finish reasons
                    finish_reason = None
                    if (
                        hasattr(current_response, "response_metadata")
                        and current_response.response_metadata
                    ):
                        finish_reason = current_response.response_metadata.get(
                            "finish_reason"
                        ) or current_response.response_metadata.get("done_reason")

                    if (
                        hasattr(current_response, "additional_kwargs")
                        and current_response.additional_kwargs
                    ):
                        finish_reason = (
                            finish_reason
                            or current_response.additional_kwargs.get(
                                "finishReason", ""
                            ).lower()
                        )

                    # Stop if we have a stop reason, or if there's simply no more tool calls and we have content
                    if finish_reason == "stop" or (
                        hasattr(current_response, "content")
                        and current_content
                        and current_content.strip()
                    ):
                        should_stop = True

                if should_stop:
                    # Add final AI message and save to history
                    if hasattr(current_response, "content") and current_content:
                        working_messages.append(AIMessage(content=current_content))
                    final_prompt = custom_trim_messages(working_messages, max_tokens=50)
                    new_history = validate_tool_calls(final_prompt)
                    logger.debug(
                        f"Final conversation history length: {new_history} messages"
                    )
                    # Save to chat history
                    if chat_history and use_memory:
                        logger.info(f"Saving final response to chat history...")
                        chat_history.clear()
                        for msg in new_history:
                            if isinstance(msg, AIMessage):
                                logger.debug(
                                    f"add AI message to chat history: {msg.content}"
                                )
                                chat_history.add_ai_message(msg.content)
                            elif "content" in msg:
                                logger.debug(f"add user message to chat history: {msg}")
                                chat_history.add_user_message(msg["content"])
                    if (
                        inputs.message_id
                        and inputs.conversation_id
                        and inputs.message_created_at
                        and current_content.startswith("bi_")
                    ):
                        board_message_model.create(
                            organization_id=inputs.organization_id,
                            details=BoardMessageForm(
                                conversation_id=inputs.conversation_id,
                                message_id=inputs.message_id,
                                board_item_id=inputs.board_item_id,
                                created_at=inputs.message_created_at,
                            ),
                        )

                    # yield AnswerChunk(
                    #     thread_id=thread_id,
                    #     message=current_content or "",
                    #     is_partial=False,
                    #     message_id=message_id,
                    #     sources=sources,
                    # )
                    return

                # Process tool calls
                if has_tool_calls:
                    # Convert ToolCallObject instances to dictionaries for LangChain compatibility
                    tool_calls_dicts = []
                    for tc in current_response.tool_calls:
                        if hasattr(tc, "__dict__") and hasattr(tc, "name"):
                            # Convert ToolCallObject to dict format expected by LangChain
                            tool_calls_dicts.append(
                                {
                                    "name": tc.name,
                                    "args": tc.args,
                                    "id": tc.id,
                                    "type": "tool_call",
                                }
                            )
                        elif isinstance(tc, dict):
                            tool_calls_dicts.append(tc)
                        else:
                            logger.warning(f"Unknown tool call format: {tc}")

                    # Add the AI message with tool calls
                    working_messages.append(
                        AIMessage(
                            content=current_content or "",
                            tool_calls=tool_calls_dicts,
                        )
                    )

                    for tool_call in current_response.tool_calls:
                        is_error = False
                        function_name = ""
                        function_args = {}
                        tool_call_id = ""

                        # Enhanced tool call parsing to handle various formats
                        logger.info(f"Processing tool call raw: {tool_call}")
                        logger.info(f"Tool call type: {type(tool_call)}")

                        # Method 1: Direct attributes (LangChain format or our reconstructed format)
                        if hasattr(tool_call, "name") and tool_call.name:
                            function_name = tool_call.name
                            function_args = (
                                tool_call.args
                                if hasattr(tool_call, "args") and tool_call.args
                                else {}
                            )
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                            logger.info(
                                f"Method 1 - name: {function_name}, args: {function_args}"
                            )

                        # Method 2: Nested function attribute (OpenAI format)
                        elif hasattr(tool_call, "function") and tool_call.function:
                            func = tool_call.function
                            function_name = getattr(func, "name", "")
                            # Handle args that might be JSON string or dict
                            args = getattr(func, "arguments", {})
                            if isinstance(args, str):
                                try:
                                    function_args = (
                                        json.loads(args) if args.strip() else {}
                                    )
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse JSON arguments: {args}"
                                    )
                                    function_args = {}
                            else:
                                function_args = args
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                            logger.info(
                                f"Method 2 - name: {function_name}, args: {function_args}"
                            )

                        # Method 3: Dictionary format
                        elif isinstance(tool_call, dict):
                            function_name = tool_call.get("name", "")
                            if not function_name and "function" in tool_call:
                                func = tool_call["function"]
                                function_name = func.get("name", "")
                                args = func.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        function_args = (
                                            json.loads(args) if args.strip() else {}
                                        )
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to parse JSON arguments: {args}"
                                        )
                                        function_args = {}
                                else:
                                    function_args = args if args else {}
                            else:
                                # Try multiple possible argument field names
                                function_args = None
                                logger.debug(
                                    f"Checking for args in tool_call keys: {list(tool_call.keys())}"
                                )

                                for arg_field in [
                                    "args",
                                    "arguments",
                                    "parameters",
                                    "input",
                                    "params",
                                ]:
                                    if arg_field in tool_call:
                                        potential_args = tool_call[arg_field]
                                        logger.debug(
                                            f"Found {arg_field}: {potential_args} (type: {type(potential_args)})"
                                        )

                                        if isinstance(potential_args, str):
                                            try:
                                                function_args = (
                                                    json.loads(potential_args)
                                                    if potential_args.strip()
                                                    else {}
                                                )
                                                logger.debug(
                                                    f"Successfully parsed JSON from {arg_field}"
                                                )
                                                break
                                            except json.JSONDecodeError:
                                                logger.debug(
                                                    f"Failed to parse JSON from {arg_field}"
                                                )
                                                continue
                                        elif isinstance(potential_args, dict):
                                            function_args = potential_args
                                            logger.debug(f"Using dict from {arg_field}")
                                            break

                                # If no args found in any field, default to empty dict
                                if function_args is None:
                                    function_args = {}
                                    logger.warning(
                                        f"No arguments found in any field. Available fields: {list(tool_call.keys())}"
                                    )

                            tool_call_id = tool_call.get("id", "")
                            logger.debug(f"Method 3 - final args: {function_args}")

                        # Method 4: Check for type field indicating different structure or incomplete tool call
                        elif (
                            hasattr(tool_call, "type") and tool_call.type == "tool_call"
                        ):
                            # This might be a partially formed tool call, skip it
                            logger.warning(
                                f"Skipping incomplete tool call: {tool_call}"
                            )
                            continue
                        else:
                            # Unknown format, try best effort extraction
                            logger.warning(
                                f"Unknown tool call format: {tool_call}, attempting best effort extraction"
                            )
                            function_name = ""
                            function_args = {}
                            tool_call_id = ""

                            # Try to extract name from common attributes
                            if hasattr(tool_call, "__dict__"):
                                attrs = tool_call.__dict__
                                logger.debug(f"Tool call attributes: {attrs}")
                                function_name = attrs.get("name", "")
                                function_args = attrs.get(
                                    "args", attrs.get("arguments", {})
                                )
                                tool_call_id = attrs.get("id", "")

                            logger.info(
                                f"Best effort extraction - name: {function_name}, args: {function_args}"
                            )

                        logger.info(
                            f"Parsed - name: '{function_name}', args: {function_args}, id: '{tool_call_id}'"
                        )

                        # Skip if we could not extract a function name
                        if not function_name or function_name.strip() == "":
                            logger.warning(
                                f"Skipping tool call with empty name: {tool_call}"
                            )
                            continue
                        # Extract workflow ID from function name
                        workflow_id = None
                        if "_" in function_name:
                            try:
                                workflow_id = function_name.split("_")[0]
                            except ValueError:
                                pass
                        logger.info(
                            f"Extracted workflow_id: {workflow_id} from function_name: {function_name}"
                        )
                        if not workflow_id:
                            is_error = True

                        # Find workflow settings
                        settings = None
                        if workflow_settings:
                            settings = next(
                                (
                                    ws
                                    for ws in workflow_settings
                                    if ws.get("function", {}).get("name")
                                    == function_name
                                ),
                                None,
                            )
                        logger.debug(
                            f"Found workflow settings: {settings} for function_name: {function_name}"
                        )

                        # Extract required parameters from settings
                        required_params = []
                        if settings:
                            parameters = settings.get("function", {}).get(
                                "parameters", {}
                            )
                            if parameters:
                                # Get required parameters list
                                required_params = parameters.get("required", [])
                                logger.info(
                                    f"Required parameters for {function_name}: {required_params}"
                                )

                        # Check if required parameters are missing
                        if required_params and not function_args:
                            logger.warning(
                                f"Required parameters missing for {function_name}: {required_params}"
                            )
                            # Add tool message with error
                            tool_message = ToolMessage(
                                content=f"The required params are missing. Here required tool params {required_params}",
                                name=function_name,
                                tool_call_id=tool_call_id,
                                status="error",
                                additional_kwargs={"input": {}},
                            )
                            working_messages.append(tool_message)
                            # Continue to next iteration to let model respond with the error
                            continue

                        organization_id = None
                        if settings:
                            organization_id = settings.get("function", {}).get(
                                "organization_id"
                            )

                        logger.info(
                            f"Using organization_id: {organization_id} for tool call"
                        )

                        if not organization_id:
                            is_error = True

                        method = "POST"
                        is_direct_response = False
                        if settings:
                            method = settings.get("function", {}).get("method", "POST")
                            is_direct_response = settings.get("function", {}).get(
                                "is_direct_response", False
                            )

                        # Handle specific tool calls
                        result = None

                        if function_name == "reset_memory":
                            result = await self._execute_reset_memory(
                                function_args, chat_history
                            )
                            # Add tool message to conversation
                            tool_message = ToolMessage(
                                content=(
                                    safe_json_dumps(result)
                                    if not isinstance(result, str)
                                    else result
                                ),
                                name="reset_memory",
                                tool_call_id=tool_call_id,
                                status="success",
                                additional_kwargs={"input": function_args},
                            )
                            working_messages.append(tool_message)
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=result,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Check if tool call limit has been reached for this tool
                        current_tool_count = tool_call_success_counts.get(
                            function_name, 0
                        )
                        if current_tool_count >= MAX_TOOL_CALLS:
                            logger.warning(
                                f"Tool call limit reached for {function_name}: {current_tool_count}/{MAX_TOOL_CALLS}"
                            )
                            # Add tool message with error
                            tool_message = ToolMessage(
                                content=f"Tool call limit reached for '{function_name}'. This tool has been called {current_tool_count} times, which exceeds the maximum of {MAX_TOOL_CALLS}. Please don't re-invoke it.",
                                name=function_name,
                                tool_call_id=tool_call_id,
                                status="success",
                                additional_kwargs={"input": function_args},
                            )
                            working_messages.append(tool_message)
                            # Continue to next tool call or iteration
                            continue

                        # Handle custom tools (including MCP server tools)
                        tool_names = [
                            tool.get("function", {}).get("name", "") for tool in tools
                        ]
                        if function_name in tool_names:
                            # Try executing via MCP server if server_context is provided
                            try:
                                matched_tool = next(
                                    (
                                        t
                                        for t in tools
                                        if t.get("function", {}).get("name")
                                        == function_name
                                    ),
                                    None,
                                )
                                server_ctx = (
                                    (matched_tool or {}).get("server_context")
                                    if matched_tool
                                    else None
                                )
                                if server_ctx:
                                    logger.debug(
                                        f"[DEBUG-MCP] [MCP] non_streaming (alt path) call name={function_name} url={server_ctx.get('url','')} token_present={bool(server_ctx.get('token'))}"
                                    )
                                    result = await execute_tool_server(
                                        token=server_ctx.get("token", ""),
                                        url=server_ctx.get("url", ""),
                                        name=function_name,
                                        params=function_args,
                                        server_data=server_ctx.get("server_data", {}),
                                    )

                                    tool_message = ToolMessage(
                                        content=(
                                            safe_json_dumps(result)
                                            if not isinstance(result, str)
                                            else result
                                        ),
                                        name=function_name,
                                        tool_call_id=tool_call_id,
                                        status="success",
                                        additional_kwargs={"input": function_args},
                                    )
                                    working_messages.append(tool_message)

                                    final_response = (
                                        safe_json_dumps(result)
                                        if not isinstance(result, str)
                                        else result
                                    )
                                    # Respect direct-response; otherwise let model continue to respond naturally
                                    if (
                                        is_direct_response
                                        and len(current_response.tool_calls) == 1
                                    ):
                                        yield AnswerChunk(
                                            thread_id=thread_id,
                                            message=final_response,
                                            is_partial=False,
                                            message_id=message_id,
                                            sources=sources,
                                        )
                                        return

                                    # Tool executed successfully, continue conversation loop
                                    continue
                            except Exception as _:
                                pass

                            # Fallback: original behavior (UI-driven custom tools)
                            # For custom tools, we need to let the frontend handle them
                            # Return current content and let the UI process the tool call
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=current_content or "",
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Handle workflow triggers
                        if is_error:
                            result = f"missing required data {function_args} to execute the function {function_name}"
                        else:
                            result = await self._async_trigger_workflow(
                                organization_id=organization_id,
                                workflow_id=str(workflow_id),
                                method=method,
                                data=function_args,
                            )

                        logger.info(f"Tool call result: {result}")

                        # Add tool message to conversation
                        tool_message = ToolMessage(
                            content=(
                                safe_json_dumps(result)
                                if not isinstance(result, str)
                                else result
                            ),
                            name=function_name,
                            tool_call_id=tool_call_id,
                            status="success",
                            additional_kwargs={"input": function_args},
                        )
                        working_messages.append(tool_message)
                        tool_call_success_counts[function_name] = (
                            tool_call_success_counts.get(function_name, 0) + 1
                        )
                        logger.info(
                            f"Tool '{function_name}' successful call count: {tool_call_success_counts[function_name]}/{MAX_TOOL_CALLS}"
                        )

                        # Check if response is too long
                        final_response = (
                            safe_json_dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                        if len(final_response) > MAX_RESPONSE_TOKENS:
                            # In a real implementation, you'd upload to file storage here
                            # For now, we'll truncate and add a notice
                            truncated_response = final_response[:MAX_RESPONSE_TOKENS]
                            file_notice = f"\n\n[Note: Response truncated due to length. Full response was {len(final_response)} characters.]"
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=truncated_response + file_notice,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Check if this is a direct response
                        if is_direct_response and len(current_response.tool_calls) == 1:
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=final_response,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                    # After processing all tool calls, continue the loop to get the next response
                    continue

                # Continue the loop if we haven't returned yet
                # The loop will continue until the AI naturally completes or stops

            # Safety net: if we somehow exit the loop without returning
            logger.warning(
                f"Safety limit reached ({SAFETY_MAX_ITERATIONS} iterations) - returning last response"
            )
            # Try to return the last response content if available
            last_ai_message = None
            for msg in reversed(working_messages):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break

            # yield AnswerChunk(
            #     thread_id=thread_id,
            #     message=(
            #         last_ai_message.content
            #         if last_ai_message
            #         else "Processing completed"
            #     ),
            #     is_partial=False,
            #     message_id=message_id,
            #     sources=sources,
            # )

        except Exception as error:
            logger.error(f"Streaming error: {error}")
            yield AnswerChunk(
                thread_id=thread_id,
                message=f"Error: {str(error)}",
                is_partial=False,
                message_id=message_id,
                sources=sources,
            )

    async def _process_streaming_response_v4(
        self,
        inputs: AnswerQuestionInputs,
        prompt_json: List[Dict],
        thread_id: str,
        message_id: str,
        model_with_tools,
        workflow_settings: List[Dict],
        tools: List[Dict],
        sources: List[Dict],
        chat_history: MongoDBChatMessageHistory,
        use_memory: bool = True,
    ) -> AsyncGenerator[AnswerChunk, None]:
        """Process streaming response from LLM with workflow trigger integration"""
        try:

            # Create working copy of prompt messages
            working_messages = prompt_json
            tool_call_success_counts = {}

            # Safety counter to prevent infinite loops
            safety_counter = 0
            logger.info(
                f"Starting streaming response processing for thread {thread_id} with message ID {message_id}"
            )
            # Continue processing until the AI naturally completes or stops
            while safety_counter < SAFETY_MAX_ITERATIONS:
                safety_counter += 1

                # Time the model invocation
                start_time = time.time()

                # Stream the response from the model
                # Use LangChain's chunk accumulation pattern
                gathered = None
                current_content = ""
                content_chunks = (
                    []
                )  # Buffer content chunks until we know if there are tool calls

                for chunk in model_with_tools.stream(working_messages):
                    # Accumulate chunks using the + operator (LangChain pattern)
                    logger.debug(f"Received chunk data: {chunk}")
                    gathered = chunk if gathered is None else gathered + chunk

                    # Handle content chunks - buffer them instead of streaming immediately
                    # Content can be either a string or a list of dicts with 'text' field
                    if hasattr(chunk, "content") and chunk.content:
                        chunk_text = ""

                        # Handle different content formats
                        if isinstance(chunk.content, str):
                            chunk_text = chunk.content
                        elif isinstance(chunk.content, list):
                            # Extract text from list of content blocks
                            for content_block in chunk.content:
                                if isinstance(content_block, dict):
                                    # Check for 'text' field in content block
                                    if (
                                        content_block.get("type") == "text"
                                        and "text" in content_block
                                    ):
                                        chunk_text += content_block["text"]
                                    # Skip tool_use blocks as they're not text content
                                elif isinstance(content_block, str):
                                    chunk_text += content_block

                        # Only add non-empty text chunks
                        if chunk_text:
                            current_content += chunk_text
                            content_chunks.append(chunk_text)
                            logger.info(f"Buffered chunk content: {chunk_text}")

                # After streaming, use the accumulated response
                current_response = gathered

                # Check if response has tool calls - if not, stream the buffered content
                has_tool_calls_in_response = (
                    current_response
                    and hasattr(current_response, "tool_calls")
                    and current_response.tool_calls
                    and len(current_response.tool_calls) > 0
                )

                # Only stream content if there are NO tool calls
                # If there are tool calls, skip streaming and let the tool execution flow handle it
                if not has_tool_calls_in_response and content_chunks:
                    for chunk_content in content_chunks:
                        yield AnswerChunk(
                            thread_id=thread_id,
                            message=chunk_content,
                            is_partial=True,
                            message_id=message_id,
                            sources=(
                                sources
                                if content_chunks.index(chunk_content) == 0
                                else []
                            ),
                        )
                logger.debug(f"Accumulated response: {current_response}")
                if current_response and hasattr(current_response, "tool_calls"):
                    logger.info(
                        f"Accumulated tool calls: {len(current_response.tool_calls) if current_response.tool_calls else 0}"
                    )
                    if current_response.tool_calls:
                        for i, tc in enumerate(current_response.tool_calls):
                            logger.info(
                                f"Accumulated tool call {i}: name={getattr(tc, 'name', None)}, id={getattr(tc, 'id', None)}"
                            )

                end_time = time.time()
                logger.info(
                    f"Streaming eval duration: {(end_time - start_time) * 1000:.2f}ms"
                )

                if not current_response:
                    logger.warning("No response received from model during streaming")
                    yield AnswerChunk(
                        thread_id=thread_id,
                        message="",
                        is_partial=True,
                        message_id=message_id,
                        sources=sources,
                    )
                    return

                # Check if we should stop (no tool calls or stop reason)
                should_stop = False

                # First, check if there are tool calls to process
                has_tool_calls = (
                    hasattr(current_response, "tool_calls")
                    and current_response.tool_calls
                    and len(current_response.tool_calls) > 0
                )

                logger.debug(f"Tool call detection - has_tool_calls: {has_tool_calls}")
                if has_tool_calls:
                    logger.info(
                        f"Number of tool calls: {len(current_response.tool_calls)}"
                    )
                    for i, tc in enumerate(current_response.tool_calls):
                        # Enhanced debugging to see the actual structure
                        logger.info(f"Tool call {i} raw object: {tc}")
                        logger.info(f"Tool call {i} type: {type(tc)}")
                        if hasattr(tc, "__dict__"):
                            logger.info(f"Tool call {i} attributes: {tc.__dict__}")

                        # Try different ways to get the name
                        name = None
                        if hasattr(tc, "name"):
                            name = tc.name
                        elif hasattr(tc, "function") and hasattr(tc.function, "name"):
                            name = tc.function.name
                        elif isinstance(tc, dict):
                            name = tc.get("name") or tc.get("function", {}).get("name")

                        logger.info(f"Tool call {i} extracted name: {name}")

                # If no tool calls, check for stop conditions
                if not has_tool_calls:
                    # Check various finish reasons
                    finish_reason = None
                    if (
                        hasattr(current_response, "response_metadata")
                        and current_response.response_metadata
                    ):
                        finish_reason = current_response.response_metadata.get(
                            "finish_reason"
                        ) or current_response.response_metadata.get("done_reason")

                    if (
                        hasattr(current_response, "additional_kwargs")
                        and current_response.additional_kwargs
                    ):
                        finish_reason = (
                            finish_reason
                            or current_response.additional_kwargs.get(
                                "finishReason", ""
                            ).lower()
                        )

                    # Stop if we have a stop reason, or if there's simply no more tool calls and we have content
                    if finish_reason == "stop" or (
                        hasattr(current_response, "content")
                        and current_content
                        and current_content.strip()
                    ):
                        should_stop = True

                if should_stop:
                    # Add final AI message and save to history
                    if hasattr(current_response, "content") and current_content:
                        working_messages.append(AIMessage(content=current_content))
                    final_prompt = custom_trim_messages(working_messages, max_tokens=50)
                    new_history = validate_tool_calls(final_prompt)
                    # Save to chat history
                    if chat_history and use_memory:
                        chat_history.clear()
                        for msg in new_history:
                            if isinstance(msg, dict):
                                if msg.get("role") == "user":
                                    chat_history.add_user_message(
                                        msg.get("content", "")
                                    )
                                elif msg.get("role") == "assistant":
                                    chat_history.add_ai_message(msg.get("content", ""))
                            elif isinstance(msg, HumanMessage):
                                chat_history.add_user_message(msg.content)
                            elif isinstance(msg, AIMessage):
                                chat_history.add_ai_message(msg.content)
                    if (
                        inputs.message_id
                        and inputs.conversation_id
                        and inputs.message_created_at
                        and current_content.startswith("bi_")
                    ):
                        board_message_model.create(
                            organization_id=inputs.organization_id,
                            details=BoardMessageForm(
                                conversation_id=inputs.conversation_id,
                                message_id=inputs.message_id,
                                board_item_id=inputs.board_item_id,
                                created_at=inputs.message_created_at,
                            ),
                        )

                    # yield AnswerChunk(
                    #     thread_id=thread_id,
                    #     message=current_content or "",
                    #     is_partial=False,
                    #     message_id=message_id,
                    #     sources=sources,
                    # )
                    return

                # Process tool calls
                if has_tool_calls:
                    # Add the AI message with tool calls
                    working_messages.append(
                        AIMessage(
                            content=current_content or "",
                            tool_calls=current_response.tool_calls,
                        )
                    )

                    for tool_call in current_response.tool_calls:
                        is_error = False
                        function_name = ""
                        function_args = {}
                        tool_call_id = ""

                        # Enhanced tool call parsing to handle various formats
                        logger.info(f"Processing tool call raw: {tool_call}")
                        logger.info(f"Tool call type: {type(tool_call)}")

                        # Method 1: Direct attributes (LangChain format or our reconstructed format)
                        if hasattr(tool_call, "name") and tool_call.name:
                            function_name = tool_call.name
                            function_args = (
                                tool_call.args
                                if hasattr(tool_call, "args") and tool_call.args
                                else {}
                            )
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                            logger.info(
                                f"Method 1 - name: {function_name}, args: {function_args}"
                            )

                        # Method 2: Nested function attribute (OpenAI format)
                        elif hasattr(tool_call, "function") and tool_call.function:
                            func = tool_call.function
                            function_name = getattr(func, "name", "")
                            # Handle args that might be JSON string or dict
                            args = getattr(func, "arguments", {})
                            if isinstance(args, str):
                                try:
                                    function_args = (
                                        json.loads(args) if args.strip() else {}
                                    )
                                except json.JSONDecodeError:
                                    logger.warning(
                                        f"Failed to parse JSON arguments: {args}"
                                    )
                                    function_args = {}
                            else:
                                function_args = args
                            tool_call_id = (
                                tool_call.id if hasattr(tool_call, "id") else ""
                            )
                            logger.info(
                                f"Method 2 - name: {function_name}, args: {function_args}"
                            )

                        # Method 3: Dictionary format
                        elif isinstance(tool_call, dict):
                            function_name = tool_call.get("name", "")
                            if not function_name and "function" in tool_call:
                                func = tool_call["function"]
                                function_name = func.get("name", "")
                                args = func.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        function_args = (
                                            json.loads(args) if args.strip() else {}
                                        )
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to parse JSON arguments: {args}"
                                        )
                                        function_args = {}
                                else:
                                    function_args = args if args else {}
                            else:
                                # Try multiple possible argument field names
                                function_args = None
                                logger.debug(
                                    f"Checking for args in tool_call keys: {list(tool_call.keys())}"
                                )

                                for arg_field in [
                                    "args",
                                    "arguments",
                                    "parameters",
                                    "input",
                                    "params",
                                ]:
                                    if arg_field in tool_call:
                                        potential_args = tool_call[arg_field]
                                        logger.debug(
                                            f"Found {arg_field}: {potential_args} (type: {type(potential_args)})"
                                        )

                                        if isinstance(potential_args, str):
                                            try:
                                                function_args = (
                                                    json.loads(potential_args)
                                                    if potential_args.strip()
                                                    else {}
                                                )
                                                logger.debug(
                                                    f"Successfully parsed JSON from {arg_field}"
                                                )
                                                break
                                            except json.JSONDecodeError:
                                                logger.debug(
                                                    f"Failed to parse JSON from {arg_field}"
                                                )
                                                continue
                                        elif isinstance(potential_args, dict):
                                            function_args = potential_args
                                            logger.debug(f"Using dict from {arg_field}")
                                            break

                                # If no args found in any field, default to empty dict
                                if function_args is None:
                                    function_args = {}
                                    logger.warning(
                                        f"No arguments found in any field. Available fields: {list(tool_call.keys())}"
                                    )

                            tool_call_id = tool_call.get("id", "")
                            logger.debug(f"Method 3 - final args: {function_args}")

                        # Method 4: Check for type field indicating different structure or incomplete tool call
                        elif (
                            hasattr(tool_call, "type") and tool_call.type == "tool_call"
                        ):
                            # This might be a partially formed tool call, skip it
                            logger.warning(
                                f"Skipping incomplete tool call: {tool_call}"
                            )
                            continue
                        else:
                            # Unknown format, try best effort extraction
                            logger.warning(
                                f"Unknown tool call format: {tool_call}, attempting best effort extraction"
                            )
                            function_name = ""
                            function_args = {}
                            tool_call_id = ""

                            # Try to extract name from common attributes
                            if hasattr(tool_call, "__dict__"):
                                attrs = tool_call.__dict__
                                logger.debug(f"Tool call attributes: {attrs}")
                                function_name = attrs.get("name", "")
                                function_args = attrs.get(
                                    "args", attrs.get("arguments", {})
                                )
                                tool_call_id = attrs.get("id", "")

                            logger.info(
                                f"Best effort extraction - name: {function_name}, args: {function_args}"
                            )

                        logger.info(
                            f"Parsed - name: '{function_name}', args: {function_args}, id: '{tool_call_id}'"
                        )

                        # Skip if we could not extract a function name
                        if not function_name or function_name.strip() == "":
                            logger.warning(
                                f"Skipping tool call with empty name: {tool_call}"
                            )
                            continue
                        # Extract workflow ID from function name
                        workflow_id = None
                        if "_" in function_name:
                            try:
                                workflow_id = function_name.split("_")[0]
                            except ValueError:
                                pass
                        logger.info(
                            f"Extracted workflow_id: {workflow_id} from function_name: {function_name}"
                        )
                        if not workflow_id:
                            is_error = True

                        # Find workflow settings
                        settings = None
                        if workflow_settings:
                            settings = next(
                                (
                                    ws
                                    for ws in workflow_settings
                                    if ws.get("function", {}).get("name")
                                    == function_name
                                ),
                                None,
                            )

                        organization_id = None
                        if settings:
                            organization_id = settings.get("function", {}).get(
                                "organization_id"
                            )

                        if not organization_id:
                            is_error = True

                        method = "POST"
                        is_direct_response = False
                        if settings:
                            method = settings.get("function", {}).get("method", "POST")
                            is_direct_response = settings.get("function", {}).get(
                                "is_direct_response", False
                            )

                        # Handle specific tool calls
                        result = None

                        if function_name == "reset_memory":
                            result = await self._execute_reset_memory(
                                function_args, chat_history
                            )
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=result,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        current_tool_count = tool_call_success_counts.get(
                            function_name, 0
                        )
                        if current_tool_count >= MAX_TOOL_CALLS:
                            logger.warning(
                                f"Tool call limit reached for {function_name}: {current_tool_count}/{MAX_TOOL_CALLS}"
                            )
                            # Add tool message with error and return immediately
                            tool_message = ToolMessage(
                                content=f"Tool call limit reached for '{function_name}'. This tool has been called {current_tool_count} times, which exceeds the maximum of {MAX_TOOL_CALLS}. Please don't reinvoke it.",
                                name=function_name,
                                tool_call_id=tool_call_id,
                                status="success",
                                additional_kwargs={"input": function_args},
                            )
                            working_messages.append(tool_message)
                            # Continue to next tool call or iteration
                            continue

                        # Handle custom tools (including MCP server tools)
                        tool_names = [
                            tool.get("function", {}).get("name", "") for tool in tools
                        ]
                        if function_name in tool_names:
                            # Try executing via MCP server if server_context is provided
                            try:
                                matched_tool = next(
                                    (
                                        t
                                        for t in tools
                                        if t.get("function", {}).get("name")
                                        == function_name
                                    ),
                                    None,
                                )
                                server_ctx = (
                                    (matched_tool or {}).get("server_context")
                                    if matched_tool
                                    else None
                                )
                                if server_ctx:
                                    logger.debug(
                                        f"[DEBUG-MCP] [MCP] non_streaming (alt path) call name={function_name} url={server_ctx.get('url','')} token_present={bool(server_ctx.get('token'))}"
                                    )
                                    result = await execute_tool_server(
                                        token=server_ctx.get("token", ""),
                                        url=server_ctx.get("url", ""),
                                        name=function_name,
                                        params=function_args,
                                        server_data=server_ctx.get("server_data", {}),
                                    )

                                    tool_message = ToolMessage(
                                        content=(
                                            safe_json_dumps(result)
                                            if not isinstance(result, str)
                                            else result
                                        ),
                                        name=function_name,
                                        tool_call_id=tool_call_id,
                                        additional_kwargs={"input": function_args},
                                    )
                                    working_messages.append(tool_message)

                                    final_response = (
                                        safe_json_dumps(result)
                                        if not isinstance(result, str)
                                        else result
                                    )
                                    # Respect direct-response; otherwise let model continue to respond naturally
                                    if (
                                        is_direct_response
                                        and len(current_response.tool_calls) == 1
                                    ):
                                        yield AnswerChunk(
                                            thread_id=thread_id,
                                            message=final_response,
                                            is_partial=False,
                                            message_id=message_id,
                                            sources=sources,
                                        )
                                        return

                                    # Tool executed successfully, continue conversation loop
                                    continue
                            except Exception as _:
                                pass

                            # Fallback: original behavior (UI-driven custom tools)
                            # For custom tools, we need to let the frontend handle them
                            # Return current content and let the UI process the tool call
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=current_content or "",
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Handle workflow triggers
                        if is_error:
                            result = f"missing required data {function_args} to execute the function {function_name}"
                            break
                        else:
                            result = await self._async_trigger_workflow(
                                organization_id=organization_id,
                                workflow_id=str(workflow_id),
                                method=method,
                                data=function_args,
                            )

                        logger.info(f"Tool call result: {result}")

                        # Add tool message to conversation
                        tool_message = ToolMessage(
                            content=(
                                safe_json_dumps(result)
                                if not isinstance(result, str)
                                else result
                            ),
                            name=function_name,
                            status="error" if is_error else "success",
                            tool_call_id=tool_call_id,
                        )
                        tool_call_success_counts[function_name] = (
                            tool_call_success_counts.get(function_name, 0) + 1
                        )
                        logger.info(
                            f"Tool '{function_name}' successful call count: {tool_call_success_counts[function_name]}/{MAX_TOOL_CALLS}"
                        )
                        working_messages.append(tool_message)

                        # Check if response is too long
                        final_response = (
                            safe_json_dumps(result)
                            if not isinstance(result, str)
                            else result
                        )
                        if len(final_response) > MAX_RESPONSE_TOKENS:
                            # In a real implementation, you'd upload to file storage here
                            # For now, we'll truncate and add a notice
                            truncated_response = final_response[:MAX_RESPONSE_TOKENS]
                            file_notice = f"\n\n[Note: Response truncated due to length. Full response was {len(final_response)} characters.]"
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=truncated_response + file_notice,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                        # Check if this is a direct response
                        if is_direct_response and len(current_response.tool_calls) == 1:
                            yield AnswerChunk(
                                thread_id=thread_id,
                                message=final_response,
                                is_partial=False,
                                message_id=message_id,
                                sources=sources,
                            )
                            return

                    # After processing all tool calls, continue the loop to get the next response
                    continue

                # Continue the loop if we haven't returned yet
                # The loop will continue until the AI naturally completes or stops

            # Safety net: if we somehow exit the loop without returning
            logger.warning(
                f"Safety limit reached ({SAFETY_MAX_ITERATIONS} iterations) - returning last response"
            )
            # Try to return the last response content if available
            last_ai_message = None
            for msg in reversed(working_messages):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg
                    break

            # yield AnswerChunk(
            #     thread_id=thread_id,
            #     message=(
            #         last_ai_message.content
            #         if last_ai_message
            #         else "Processing completed"
            #     ),
            #     is_partial=False,
            #     message_id=message_id,
            #     sources=sources,
            # )

        except Exception as error:
            logger.error(f"Streaming error: {error}")
            yield AnswerChunk(
                thread_id=thread_id,
                message=f"There was an error processing your request with llm provider. Please try again later.",
                is_partial=False,
                message_id=message_id,
                sources=sources,
            )

    async def build_sub_agents(
        self, assistant_ids: List[str], all_models: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """
        Build agents for a list of assistant IDs using LangGraph with proper LLM and tools setup
        Python equivalent of buildAgents, following agent.py answer_question pattern

        Args:
            assistant_ids: List of assistant IDs
            organization_id: Organization ID for context

        Returns:
            List of agent configurations with LLM and tools
        """
        try:
            agents = []

            if not assistant_ids:
                return agents

            # Get sub-agent details
            sub_agents = await get_assistants(assistant_ids)

            logger.info(
                f"Building {len(sub_agents)} sub-agents for assistants: {assistant_ids}"
            )

            for assistant in sub_agents:
                # Build individual agent using the build_agent function
                organization_id = assistant.get("organization_id", "")
                agent_config = await self.build_agent(
                    assistant, organization_id, True, all_models
                )

                if agent_config:
                    logger.info(
                        f"Built agent for assistant {assistant.get('assistant_id', 'unknown')}"
                    )
                    agents.append(agent_config)
                else:
                    logger.warning(
                        f"Failed to build agent for assistant {assistant.get('assistant_id', 'unknown')}"
                    )

            logger.info(
                f"Successfully built {len(agents)} sub-agents out of {len(sub_agents)} requested"
            )
            return agents

        except Exception as e:
            logger.error(f"Error building sub-agents: {e}")
            return []

    async def build_agent(
        self,
        assistant: Dict[str, Any],
        organization_id: str = "",
        streaming: bool = False,
        llm_models: Dict[str, List[Dict]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Build a LangGraph agent for a single assistant with complete LLM and tools setup

        Args:
            assistant: Assistant configuration dictionary
            organization_id: Organization ID for context

        Returns:
            Agent configuration dictionary or None if failed
        """
        try:

            # Get model configuration similar to answer_question
            logger.info(
                f"Building agent for assistant: {assistant.get('name', 'unknown')}"
            )
            model_id = assistant.get("model_id", _get_default_model())
            if model_id == "Default":
                model_id = _get_default_model()

            logger.info(f"Using model ID: {model_id} for sub agent ")

            assistant_name = self._normalize_agent_name(
                assistant.get("name", f"Agent-{assistant.get('assistant_id', '')}")
            )
            temperature = assistant.get("temperature", 0.1)

            version = assistant.get("version", 1)

            detected_provider = detect_provider(
                model_id=model_id, models=llm_models, default_provider=PROVIDER
            )
            # Create LLM provider instance (following agent.py pattern)
            provider_type = detected_provider
            llm_provider = get_llm_provider(detected_provider, CONFIG)

            # Create the main LLM for this agent
            llm = llm_provider.create_llm(
                model_id, llm_models, streaming=True, temperature=temperature
            )

            # Get workflow settings if the assistant has workflow function calls
            workflow_function_call = assistant.get("workflow_function_call", [])
            workflow_settings = await self._get_workflow_settings(
                workflow_function_call, organization_id, version
            )

            # Get assistant-specific tools
            assistant_tools = assistant.get("tools", [])

            # Create a comprehensive tool set for the agent
            tools_list = []

            # Add workflow tools
            if workflow_settings:
                tools_list.extend(workflow_settings)

            # Add assistant-specific tools
            if assistant_tools:
                tools_list.extend(assistant_tools)

            # Build assistant prompt
            instructions = assistant.get("instructions", "")
            assistant_prompt = build_assistant_prompt(RAG_SYSTEM_PROMPT, instructions)
            logger.info(f"agent tool list : {len(tools_list)}")
            # Create the agent using LangGraph's create_react_agent

            llm_with_tools = llm.bind_tools(
                tools=tools_list,
                parallel_tool_calls=False,
            )

            if (
                assistant.get("file_ids")
                or assistant.get("knowledge_hubs")
                or assistant.get("board_id")
            ):
                logger.info("Building agent with hook")
                agent = create_react_agent(
                    model=llm_with_tools,
                    tools=tools_list,
                    prompt=assistant_prompt,
                    name=assistant_name,
                    pre_model_hook=self.agent_pre_hook,
                )
            else:
                agent = create_react_agent(
                    model=llm_with_tools,
                    tools=tools_list,
                    prompt=instructions,
                    name=assistant_name,
                )

            handoff = create_custom_handoff_tool(
                agent_name=f"{assistant_name}",
                description=f"Assign task to a {assistant_name} or help.",
                add_handoff_messages=True,
            )

            # Create agent configuration dictionary
            agent_config = {
                "agent": agent,
                "assistant_id": assistant.get("assistant_id"),
                "file_ids": assistant.get("file_ids", []),
                "knowledge_hubs": assistant.get("knowledge_hubs", []),
                "board_id": assistant.get("board_id", ""),
                "name": assistant_name,
                "model_id": model_id,
                "provider": provider_type,
                "instructions": instructions,
                "organization_id": organization_id,
                "llm_provider": llm_provider,  # Keep reference for potential future use
                "handoff": handoff,
                "workflow_settings": workflow_settings,
            }

            logger.info(
                f"Successfully built agent: {assistant.get('name')} "
                f"model: {model_id}, agent_type: {type(agent).__name__}"
            )

            return agent_config

        except Exception as e:
            logger.error(
                f"Error building agent {assistant.get('assistant_id', 'unknown')}: {e}"
            )
            return None

    def build_supervisor(
        self, name, llm, combined_workflow_settings, tools, chat_history, thread_id
    ):
        """
        Build a supervisor agent for managing multiple assistants
        This is a placeholder function that can be extended to create a supervisor agent
        """

        # Supervisor logic can be implemented here
        async def supervisor_node(state):
            messages = state["messages"]

            # Get the last message
            last_message = messages[-1]
            logger.info(f"Supervisor processing last message: {last_message}")

            agent_name = ""

            # First, check if the last message indicates completion/stop
            should_stop_from_last_message = False

            # Check if last message has stop indication in response metadata
            if (
                hasattr(last_message, "response_metadata")
                and last_message.response_metadata
            ):
                finish_reason = last_message.response_metadata.get("finish_reason")
                if finish_reason == "stop":
                    should_stop_from_last_message = True
                    logger.info(
                        "Last message indicated stop via response_metadata finish_reason"
                    )

            # Check if last message has a name attribute (from specific agent) and stop reason
            if (
                hasattr(last_message, "name")
                and last_message.name
                and hasattr(last_message, "response_metadata")
                and last_message.response_metadata.get("finish_reason") == "stop"
            ):
                should_stop_from_last_message = True
                logger.info(
                    f"Agent '{last_message.name}' completed task with stop reason"
                )

            # If last message indicates stop, end the conversation
            if should_stop_from_last_message:
                logger.info(
                    "Supervisor detected completion from last message - ending conversation"
                )
                return Command(goto=END)

            # Check if the LLM wants to stop based on response metadata
            should_stop = False
            # If it has tool calls, execute them and add results
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                logger.info(f"Last message has tool calls: {last_message.tool_calls}")
                for tool_call in last_message.tool_calls:
                    # Execute your tool
                    logger.info(f"Processing tool call: {tool_call}")
                    tool_id = tool_call.get("id", "")
                    tool_name = tool_call.get("name", "")
                    result = f"transfer to {tool_name} tool successfully"
                    logger.info(f"supervisor Tool call result: {result}")
                    if tool_name.startswith("transfer_to"):
                        # Handle transfer_to tool calls
                        agent_name = tool_name.replace("transfer_to_", "")
                    if (
                        not tool_name.startswith("transfer_to")
                        and not tool_name.startswith("retrieve_context")
                        and not tool_name.startswith("aggregate_context")
                    ):
                        logger.info(f"Processing tool call: {tool_call}")
                        result = await self._process_tool_call(
                            tool_call,
                            combined_workflow_settings,
                            tools,
                            chat_history,
                            thread_id,
                        )

                    if tool_name == "reset_memory":
                        should_stop = True
                        logger.info(
                            "Supervisor detected reset_memory tool call - ending conversation"
                        )

                    # Create and append ToolMessage
                    logger.info(f"Adding tool message: {result}")
                    tool_msg = ToolMessage(content=result, tool_call_id=tool_call["id"])

                    messages.append(tool_msg)
            # logger.info(f"Triggering supervisor node {messages}")

            # Continue with your supervisor logic
            response = await llm.ainvoke(messages)
            logger.info(f"Supervisor response: {response}")
            content = response.content if hasattr(response, "content") else ""
            logger.info(f"Supervisor response content: {content}")

            # Check various finish/stop reasons
            if hasattr(response, "response_metadata") and response.response_metadata:
                finish_reason = response.response_metadata.get("finish_reason")
                if finish_reason == "stop":
                    should_stop = True
                    logger.info("LLM indicated stop via finish_reason")

            # Check additional kwargs for stop conditions
            if hasattr(response, "additional_kwargs") and response.additional_kwargs:
                finish_reason = response.additional_kwargs.get(
                    "finishReason", ""
                ).lower()
                if finish_reason == "stop":
                    should_stop = True
                    logger.info("LLM indicated stop via additional_kwargs")

            # Check if response has content but no tool calls (natural completion)
            if (
                content
                and content.strip()
                and not (hasattr(response, "tool_calls") and response.tool_calls)
            ):
                should_stop = True
                logger.info("LLM provided final response without tool calls")

            # If LLM wants to stop, end the conversation
            if should_stop:
                logger.info("Supervisor detected stop condition - ending conversation")
                return Command(goto=END)

            # Extract tool name from the response
            tool_name = ""
            agent_name = ""
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Get the first tool call
                first_tool_call = response.tool_calls[0]
                if hasattr(first_tool_call, "name"):
                    tool_name = first_tool_call.name
                elif isinstance(first_tool_call, dict):
                    tool_name = first_tool_call.get("name", "")
                logger.info(f"Extracted tool name from response: {tool_name}")

                # If it's a transfer tool, extract the agent name
                if tool_name.startswith("transfer_to_"):
                    agent_name = tool_name.replace("transfer_to_", "")
                elif tool_name == "reset_memory":
                    # function_args = first_tool_call.get("args", {})
                    # reset_memory_msg = await self._execute_reset_memory(
                    #     function_args, chat_history
                    # )
                    # logger.info(f"Reset memory result: {reset_memory_msg}")
                    # reset_message = AIMessage(
                    #     content=reset_memory_msg,
                    #     tool_calls=[{
                    #         "name": "reset_memory",
                    #         "args": function_args,
                    #         "id": first_tool_call.get("id", "")
                    #     }]
                    # )

                    # tool_msg = ToolMessage(
                    #     content=reset_memory_msg,
                    #     tool_call_id=first_tool_call.get("id", "")
                    # )

                    # messages.append(reset_message)
                    # messages.append(tool_msg)

                    return Command(goto=name)

                else:
                    # Non-transfer tool calls might indicate task completion
                    logger.info(f"Non-transfer tool call detected: {tool_name}")

                    # You might want to handle non-transfer tools differently here
                    # For now, we'll treat them as completion signals
                    agent_name = ""

            logger.info(f"Supervisor agent name: {agent_name}")
            if not agent_name:
                # If no agent name is specified, end the conversation
                logger.info("No agent specified - ending conversation")
                return Command(goto=END)
            return Command(goto=agent_name)

        return supervisor_node

    def _normalize_agent_name(self, agent_name: str) -> str:
        """
        Normalize agent name from format like "[Assistant] Math Expert" to "assistant_math_expert"

        Args:
            agent_name: Original agent name that may contain brackets, spaces, and special chars

        Returns:
            Normalized agent name in lowercase with underscores
        """
        # Remove content within brackets (including the brackets themselves)
        name_without_brackets = re.sub(r"\[.*?\]", "", agent_name)

        # Remove leading/trailing whitespace
        name_clean = name_without_brackets.strip()

        # Convert to lowercase
        name_lower = name_clean.lower()

        # Replace spaces and special characters with underscores
        name_normalized = re.sub(r"[^\w]+", "_", name_lower)

        # Remove leading/trailing underscores
        name_normalized = name_normalized.strip("_")

        # Ensure we have a valid name (fallback if empty)
        if not name_normalized:
            name_normalized = "agent"

        return name_normalized

    async def agent_pre_hook(self, state: Any, config) -> Any:
        """
        Pre-hook for agent to set up any necessary state or context
        This can be used to initialize conversation history, context, etc.
        """
        # logger.info(f"Running agent pre-hook for state: {state}")
        # logger.info(f"Agent pre-hook config: {config}")
        metadata = config.get("metadata", {})
        # logger.info(f"Agent pre-hook metadata: {metadata}")
        assistant_id = metadata.get("assistant_id", "")
        organization_id = metadata.get("organization_id", "")

        # Extract question from HumanMessage in state
        question = ""
        if isinstance(state, dict) and "messages" in state:
            messages = state["messages"]
            if isinstance(messages, list) and len(messages) > 0:
                # Find the last HumanMessage and extract its content
                for message in reversed(messages):
                    if hasattr(message, "__class__") and "HumanMessage" in str(
                        message.__class__
                    ):
                        question = (
                            message.content if hasattr(message, "content") else ""
                        )
                        break

        logger.info(f"Extracted question from state: {question}")

        # Only proceed with RAG retrieval if there are file_ids or knowledge_hubs
        file_ids = metadata.get("file_ids", [])
        knowledge_hubs = metadata.get("knowledge_hubs", [])
        board_id = metadata.get("board_id", "")
        folder_ids = metadata.get("folder_ids", [])

        if file_ids or knowledge_hubs or board_id or folder_ids:
            try:
                # Create a compatible inputs object for _retrieve_context
                from types import SimpleNamespace

                inputs = SimpleNamespace()
                inputs.question = question
                inputs.assistant_id = assistant_id
                inputs.organization_id = organization_id
                inputs.file_ids = file_ids
                inputs.knowledge_hubs = knowledge_hubs
                inputs.board_id = board_id
                inputs.folder_ids = folder_ids

                model_id = metadata.get("model_id", _get_default_model())
                if not model_id:
                    model_id = _get_default_model()

                temperature = metadata.get("temperature", 0.1)

                logger.info(
                    f"Pre-hook for assistant_id: {assistant_id}, file_ids: {file_ids}, knowledge_hubs: {knowledge_hubs}"
                )

                # Build filters and set up RAG components
                filters = build_filters(
                    {
                        "assistant_id": assistant_id,
                        "file_ids": file_ids,
                        "knowledge_hubs": knowledge_hubs,
                        "board_id": board_id,
                        "folder_ids": folder_ids,
                    },
                    organization_id,
                )

                collection_name = self._determine_collection("")
                if _DB_TYPE == "postgresql":
                    rag_collection = None
                else:
                    db_client = await mongodb_client.get_client()
                    db = db_client[self.db_name]
                    rag_collection = db[collection_name]
                _emb_provider_name = EMBEDDING_PROVIDER if EMBEDDING_PROVIDER else "openai"
                _emb_provider = get_llm_provider(_emb_provider_name, CONFIG)
                embedding_model = _emb_provider.create_embedding_model()
                _ret_provider_name = RETRIEVER_PROVIDER if RETRIEVER_PROVIDER else "openai"
                _ret_provider = get_llm_provider(_ret_provider_name, CONFIG)
                # Pass None so the provider uses its own default retriever model
                retriever_model = _ret_provider.create_retriever_model(None, temperature)
                logger.info(f"agent_pre_hook retriever_model: type={type(retriever_model).__name__}, openai_api_base={getattr(retriever_model, 'openai_api_base', 'NOT_SET')}, model={getattr(retriever_model, 'model_name', 'UNKNOWN')}")
                vector_store = await self.create_vector_store(
                    embedding_model, collection_name
                )

                # Retrieve context
                context_result, sources = await self._retrieve_context(
                    organization_id,
                    inputs,
                    filters,
                    vector_store,
                    retriever_model,
                    rag_collection,
                )

                logger.info(
                    f"Pre-hook retrieved context length: {len(context_result) if context_result else 0}"
                )
                logger.info(
                    f"Pre-hook retrieved sources count: {len(sources) if sources else 0}"
                )

                # Add context and sources to state
                if isinstance(state, dict):
                    content = {"context": context_result, "sources": sources}
                    state["llm_input_messages"] = [
                        AIMessage(content=safe_json_dumps(content))
                    ]
                    logger.info("Added context_result and sources to state")
                else:
                    logger.warning(
                        "State is not a dictionary, cannot add context_result and sources"
                    )

            except Exception as e:
                logger.error(f"Error in agent pre-hook RAG retrieval: {e}")
        else:
            logger.info(
                "No file_ids, knowledge_hubs, folder_ids, or board_id found - skipping RAG retrieval"
            )

        return state

    def should_use_tools(state):
        last_message = state["messages"][-1]
        return hasattr(last_message, "tool_calls") and last_message.tool_calls


def get_all_available_models_from_custom_provider(
    custom_provider: Dict[str, Any],
) -> Dict[str, List[Dict]]:
    """
    Convert custom provider models to the expected dictionary format.

    Custom providers store models as a list, but the system expects a dictionary
    with keys like "bedrockModels", "openaiModels", etc.

    Args:
        custom_provider: Custom provider dictionary from database with structure:
            {
                "type": "bedrock",  # provider type
                "models": [...],    # list of model dictionaries
                ...
            }

    Returns:
        Dictionary containing models in the expected format:
        {
            "bedrockModels": [...],
            # or "openaiModels": [...], etc.
        }
    """
    provider_models = custom_provider.get("models") or []
    provider_type = custom_provider.get("type", "").lower()

    # Map provider type to model key (e.g., "bedrock" -> "bedrockModels")
    provider_key_map = {
        "bedrock": "bedrockModels",
        "openai": "openaiModels",
        "ollama": "ollamaModels",
        "lmstudio": "lmstudioModels",
        "vllm": "vllmModels",
        "google": "googleModels",
    }

    # Convert list of models to expected dictionary format
    if isinstance(provider_models, list):
        model_key = provider_key_map.get(provider_type, f"{provider_type}Models")
        return {model_key: provider_models}
    if isinstance(provider_models, dict):
        # Already in the correct format
        return provider_models

    # Fallback to empty dict if format is unexpected
    logger.warning(
        "Custom provider models is neither list nor dict, got %s",
        type(provider_models),
    )
    return {}


async def get_all_available_models(
    config: Dict[str, Any] = None,
) -> Dict[str, List[Dict]]:
    """
    Get all available models from all providers: OpenAI, Ollama, LMStudio, and Bedrock

    Args:
        config: Optional configuration dictionary. If not provided, uses global CONFIG

    Returns:
        Dictionary containing models from all providers:
        {
            "openaiModels": [...],
            "ollamaModels": [...],
            "lmstudioModels": [...],
            "bedrockModels": [...]
        }
    """
    if config is None:
        config = CONFIG

    all_models = {
        "openaiModels": [],
        "ollamaModels": [],
        "lmstudioModels": [],
        "bedrockModels": [],
        "vllmModels": [],
        "googleModels": [],
    }

    providers = (
        SUPPORTED_PROVIDERS.split(",")
        if isinstance(SUPPORTED_PROVIDERS, str)
        else PROVIDER
    )
    logger.info(f"Fetching models from providers: {providers}")

    for provider_type in providers:
        try:
            logger.info(f"Fetching models from {provider_type} provider...")

            # Create provider instance
            provider = get_llm_provider(provider_type, config)

            # Get available models from this provider
            provider_models = await provider.get_available_models()

            # Merge models into all_models dictionary
            for key, models in provider_models.items():
                if key in all_models:
                    logger.info(f"Merging models into existing key: {key} {models}")
                    all_models[key].extend(models)
                    logger.info(f"Added {len(models)} models from {provider_type}")
                else:
                    all_models[key] = models
                    logger.info(
                        f"Added {len(models)} models from {provider_type} under new key: {key}"
                    )

        except Exception as e:
            logger.error(f"Failed to fetch models from {provider_type}: {str(e)}")
            # Continue with other providers even if one fails
            continue

    # Log summary
    total_models = sum(len(models) for models in all_models.values())
    logger.info(f"Successfully fetched {total_models} total models from all providers")

    return all_models


def get_llm_provider(provider_type: str, config: Dict[str, Any]) -> LLMProvider:
    """
    Factory function to get LLM provider instance
    Similar to get_storage_provider in provider.py
    """
    providers = (
        provider_type.split(",") if isinstance(provider_type, str) else provider_type
    )
    default_provider = providers[0] if providers else "openai"
    logger.info(
        f"Using selected LLM provider: {providers} (default: {default_provider})"
    )

    # Treat "custom" as an alias of the OpenAI provider so that external
    # systems can send `type: "custom"` but still reuse our OpenAI-compatible
    # implementation (including custom base_url, api_key, etc.)
    if default_provider in ("openai", "custom"):
        from .models.openai import OpenAIProvider

        return OpenAIProvider(config)
    elif default_provider == "ollama":
        from .models.ollama import OllamaProvider

        return OllamaProvider(config)
    elif default_provider == "lmstudio":
        from .models.llmstudio import LMStudioProvider

        return LMStudioProvider(config)
    elif default_provider == "google":
        # Google AI (Gemini) provider
        from .models.google import GoogleAIProvider

        return GoogleAIProvider(config)
    elif default_provider == "bedrock":
        from .models.bedrock import BedrockProvider

        return BedrockProvider(config)
    elif default_provider == "vllm":
        from .models.vllm import VLLMProvider

        return VLLMProvider(config)
    else:
        raise RuntimeError(f"Unsupported LLM provider: {provider_type}")


# Global provider instance (similar to Storage in provider.py)
def create_llm_agent(provider_type: str, config: Dict[str, Any]) -> LLMProvider:
    """Create LLM agent with specified provider"""
    return get_llm_provider(provider_type, config)


llm_agent = get_llm_provider(PROVIDER, CONFIG)
