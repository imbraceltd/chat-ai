"""
Tool definitions for LLM agents
Python equivalents of JavaScript tool definitions from tool.service.js
"""

import asyncio
import aiohttp
import logging
import os
from typing import Dict, Any, List, Optional, AsyncGenerator

try:
    from strands.types.tools import AgentTool, ToolSpec, ToolUse, ToolResult
except ImportError:
    # Fallback if strands is not available
    AgentTool = None
    ToolSpec = None
    ToolUse = None
    ToolResult = None

logger = logging.getLogger(__name__)

# Configuration - would typically come from config module
ENV = "development"  # This would come from config.environment
NOKIA_WEBHOOK_URL = os.environ.get("NOKIA_WEBHOOK_URL", "")


async def display_ai_prediction() -> str:
    """
    Display AI prediction - Python equivalent of displayAIPrediction function
    """
    try:
        logger.info("displayAIPrediction")

        async with aiohttp.ClientSession() as session:
            async with session.get(NOKIA_WEBHOOK_URL) as response:
                data = await response.json()
                return data

    except Exception as error:
        logger.error(f"Error in display_ai_prediction: {error}")
        return "can not send message"


def create_markdown_tool() -> Dict[str, Any]:
    """
    Create markdown tool definition
    Python equivalent of markdownTool
    """
    return {
        "type": "function",
        "function": {
            "name": "markdown_tool",
            "description": "Convert the input text to markdown format. The input text should be in plain text format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the markdown document.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The content of the markdown document. This should be in plain text format.",
                    },
                },
                "required": ["title"],
            },
        },
    }


def create_generate_echarts_config_tool() -> Dict[str, Any]:
    """
    Create ECharts config generation tool definition
    Python equivalent of generateEChartsConfigTool
    """
    return {
        "type": "function",
        "function": {
            "name": "generate_echarts_config",
            "description": "Generates an Apache ECharts JSON configuration for visualizing data as a bar, line, or pie chart. Supports multiple value fields with individual series types (bar or line) and dual y-axes for different scales. Includes additional properties in tooltips.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chartType": {
                        "type": "string",
                        "enum": ["bar", "line", "pie"],
                        "description": "Base chart type: 'bar', 'line', or 'pie'.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional chart title. Defaults to 'Data Visualization: [valueKeys] by [labelKey]'.",
                    },
                    "labelKey": {
                        "type": "string",
                        "description": "Property name for chart labels (string or time-based property for xAxis in bar/line, or name in pie). Defaults to first string or time-based property.",
                    },
                    "valueKeys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of property names for chart values (numeric properties). Defaults to first numeric property.",
                    },
                    "seriesTypes": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["bar", "line"]},
                        "description": "Array of series types (bar or line) for each valueKey in bar or line charts. Must match valueKeys length. Defaults to chartType.",
                    },
                    "groupBy": {
                        "type": ["string", "null"],
                        "description": "Optional property name to group data for multi-series bar or line charts (not for pie).",
                    },
                    "xAxisType": {
                        "type": "string",
                        "enum": ["category", "value", "time"],
                        "description": "Type of xAxis for bar or line charts (category, value, or time). Defaults to 'category'.",
                    },
                    "yAxis": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "valueKey": {
                                    "type": "string",
                                    "description": "Value key associated with this yAxis.",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["value", "category"],
                                    "description": "Type of yAxis.",
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Optional name for the yAxis.",
                                },
                            },
                            "required": ["valueKey", "type"],
                        },
                        "description": "Array of yAxis configurations for bar or line charts, allowing dual y-axes for different scales.",
                    },
                    "tooltip": {
                        "type": "object",
                        "properties": {
                            "trigger": {
                                "type": "string",
                                "enum": ["item", "axis", "none"],
                                "description": "Tooltip trigger type. Defaults to 'item' for pie, 'axis' for bar/line.",
                            },
                            "extraFields": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Additional data fields to include in tooltips.",
                            },
                        },
                        "required": ["trigger"],
                    },
                    "legend": {
                        "type": ["object", "null"],
                        "properties": {
                            "dataField": {
                                "type": "string",
                                "description": "Optional property name for legend data (used when groupBy is specified).",
                            },
                        },
                        "description": "Legend configuration for multi-series charts.",
                    },
                },
                "required": [
                    "chartType",
                    "valueKeys",
                    "labelKey",
                    "xAxisType",
                    "yAxis",
                    "tooltip",
                    "seriesTypes",
                    "legend",
                    "title",
                ],
            },
        },
    }


def create_combine_json_tool() -> Dict[str, Any]:
    """
    Create JSON combination tool definition
    Python equivalent of combineJSONTool
    """
    return {
        "type": "function",
        "function": {
            "name": "combine_json_tool",
            "description": "Combines multiple JSON arrays based on a specified groupBy field",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Array of JSON arrays to combine",
                        "items": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "groupBy": {
                        "type": "string",
                        "description": "The common field to group by when combining JSON data",
                    },
                },
                "required": ["data", "groupBy"],
            },
        },
    }


def create_retriever_tool() -> Dict[str, Any]:
    """
    Create retriever tool definition
    Python equivalent of retrieverTool
    """
    return {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": """Fetch relevant information from the knowledge base based on the user query.
          Generate two types of search queries:
            - vector_search_query: A natural, full-sentence question that reflects the user's intent and provides semantic context.
            - full_text_search_query: A concise keyword-style query using the core terms from the original question to improve full-text search precision.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "vector_search_query": {
                        "type": "string",
                        "description": "A natural language query for semantic (vector-based) search",
                    },
                    "full_text_search_query": {
                        "type": "string",
                        "description": "A short, keyword-focused query for full-text search",
                    },
                },
                "required": ["vector_search_query", "full_text_search_query"],
            },
        },
    }


def create_reset_memory_tool() -> Dict[str, Any]:
    """
    Create reset memory tool definition
    Python equivalent of resetMemoryTool
    """
    return {
        "type": "function",
        "function": {
            "name": "reset_memory",
            "description": "Clear conversation history and reset the memory of the bot. Only use this tool when the user explicitly requests it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_id": {
                        "type": "string",
                        "description": "Unique identifier of the conversation",
                    },
                },
            },
        },
    }


def create_aggregation_tool() -> Dict[str, Any]:
    """
    Create aggregation tool definition
    Python equivalent of aggregationTool
    """
    return {
        "type": "function",
        "function": {
            "name": "aggregate_context",
            "description": "Perform aggregation by using aggregate pipeline in mongo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query for retrieving relevant documents",
                    },
                },
                "required": ["query"],
            },
        },
    }


def create_aggregation_tool_v2() -> Dict[str, Any]:
    """
    Create aggregation tool v2 definition
    Python equivalent of aggregationToolV2
    """
    return {
        "type": "function",
        "function": {
            "name": "aggregate_context",
            "description": "Perform aggregation by using aggregate pipeline in mongo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aggregate_pipeline": {
                        "type": "string",
                        "description": "a mongo aggregate pipeline",
                    },
                },
                "required": ["query"],
            },
        },
    }


def create_run_ai_prediction_tool() -> Dict[str, Any]:
    """
    Create AI prediction tool definition
    Python equivalent of runAIPredictionTool
    """
    return {
        "type": "function",
        "function": {
            "name": "run_ai_prediction",
            "description": "Run ai prediction tool.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }


def create_parquet_query_tool() -> Dict[str, Any]:
    """
    Create parquet query tool definition for querying S3 parquet files with DuckDB
    """
    return {
        "type": "function",
        "function": {
            "name": "query_parquet",
            "description": "Execute SQL queries on parquet files stored in S3. Use this tool when you need to analyze structured data from parquet files. The query should be a valid SQL statement that can be executed by DuckDB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute on the parquet files. Use standard SQL syntax supported by DuckDB.",
                    },
                    "parquet_files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "s3_path": {
                                    "type": "string",
                                    "description": "S3 path to the parquet file",
                                },
                                "alias": {
                                    "type": "string",
                                    "description": "Optional table alias for the parquet file in the query",
                                },
                            },
                            "required": ["s3_path"],
                        },
                        "description": "List of parquet files to query, each with S3 path and optional alias",
                    },
                },
                "required": ["query", "parquet_files"],
            },
        },
    }


def create_folder_info_tool() -> Dict[str, Any]:
    """
    Create folder information tool definition.
    Queries the data-board API to retrieve folder structure, file counts and file listings
    for the AI assistant's configured folder IDs.
    Useful for answering questions like "How many documents/files are in the folders?"
    """
    return {
        "type": "function",
        "function": {
            "name": "query_folder_info",
            "description": (
                "Retrieve folder structure information including file counts, file names, "
                "and subfolder hierarchy for the AI assistant's configured folders. "
                "Use this tool when the user asks about the number of documents or files "
                "stored in folders, the folder structure, what files exist in a folder, "
                "or any folder/file inventory questions. The folder IDs are automatically "
                "resolved from the assistant configuration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's original question about folder contents or file inventory.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to include nested subfolders recursively. Defaults to true.",
                        "default": True,
                    },
                },
                "required": ["query"],
            },
        },
    }


class ToolRegistry:
    """Registry for managing tool definitions"""

    def __init__(self):
        self._tools = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools from tool.service.js"""
        self.register("markdown_tool", create_markdown_tool)
        self.register("generate_echarts_config", create_generate_echarts_config_tool)
        self.register("combine_json_tool", create_combine_json_tool)
        self.register("retrieve_context", create_retriever_tool)
        self.register("reset_memory", create_reset_memory_tool)
        self.register("aggregate_context", create_aggregation_tool)
        self.register("aggregate_context_v2", create_aggregation_tool_v2)
        self.register("run_ai_prediction", create_run_ai_prediction_tool)
        self.register("query_parquet", create_parquet_query_tool)
        self.register("query_folder_info", create_folder_info_tool)

    def register(self, name: str, tool_factory):
        """Register a tool factory function"""
        self._tools[name] = tool_factory

    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get a tool definition by name"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self._tools[name]()

    def get_tools(self, names: List[str]) -> List[Dict[str, Any]]:
        """Get multiple tool definitions"""
        return [self.get_tool(name) for name in names]

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tool definitions"""
        return [factory() for factory in self._tools.values()]

    def list_tool_names(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())


# Global tool registry instance
tool_registry = ToolRegistry()


# Convenience functions for accessing tools (equivalent to JS exports)
def get_markdown_tool() -> Dict[str, Any]:
    """Get markdown tool definition"""
    return tool_registry.get_tool("markdown_tool")


def get_generate_echarts_config_tool() -> Dict[str, Any]:
    """Get ECharts config generation tool definition"""
    return tool_registry.get_tool("generate_echarts_config")


def get_combine_json_tool() -> Dict[str, Any]:
    """Get JSON combination tool definition"""
    return tool_registry.get_tool("combine_json_tool")


def get_retriever_tool() -> Dict[str, Any]:
    """Get retriever tool definition"""
    return tool_registry.get_tool("retrieve_context")


def get_reset_memory_tool() -> Dict[str, Any]:
    """Get reset memory tool definition"""
    return tool_registry.get_tool("reset_memory")


def get_aggregation_tool() -> Dict[str, Any]:
    """Get aggregation tool definition"""
    return tool_registry.get_tool("aggregate_context")


def get_aggregation_tool_v2() -> Dict[str, Any]:
    """Get aggregation tool v2 definition"""
    return tool_registry.get_tool("aggregate_context_v2")


def get_run_ai_prediction_tool() -> Dict[str, Any]:
    """Get AI prediction tool definition"""
    return tool_registry.get_tool("run_ai_prediction")


def get_parquet_query_tool() -> Dict[str, Any]:
    """Get parquet query tool definition"""
    return tool_registry.get_tool("query_parquet")


def get_folder_info_tool() -> Dict[str, Any]:
    """Get folder info tool definition"""
    return tool_registry.get_tool("query_folder_info")

# Predefined tool sets for common use cases
def get_standard_tools() -> List[Dict[str, Any]]:
    """Get standard tool set for most use cases"""
    return tool_registry.get_tools(["reset_memory", "retrieve_context"])


def get_rag_tools() -> List[Dict[str, Any]]:
    """Get tools for RAG functionality"""
    return tool_registry.get_tools(
        ["reset_memory", "retrieve_context", "aggregate_context"]
    )


def get_nokia_tools() -> List[Dict[str, Any]]:
    """Get tools for Nokia-specific functionality"""
    return tool_registry.get_tools(
        ["reset_memory", "retrieve_context", "aggregate_context", "run_ai_prediction"]
    )


def get_visualization_tools() -> List[Dict[str, Any]]:
    """Get tools for data visualization"""
    return tool_registry.get_tools(
        ["generate_echarts_config", "combine_json_tool", "aggregate_context"]
    )


def get_all_available_tools() -> List[Dict[str, Any]]:
    """Get all available tool definitions"""
    return tool_registry.get_all_tools()


class OpenAIAgentTool(AgentTool if AgentTool else object):
    """Concrete implementation of AgentTool for OpenAI-format tools converted to Strands format."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        is_dynamic: bool = False,
    ):
        """Initialize the OpenAI Agent Tool.

        Args:
            name: The unique name of the tool
            description: Description of what the tool does
            input_schema: JSON Schema for the tool's input parameters
            is_dynamic: Whether the tool is dynamic
        """
        if AgentTool:
            super().__init__()
        self._name = name
        self._description = description
        self._input_schema = input_schema
        self._is_dynamic = is_dynamic

    @property
    def tool_name(self) -> str:
        """The unique name of the tool used for identification and invocation."""
        return self._name

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """Tool specification that describes its functionality and parameters."""
        if ToolSpec:
            # Return ToolSpec TypedDict format
            return {
                "name": self._name,
                "description": self._description,
                "inputSchema": self._input_schema,
            }
        else:
            # Fallback format
            return {
                "name": self._name,
                "description": self._description,
                "inputSchema": self._input_schema,
            }

    @property
    def tool_type(self) -> str:
        """The type of the tool implementation."""
        return "openai_converted"

    async def stream(
        self, tool_use: Dict[str, Any], invocation_state: Dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        """Stream tool events and return the final result.

        This is a placeholder implementation that yields an error result.
        Actual tool execution should be handled by the agent framework.

        Args:
            tool_use: The tool use request containing tool ID and parameters
            invocation_state: Caller-provided kwargs from agent invocation
            **kwargs: Additional keyword arguments

        Yields:
            Tool result indicating this tool needs external handling
        """
        # This is a placeholder - actual execution happens in the agent framework
        result = {
            "content": [
                {
                    "text": f"Tool '{self._name}' execution should be handled by the agent framework"
                }
            ],
            "status": "error",
            "toolUseId": tool_use.get("toolUseId", ""),
        }
        yield result


def convert_openai_tool_to_bedrock_tools(tools: List[Dict[str, Any]]) -> List[Any]:
    """
    Converts a list of OpenAI-format tools to Strands AgentTool instances.

    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "...",
            "parameters": {...}
        }
    }

    Strands AgentTool format with ToolSpec:
    AgentTool instance with:
    - tool_name: str
    - tool_spec: ToolSpec with name, description, inputSchema
    - tool_type: str

    Args:
        tools: List of OpenAI-format tool definitions

    Returns:
        List of AgentTool instances (or dict fallback if strands not available)
    """
    agent_tools = []

    for tool in tools:
        if not isinstance(tool, dict):
            logger.warning(f"Skipping non-dict tool: {tool}")
            continue

        # Extract function details from OpenAI format
        if "function" in tool and isinstance(tool["function"], dict):
            function = tool["function"]
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters", {})
        elif "type" in tool and "function" in tool:
            # Already has the structure we need
            function = tool["function"]
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters", {})
        else:
            # Assume it's already in a simple format
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})

        if not name:
            logger.warning(f"Skipping tool without name: {tool}")
            continue

        # Create AgentTool instance or fallback dict
        if AgentTool:
            agent_tool = OpenAIAgentTool(
                name=name,
                description=description,
                input_schema=parameters,
                is_dynamic=True,
            )
            agent_tools.append(agent_tool)
            logger.debug(f"Converted tool '{name}' to AgentTool instance")
        else:
            # Fallback to dict format if strands not available
            bedrock_tool = {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": parameters},
                }
            }
            agent_tools.append(bedrock_tool)
            logger.debug(
                f"Converted tool '{name}' to dict format (strands not available)"
            )

    logger.info(
        f"Converted {len(agent_tools)} tools from OpenAI format to AgentTool instances"
    )
    return agent_tools


def convert_openai_tools_to_bedrock_tools(tools: List[Dict[str, Any]]) -> List[Any]:
    """
    Converts a list of Strands AgentTool definitions to OpenAI-format tools.

    Strands AgentTool format with ToolSpec:
    {
        "toolSpec": {
            "name": "tool_name",
            "description": "...",
            "inputSchema": {
                "json": {...}
            }
        }
    }
    OpenAI format:
    {
        "type": "function",
        "function": {
            "name": "tool_name",
            "description": "...",
            "parameters": {...}
        }
    }
    """
    bedrock_tools = []

    for tool in tools:
        if not isinstance(tool, dict):
            logger.warning(f"Skipping non-dict tool: {tool}")
            continue

        # Extract function details from OpenAI format
        if "function" in tool and isinstance(tool["function"], dict):
            function = tool["function"]
            name = function.get("name", "")
            description = function.get("description", "")
            parameters = function.get("parameters", {})
        else:
            logger.warning(f"Skipping tool without function: {tool}")
            continue

        if not name:
            logger.warning(f"Skipping tool without name: {tool}")
            continue

        # Create Bedrock tool definition
        bedrock_tool = {
            "toolSpec": {
                "name": name,
                "description": description,
                "inputSchema": {"json": parameters},
            }
        }
        bedrock_tools.append(bedrock_tool)
        logger.debug(f"Converted tool '{name}' to Bedrock format")

    logger.info(
        f"Converted {len(bedrock_tools)} tools from OpenAI format to Bedrock format"
    )
    return bedrock_tools
