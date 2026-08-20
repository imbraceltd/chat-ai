from pydantic import BaseModel
from typing import Optional, List, Dict


class AnswerQuestionInputs(BaseModel):
    """Input parameters matching the JavaScript answerQuestion function"""

    question: str = ""
    file_content: str = ""
    assistant_id: str = ""
    provider_id: Optional[str] = None  # Optional provider ID for custom LLM config
    meta_data: Optional[Dict] = None  # Optional metadata (e.g. tool_server)
    file_ids: List[str] = []
    folder_ids: List[str] = []
    instructions: str = ""
    thread_id: str = ""
    board_ids: List[str] = []
    conversation_id: str = ""
    message_id: str = ""
    message_created_at: str = ""
    created_at: str = ""
    knowledge_hubs: List[str] = []
    workflow_function_call: Optional[List] = []
    tools: List[Dict] = []
    show_thinking_process: bool = False
    guardrail_id: str = ""
    preload_information: str = ""
    streaming: bool = False
    sub_agents: Optional[List[str]] = []
    visualization: bool = False
    default_folder_id: Optional[str] = None
    response_mode: Optional[str] = "standard"


class UploadResult(BaseModel):
    """Result from file upload matching JS return structure"""

    id: str
    bytes: int
    filename: str
    assistant_id: str
    created_at: str
    board_id: str
    boarditem_id: str


class AnswerChunk(BaseModel):
    """Response chunk for streaming answers"""

    thread_id: str
    message: str
    reasoning: Optional[str] = None
    is_partial: Optional[bool] = False
    message_id: Optional[str] = ""
    sources: Optional[List[Dict]] = []
    echart: Optional[Dict] = None
    echart_id: Optional[str] = ""


class ToolCallChunk(BaseModel):
    """Response chunk for tool calls"""

    thread_id: str
    toolCall: str
    args: Optional[Dict] = {}
    is_partial: Optional[bool] = False
    message_id: Optional[str] = ""
    sources: Optional[List[Dict]] = []


class AgentChunk(BaseModel):
    """Response chunk for agent calls"""

    thread_id: str
    agent: str
    args: Optional[Dict] = {}
    is_partial: Optional[bool] = False
    message_id: Optional[str] = ""
    sources: Optional[List[Dict]] = []


class MessageDetails(BaseModel):
    """Message details for getLatestMessage"""

    created_at: str = ""
    conversation_id: str = ""
    message_id: str = ""
