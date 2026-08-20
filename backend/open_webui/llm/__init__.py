"""
LLM Package
Provider-agnostic LLM agent interface and implementations
"""

from .main import AnswerQuestionInputs, AnswerChunk, UploadResult, MessageDetails
from .agent import LLMProvider, create_llm_agent, get_llm_provider
from .models import (
    OpenAIProvider,
    OllamaProvider,
    LMStudioProvider,
    BedrockProvider,
    VLLMProvider,
    GoogleAIProvider,
)

__all__ = [
    # Data models
    "AnswerQuestionInputs",
    "AnswerChunk",
    "UploadResult",
    "MessageDetails",
    # Provider pattern
    "LLMProvider",
    "create_llm_agent",
    "get_llm_provider",
    # Provider implementations
    "OpenAIProvider",
    "OllamaProvider",
    "LMStudioProvider",
    "BedrockProvider",
    "VLLMProvider",
    "GoogleAIProvider",
]
