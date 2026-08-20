"""
Vercel AI SDK Data Stream Protocol helpers.

This module provides utilities to generate Server-Sent Events (SSE) formatted
payloads compatible with the Vercel AI SDK Data Stream Protocol (v1).

Protocol reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol

The frontend must use `useChat` from `@ai-sdk/react` which expects these
SSE event types when the `x-vercel-ai-ui-message-stream: v1` header is set.
"""

import json
import uuid
from typing import Any, Dict, List, Optional


def _sse(payload: Dict[str, Any]) -> str:
    """Format a dictionary as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Message lifecycle events
# ---------------------------------------------------------------------------

def message_start(message_id: str) -> str:
    """Emit a message-start event."""
    return _sse({"type": "start", "messageId": message_id})


def start_step() -> str:
    """Emit a start-step event (beginning of one LLM call)."""
    return _sse({"type": "start-step"})


def finish_step() -> str:
    """Emit a finish-step event (one LLM call completed)."""
    return _sse({"type": "finish-step"})


def finish_message() -> str:
    """Emit a finish event (entire message completed)."""
    return _sse({"type": "finish"})


def stream_done() -> str:
    """Emit the terminal [DONE] marker."""
    return "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Text streaming (start / delta / end)
# ---------------------------------------------------------------------------

def text_start(text_id: str) -> str:
    return _sse({"type": "text-start", "id": text_id})


def text_delta(text_id: str, delta: str) -> str:
    return _sse({"type": "text-delta", "id": text_id, "delta": delta})


def text_end(text_id: str) -> str:
    return _sse({"type": "text-end", "id": text_id})


# ---------------------------------------------------------------------------
# Reasoning streaming (start / delta / end)
# ---------------------------------------------------------------------------

def reasoning_start(reasoning_id: str) -> str:
    return _sse({"type": "reasoning-start", "id": reasoning_id})


def reasoning_delta(reasoning_id: str, delta: str) -> str:
    return _sse({"type": "reasoning-delta", "id": reasoning_id, "delta": delta})


def reasoning_end(reasoning_id: str) -> str:
    return _sse({"type": "reasoning-end", "id": reasoning_id})


# ---------------------------------------------------------------------------
# Tool call streaming
# ---------------------------------------------------------------------------

def tool_input_start(tool_call_id: str, tool_name: str) -> str:
    return _sse({
        "type": "tool-input-start",
        "toolCallId": tool_call_id,
        "toolName": tool_name,
    })


def tool_input_delta(tool_call_id: str, input_text_delta: str) -> str:
    return _sse({
        "type": "tool-input-delta",
        "toolCallId": tool_call_id,
        "inputTextDelta": input_text_delta,
    })


def tool_input_available(
    tool_call_id: str, tool_name: str, tool_input: Dict[str, Any]
) -> str:
    return _sse({
        "type": "tool-input-available",
        "toolCallId": tool_call_id,
        "toolName": tool_name,
        "input": tool_input,
    })


def tool_output_available(
    tool_call_id: str, output: Any
) -> str:
    return _sse({
        "type": "tool-output-available",
        "toolCallId": tool_call_id,
        "output": output,
    })


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def source_url(source_id: str, url: str, title: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {
        "type": "source-url",
        "sourceId": source_id,
        "url": url,
    }
    if title:
        payload["title"] = title
    return _sse(payload)


def source_document(
    source_id: str,
    media_type: str = "file",
    title: Optional[str] = None,
) -> str:
    payload: Dict[str, Any] = {
        "type": "source-document",
        "sourceId": source_id,
        "mediaType": media_type,
    }
    if title:
        payload["title"] = title
    return _sse(payload)


# ---------------------------------------------------------------------------
# Custom data parts
# ---------------------------------------------------------------------------

def data_part(data_type: str, data: Any) -> str:
    """Emit a custom data-* part (e.g. data-echart, data-agent)."""
    return _sse({"type": f"data-{data_type}", "data": data})


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

def error_part(error_text: str) -> str:
    return _sse({"type": "error", "errorText": error_text})


# ---------------------------------------------------------------------------
# Convenience: convert sources list to SSE events
# ---------------------------------------------------------------------------

def emit_sources(sources: List[Dict]) -> str:
    """Convert a list of source dicts into SSE source-document events."""
    result = ""
    for src in sources:
        sid = src.get("id") or src.get("file_id") or str(uuid.uuid4())
        title = src.get("filename") or src.get("title") or src.get("file_name") or ""
        result += source_document(source_id=sid, title=title)
    return result


def generate_id() -> str:
    """Generate a unique ID for Vercel AI SDK stream parts."""
    return str(uuid.uuid4()).replace("-", "")[:24]
