"""
Document AI assistant functions for enhanced PDF processing.

This module provides V2 and V3 versions of assistant functions for processing
documents with vision-capable AI models. V3 includes enhanced chunking and
concurrent processing for large PDFs.
"""

import json
import copy
import logging
import asyncio
from typing import Dict, Any, Optional, List, Union
import math
import re
import io
import httpx
from PIL import Image
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from open_webui.utils.document import (
    fetch_url_as_base64_string,
    pdf_base64_to_png_base64,
    docx_base64_to_text,
)

log = logging.getLogger(__name__)


async def _run_ocr(images_base64, ocr_engine: Optional[str] = None, page_numbers: Optional[List[int]] = None) -> List[Any]:
    """Run the configured real-OCR engine over page image(s).

    Returns a list of OcrToken (or [] when OCR is unavailable / fails). Designed to
    be fired concurrently with the Step-1 vision LLM call (the engine offloads the
    CPU-bound work to a worker thread), so it overlaps the network-bound LLM call
    with ~no added wall-clock. Never raises into the caller.
    """
    if not images_base64:
        return []
    images = images_base64 if isinstance(images_base64, list) else [images_base64]
    try:
        from open_webui.utils.ocr import create_ocr_engine

        engine = create_ocr_engine(ocr_engine)
        if engine is None:
            return []
        tokens = await engine.extract_tokens(images, page_numbers)
        log.info(f"OCR ({engine.get_name()}): {len(tokens)} tokens from {len(images)} image(s)")
        return tokens
    except Exception as e:
        log.warning(f"OCR failed, continuing without grounded confidence: {e}")
        return []


def _ground_confidence(data: Any, tokens: List[Any], threshold: float = 0.0) -> Any:
    """Attach grounded {value, box, confidence} to each leaf of ``data`` using OCR tokens.

    CPU-bound (string matching); call via ``asyncio.to_thread`` so it never blocks the
    event loop. Returns ``data`` unchanged when there are no tokens or on any failure.
    """
    if not tokens:
        return data
    try:
        import time
        from open_webui.utils.ocr import reconcile_fields

        start = time.time()
        grounded = reconcile_fields(data, tokens, threshold)
        log.info(f"OCR grounding: reconciled against {len(tokens)} tokens in {time.time() - start:.2f}s")
        return grounded
    except Exception as e:
        log.warning(f"OCR reconciliation failed, returning ungrounded data: {e}")
        return data


async def _ground_confidence_async(data: Any, tokens: List[Any], threshold: float = 0.0) -> Any:
    """Run :func:`_ground_confidence` off the event loop."""
    if not tokens:
        return data
    return await asyncio.to_thread(_ground_confidence, data, tokens, threshold)


def _ocr_kwargs_from_options(options: Dict[str, Any]) -> Dict[str, Any]:
    """Build the OCR-confidence kwargs shared by the V2 assistants from an options dict.

    OCR confidence is auto-enabled when ``extractRawData`` is set (that mode already
    emits the wrapped {value, box, confidence} shape) or when ``enableOcrConfidence``
    is set explicitly.
    """
    return {
        "enable_ocr_confidence": bool(
            options.get("enableOcrConfidence") or options.get("extractRawData")
        ),
        "ocr_engine": options.get("ocrEngine"),
        "confidence_threshold": options.get("confidenceThreshold") or 0.0,
    }


def _format_image_content(base64_data: str, file_extension: str = "png", provider: Optional[str] = None) -> Dict[str, Any]:
    """Format a base64-encoded image for the appropriate provider.

    - Google/Gemini: uses 'media' type with mime_type and data (ChatGoogleGenerativeAI)
    - Bedrock: uses 'image' type with source (ChatBedrockConverse native format)
    - Others (OpenAI, etc.): uses 'image_url' with data URI
    """
    mime_type = f"image/{file_extension}"
    if provider in ("google", "gemini"):
        return {
            "type": "media",
            "mime_type": mime_type,
            "data": base64_data,
        }
    if provider == "bedrock":
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": base64_data},
        }
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
    }


# Keys not supported by Bedrock's structured-output JSON Schema subset (Draft 2020-12 subset).
# See https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html
_BEDROCK_UNSUPPORTED_SCHEMA_KEYS = {
    "example", "default", "title", "format",
    "minLength", "maxLength", "pattern",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minItems", "maxItems", "uniqueItems",
    "$ref", "$defs", "definitions", "$schema", "$id",
}


def _sanitize_schema_node(node: Any) -> Any:
    """Recursively rewrite one JSON-schema node into Bedrock's supported subset."""
    if not isinstance(node, dict):
        return node

    # Determine effective type (tolerant of loose/auto schemas without an explicit type).
    node_type = node.get("type")
    if node_type is None:
        if "properties" in node:
            node_type = "object"
        elif "items" in node:
            node_type = "array"

    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _BEDROCK_UNSUPPORTED_SCHEMA_KEYS:
            continue
        out[key] = value

    if node_type == "object":
        out["type"] = "object"
        props = node.get("properties") or {}
        sanitized_props = {k: _sanitize_schema_node(v) for k, v in props.items()}
        out["properties"] = sanitized_props
        # Bedrock structured output requires every property be listed in `required`.
        out["required"] = list(sanitized_props.keys())
        out["additionalProperties"] = False
    elif node_type == "array":
        out["type"] = "array"
        if "items" in node:
            out["items"] = _sanitize_schema_node(node["items"])
    else:
        # Scalar (or unknown). Coerce unknown/non-standard types to string; keep enum.
        out["type"] = node_type if node_type in ("string", "number", "integer", "boolean", "null") else "string"

    return out


def _sanitize_schema_for_bedrock(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy a (loose) board output schema into Bedrock's structured-output JSON
    Schema subset: every object gets additionalProperties=false and a full `required`
    list, and unsupported keywords are stripped. Never mutates the input (the same
    schema is still embedded as prompt text elsewhere).
    """
    return _sanitize_schema_node(copy.deepcopy(schema))


def _wrap_schema_for_array(object_schema: Dict[str, Any], key: str = "records") -> Dict[str, Any]:
    """Wrap an object schema so the structured-output root is an object containing an
    array. Bedrock json_schema requires an object root, not a bare array."""
    return {
        "type": "object",
        "properties": {key: {"type": "array", "items": object_schema}},
    }


def _sanitize_schema_node_strict(node: Any, required_in_parent: bool = True) -> Any:
    """Rewrite one JSON-schema node into the OpenAI/vLLM strict json_schema subset.

    Like the Bedrock sanitizer (additionalProperties=false, every property listed in
    `required`, unsupported keywords stripped), but additionally makes absent-optional
    scalar fields **nullable** (``type: [base, "null"]``). Strict mode requires every
    property in `required`, so without nullability the model is forced to invent a value
    for fields that aren't present; nullability lets it emit ``null`` instead.

    ``required_in_parent`` is whether this node's field was in its parent's original
    ``required`` list. Nested objects keep their own `required`, so array rows / sub-objects
    stay non-null.
    """
    if not isinstance(node, dict):
        return node

    node_type = node.get("type")
    if node_type is None:
        if "properties" in node:
            node_type = "object"
        elif "items" in node:
            node_type = "array"

    out: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _BEDROCK_UNSUPPORTED_SCHEMA_KEYS:
            continue
        out[key] = value

    if node_type == "object":
        out["type"] = "object"
        props = node.get("properties") or {}
        orig_required = set(node.get("required") or [])
        sanitized_props = {
            k: _sanitize_schema_node_strict(v, required_in_parent=(k in orig_required))
            for k, v in props.items()
        }
        out["properties"] = sanitized_props
        out["required"] = list(sanitized_props.keys())
        out["additionalProperties"] = False
    elif node_type == "array":
        out["type"] = "array"
        if "items" in node:
            # Array rows recurse with their own object `required`, so they stay non-null.
            out["items"] = _sanitize_schema_node_strict(node["items"], required_in_parent=True)
    else:
        base = node_type if node_type in ("string", "number", "integer", "boolean", "null") else "string"
        if not required_in_parent and base != "null":
            out["type"] = [base, "null"]
            # A nullable union with an enum must allow null as a value.
            if isinstance(out.get("enum"), list) and None not in out["enum"]:
                out["enum"] = [*out["enum"], None]
        else:
            out["type"] = base

    return out


def _sanitize_schema_for_vllm(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy a (loose) board output schema into the strict OpenAI/vLLM json_schema
    subset for guided decoding. Absent-optional scalars become nullable so the model can
    emit null rather than hallucinate. Never mutates the input (the loose schema is still
    embedded as prompt text by the Step-2 callers)."""
    return _sanitize_schema_node_strict(copy.deepcopy(schema), required_in_parent=True)


def _bedrock_supports_structured_output(model_id: str) -> bool:
    """Whether a Bedrock model id supports native structured output (outputConfig.textFormat).

    Supported families: Qwen3, Kimi/Moonshot, Mistral, DeepSeek, gpt-oss, MiniMax,
    Nemotron, Gemma, and Claude 4.x. NOT supported: Llama, Claude 3.x, Titan, Nova.
    """
    if not model_id:
        return False
    m = model_id.lower()
    supported = ("qwen", "kimi", "moonshot", "mistral", "deepseek",
                 "gpt-oss", "minimax", "nemotron", "gemma")
    if any(s in m for s in supported):
        return True
    if "claude" in m or "anthropic" in m:
        return any(fam in m for fam in ("sonnet-4", "opus-4", "haiku-4", "claude-4"))
    return False


async def _bedrock_converse_structured_http(
    model,
    messages,
    output_schema: Dict[str, Any],
    max_tokens: int = 8000,
    schema_name: str = "document_output",
):
    """Call the Bedrock Converse API over signed HTTP (httpx + SigV4), sending a native
    structured-output `outputConfig.textFormat` JSON schema.

    Used instead of the boto3 client because the pinned botocore (1.36.x) predates the
    `outputConfig` parameter; raw signed HTTP bypasses botocore's bundled service model.
    Text-only (DocumentAI Step-2 messages carry no images). Returns a
    SimpleNamespace(content=...) to match the LangChain response contract.
    """
    import boto3
    from urllib.parse import quote
    from types import SimpleNamespace
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from langchain_core.messages import SystemMessage, HumanMessage

    # Extract model id + credentials from the LangChain ChatBedrockConverse instance.
    model_id = getattr(model, "model_id", None) or getattr(model, "model", None) or "unknown"
    region = getattr(model, "region_name", None) or "us-east-1"
    access_key = getattr(model, "aws_access_key_id", None)
    secret_key = getattr(model, "aws_secret_access_key", None)
    if hasattr(access_key, "get_secret_value"):
        access_key = access_key.get_secret_value()
    if hasattr(secret_key, "get_secret_value"):
        secret_key = secret_key.get_secret_value()

    # Resolve credentials: explicit keys if present, else the default chain (env/role/SSO).
    session_kwargs = {"region_name": region}
    if access_key and secret_key:
        session_kwargs["aws_access_key_id"] = access_key
        session_kwargs["aws_secret_access_key"] = secret_key
    session = boto3.Session(**session_kwargs)
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials available for Bedrock structured output")

    # Convert LangChain messages -> Converse format (text only).
    system_parts = []
    converse_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append({"text": str(msg.content)})
            continue
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content
        if isinstance(content, str):
            converse_messages.append({"role": role, "content": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append({"text": item.get("text", "")})
            converse_messages.append({"role": role, "content": parts})

    sanitized = _sanitize_schema_for_bedrock(output_schema)
    body = {
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0, "topP": 0.1},
        "outputConfig": {
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "schema": json.dumps(sanitized),
                        "name": schema_name,
                        "description": "Structured document extraction output",
                    }
                },
            }
        },
    }
    if system_parts:
        body["system"] = system_parts

    # Sign and send the EXACT same body string (SigV4 payload hash must match).
    body_str = json.dumps(body)
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{quote(model_id, safe='')}/converse"
    aws_request = AWSRequest(
        method="POST",
        url=url,
        data=body_str,
        headers={"Content-Type": "application/json"},
    )
    SigV4Auth(credentials, "bedrock", region).add_auth(aws_request)

    log.info(f"Bedrock structured-output HTTP: model_id={model_id}, region={region}, "
             f"num_messages={len(converse_messages)}, num_system={len(system_parts)}, max_tokens={max_tokens}")

    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, content=body_str, headers=dict(aws_request.headers))
        if response.status_code != 200:
            log.error(f"Bedrock structured-output HTTP {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
        data = response.json()

    output = data.get("output", {})
    msg_content = output.get("message", {}).get("content", [])
    text_parts = [p.get("text", "") for p in msg_content if "text" in p]
    result_text = "\n".join(text_parts)
    log.info(f"Bedrock structured-output response: output_len={len(result_text)}")
    return SimpleNamespace(content=result_text)


async def _bedrock_converse_invoke(model, messages, max_tokens: int = 8000):
    """Call Bedrock Converse API directly via boto3, bypassing LangChain.

    Extracts credentials and model_id from the LangChain ChatBedrockConverse
    instance, builds the Converse payload manually, and returns a
    SimpleNamespace with .content matching LangChain's response shape.
    """
    import base64 as b64
    import boto3
    from types import SimpleNamespace
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    # Extract config from LangChain model
    model_id = getattr(model, 'model_id', None) or getattr(model, 'model', None) or 'unknown'
    region = getattr(model, 'region_name', None) or 'us-east-1'
    access_key = getattr(model, 'aws_access_key_id', None)
    secret_key = getattr(model, 'aws_secret_access_key', None)
    # Handle Pydantic SecretStr
    if hasattr(access_key, 'get_secret_value'):
        access_key = access_key.get_secret_value()
    if hasattr(secret_key, 'get_secret_value'):
        secret_key = secret_key.get_secret_value()

    log.info(f"Bedrock direct invoke: model_id={model_id}, region={region}, max_tokens={max_tokens}")

    # Build boto3 client
    client_kwargs = {"region_name": region}
    if access_key and secret_key:
        client_kwargs["aws_access_key_id"] = access_key
        client_kwargs["aws_secret_access_key"] = secret_key
    client = boto3.client("bedrock-runtime", **client_kwargs)

    # Convert LangChain messages to Converse API format
    system_parts = []
    converse_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append({"text": str(msg.content)})
            continue

        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = msg.content

        if isinstance(content, str):
            converse_messages.append({"role": role, "content": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        parts.append({"text": item.get("text", "")})
                    elif item_type == "image":
                        # Native bedrock format: {"type": "image", "source": {...}}
                        source = item.get("source", {})
                        img_bytes = b64.b64decode(source.get("data", ""))
                        fmt = source.get("media_type", "image/png").split("/")[-1]
                        parts.append({"image": {"format": fmt, "source": {"bytes": img_bytes}}})
                    elif item_type == "image_url":
                        # OpenAI format: data:image/png;base64,...
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            import re
                            m = re.match(r"^data:image/(?P<fmt>.+);base64,(?P<data>.+)$", url)
                            if m:
                                img_bytes = b64.b64decode(m.group("data"))
                                parts.append({"image": {"format": m.group("fmt"), "source": {"bytes": img_bytes}}})
                    elif item_type == "media":
                        # Google format
                        img_bytes = b64.b64decode(item.get("data", ""))
                        fmt = item.get("mime_type", "image/png").split("/")[-1]
                        parts.append({"image": {"format": fmt, "source": {"bytes": img_bytes}}})
            converse_messages.append({"role": role, "content": parts})

    # Build request
    request_kwargs = {
        "modelId": model_id,
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0, "topP": 0.1},
    }
    if system_parts:
        request_kwargs["system"] = system_parts

    log.info(f"Bedrock Converse payload: modelId={model_id}, "
             f"num_messages={len(converse_messages)}, "
             f"num_system={len(system_parts)}, "
             f"content_types={[p.get('image', {}).get('format') or 'text' for m in converse_messages for p in m.get('content', [])]}")

    # Call synchronously in thread to avoid blocking
    response = await asyncio.to_thread(client.converse, **request_kwargs)

    # Extract response text
    output = response.get("output", {})
    msg_content = output.get("message", {}).get("content", [])
    text_parts = [p.get("text", "") for p in msg_content if "text" in p]
    result_text = "\n".join(text_parts)

    log.info(f"Bedrock Converse response: status={response.get('ResponseMetadata', {}).get('HTTPStatusCode')}, "
             f"output_len={len(result_text)}")

    return SimpleNamespace(content=result_text)


def _extract_vllm_config(model) -> Optional[Dict[str, str]]:
    """Extract base_url, api_key, and model name from a ChatOpenAI (vLLM) instance."""
    try:
        from langchain_openai import ChatOpenAI
        if not isinstance(model, ChatOpenAI):
            return None
        base_url = str(model.openai_api_base or "")
        api_key = model.openai_api_key.get_secret_value() if model.openai_api_key else ""
        model_name = model.model_name or ""
        if not base_url:
            return None
        return {"base_url": base_url, "api_key": api_key, "model": model_name}
    except Exception:
        return None


def _vllm_structured_output_enabled() -> bool:
    """Whether vLLM should use strict json_schema structured output (config-gated)."""
    try:
        from open_webui.config import DOCUMENT_AI_VLLM_STRUCTURED_OUTPUT

        return bool(getattr(DOCUMENT_AI_VLLM_STRUCTURED_OUTPUT, "value", DOCUMENT_AI_VLLM_STRUCTURED_OUTPUT))
    except Exception:
        return True


async def vllm_http_invoke(
    model,
    messages: List,
    json_mode: bool = True,
    timeout: int = 120,
    enable_thinking: bool = False,
    max_tokens: Optional[int] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    schema_name: str = "document_output",
) -> str:
    """
    Invoke a vLLM model via direct HTTP instead of LangChain.

    Args:
        model: LangChain ChatOpenAI model (used to extract base_url/api_key/model)
        messages: List of LangChain BaseMessage or dicts
        json_mode: Whether to request JSON output
        timeout: Request timeout in seconds
        output_schema: When provided (and structured output is enabled), sent as a strict
            json_schema response_format for guided decoding; otherwise falls back to json_object.

    Returns:
        Raw response content string
    """
    config = _extract_vllm_config(model)
    if not config:
        raise ValueError("Cannot extract vLLM config from model")

    # Convert LangChain messages to OpenAI format
    formatted_messages = []
    for msg in messages:
        if hasattr(msg, 'content') and hasattr(msg, 'type'):
            # LangChain BaseMessage
            role = "system" if msg.type == "system" else "user" if msg.type == "human" else msg.type
            formatted_messages.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            formatted_messages.append(msg)
        else:
            formatted_messages.append({"role": "user", "content": str(msg)})

    base_url = config["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if config["api_key"]:
        headers["Authorization"] = f"Bearer {config['api_key']}"

    payload = {
        "model": config["model"],
        "messages": formatted_messages,
        "temperature": 0,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking
        },
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response_format_mode = "none"
    if output_schema and _vllm_structured_output_enabled():
        # Strict json_schema guided decoding — constrains generation to the board schema.
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": _sanitize_schema_for_vllm(output_schema),
            },
        }
        response_format_mode = "json_schema"
    elif json_mode:
        payload["response_format"] = {"type": "json_object"}
        response_format_mode = "json_object"
    log.info(
        f"Request custom vllm {url}, model={config['model']}, response_format={response_format_mode}"
    )
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            log.error(f"vLLM HTTP error {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        log.info(f"vLLM HTTP response received, length={len(content)}")
        return content


async def invoke_with_structured_output(model, messages, model_provider: Optional[str] = None, enable_thinking: bool = False, max_tokens: Optional[int] = None, output_schema: Optional[Dict[str, Any]] = None, schema_name: str = "document_output", json_mode: bool = True):
    """
    Invoke a model with structured JSON output appropriate for the provider.

    Args:
        model: LangChain chat model
        messages: Messages to send (list of BaseMessage or a prompt chain input)
        model_provider: Provider type ("openai", "vllm", "ollama", "gemini", "bedrock", etc.)
        output_schema: Optional JSON schema. When provided for a capable Bedrock model,
            enables native structured output (outputConfig.textFormat); ignored otherwise.
        json_mode: Whether to constrain the response to JSON. Currently honored by the vLLM
            branch: when False (and no output_schema), no response_format is sent, so the
            model extracts freely (used for Step 1 when Step 2 produces the JSON).

    Returns:
        Model response
    """
    if model_provider in ("gemini", "google"):
        # Gemini/Google doesn't support response_format binding, rely on prompt instructions
        # Set max_output_tokens directly on the model (not via .bind()) as ChatGoogleGenerativeAI
        # passes it as a constructor/model param, not an API invocation kwarg
        target_max = max_tokens or 65536
        if not getattr(model, 'max_output_tokens', None):
            model.max_output_tokens = target_max
        return await model.ainvoke(messages)
    elif model_provider == "bedrock":
        # Bedrock Converse supports native structured output via outputConfig.textFormat,
        # but only for certain model families and only on a recent botocore. We send it over
        # signed HTTP (bypassing the pinned botocore) when a schema is available and the model
        # supports it; otherwise fall back to prompt-based JSON (the historical behavior).
        model_id = getattr(model, "model_id", None) or getattr(model, "model", None) or ""
        if output_schema and _bedrock_supports_structured_output(model_id):
            try:
                return await _bedrock_converse_structured_http(
                    model, messages, output_schema,
                    max_tokens=max_tokens or 8000, schema_name=schema_name,
                )
            except Exception as e:
                log.warning(
                    f"Bedrock native structured output failed for {model_id}, "
                    f"falling back to prompt-based: {type(e).__name__}: {e}"
                )
        # Fallback: rely on prompt instructions for JSON output
        if max_tokens is not None:
            model = model.bind(max_tokens=max_tokens)
        return await model.ainvoke(messages)
    elif model_provider == "ollama":
        # Ollama uses format="json" natively
        json_model = model.bind(format="json")
        return await json_model.ainvoke(messages)
    elif model_provider == "vllm":
        # vLLM: use direct HTTP for faster processing. When a schema is available, request
        # strict json_schema (guided decoding). When json_mode is False (e.g. Step-1
        # extraction under a board — Step 2 emits the JSON), send no response_format at all,
        # so the model extracts freely instead of paying for slow/unreliable guided JSON.
        try:
            raw_content = await vllm_http_invoke(
                model, messages, json_mode=json_mode, timeout=900,
                enable_thinking=enable_thinking, max_tokens=max_tokens,
                output_schema=output_schema, schema_name=schema_name,
            )
            # Return as a simple object with .content to match LangChain response format
            from types import SimpleNamespace
            return SimpleNamespace(content=raw_content)
        except Exception as e:
            log.warning(f"vLLM direct HTTP failed: {e}, falling back to LangChain")
            bind_kwargs = {}
            if json_mode and not output_schema:
                bind_kwargs["response_format"] = {"type": "json_object"}
            if max_tokens is not None:
                bind_kwargs["max_tokens"] = max_tokens
            try:
                fallback_model = model.bind(**bind_kwargs) if bind_kwargs else model
                return await fallback_model.ainvoke(messages)
            except Exception:
                fallback = model.bind(max_tokens=max_tokens) if max_tokens is not None else model
                return await fallback.ainvoke(messages)
    else:
        # OpenAI / OpenAI-compatible providers
        bind_kwargs = {"response_format": {"type": "json_object"}}
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        try:
            json_model = model.bind(**bind_kwargs)
            return await json_model.ainvoke(messages)
        except Exception as e:
            if "response_format" in str(e) or "responseFormat" in str(e):
                log.warning(f"Model failed with response_format, retrying without it: {e}")
                fallback = model.bind(max_tokens=max_tokens) if max_tokens is not None else model
                return await fallback.ainvoke(messages)
            else:
                raise e


def _extract_content_text(response) -> str:
    """Extract text content from a model response (LangChain, SimpleNamespace, or dict)."""
    if hasattr(response, 'dict'):
        obj = response.dict()
        return obj.get("content", "") if isinstance(obj, dict) else str(obj)
    if hasattr(response, 'content'):
        return response.content
    if isinstance(response, dict):
        return response.get("content", "")
    return str(response)


def clean_json_response(raw_content: str) -> str:
    """
    Clean JSON response by removing markdown wrappers and thinking tokens.
    
    Args:
        raw_content: Raw content from LLM response
        
    Returns:
        Cleaned JSON string
    """
    cleaned_content = raw_content
    
    # Remove Kimi thinking tokens if present
    cleaned_content = cleaned_content.replace("◁think▷", "").replace("◁/think▷", "")
    
   # Remove markdown code block wrapper
    cleaned_content = cleaned_content.strip()
    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:]
    
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-3]
    
    return cleaned_content.strip()


def parse_json_with_repair(raw_content: str) -> Dict[str, Any]:
    """
    Parse JSON with automatic repair for malformed responses.
    
    Args:
        raw_content: Raw JSON string
        
    Returns:
        Parsed JSON object
    """
    cleaned_content = clean_json_response(raw_content)
    
    # Check for invalid JSON pattern (key without value)
    invalid_json_pattern = r'^\{\s*"[^"]+"\s*\}$'
    import re
    if re.match(invalid_json_pattern, cleaned_content):
        log.warning('Detected invalid JSON structure (key without value)')
        key_match = re.search(r'"([^"]+)"', cleaned_content)
        if key_match:
            return {
                "document_title": key_match.group(1),
                "note": "Model returned only a title/key without structured data"
            }
        return {
            "error": "Invalid JSON structure",
            "raw_text": cleaned_content
        }
    
    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        log.error(f'Error parsing JSON: {e}')
        log.error(f'Problematic content: {cleaned_content[:500]}')
        
        # Try json repair as fallback
        try:
            # Simple repair: try to fix common issues
            repaired = cleaned_content
            # Add missing closing braces/brackets
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            repaired += '}' * open_braces
            repaired += ']' * open_brackets
            
            return json.loads(repaired)
        except Exception as repair_error:
            log.error(f'Error repairing JSON: {repair_error}')
            return {
                "error": "Failed to parse JSON response",
                "error_message": str(repair_error),
                "raw_content": raw_content,
                "note": "Response is not valid JSON and could not be repaired"
            }


async def _docx_text_to_structured_json(
    text: str,
    model: Any,
    model_provider: Optional[str] = None,
    additional_document_instructions: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert DOCX plain text into structured JSON via the process model.

    Mimics what Step 1 (vision) does for images, but operates on text input.
    The LLM organizes the raw text into a structured JSON with labels,
    sections, tables, etc.
    """
    if not text or not text.strip():
        return {
            "raw_text": "",
            "note": "DOCX appears empty or contains no extractable text.",
        }

    log.info("DOCX Step 1: Converting raw text to structured JSON via LLM...")

    messages = [
        SystemMessage(
            content=(
                "You are a document analysis assistant. Your task is to take raw text extracted from a DOCX document "
                "and organize it into a structured JSON format. Be comprehensive and include all details. "
                "Return the information in a structured JSON format with clear categories."
            )
        ),
        HumanMessage(
            content=f"""Please organize the following document text into a comprehensive structured JSON.
Include all labels, headers, descriptions, title, text, numbers, dates, tables, forms, and any other information present.
Organize the data in a comprehensive JSON structure matching the original document layout.
Notes:
- Try best and pay most attention to extract numerical data accurately.
- For numerical data or id, please ensure accuracy and clarity.
- Only do raw organization and do not try to interpret or format the data.
- Do exhaustive extraction from the text from top to bottom.
- Do not alter, interpret or format the data, just organize as is.
- Ignore empty sections or fields with no content.
Additional Instructions: {additional_document_instructions or "None"}

Document Text:
{text}
"""
        ),
    ]

    response = await invoke_with_structured_output(model, messages, model_provider)
    raw_content = _extract_content_text(response)
    return parse_json_with_repair(raw_content)


async def _run_step2_schema_mapping(
    extracted_data: Any,
    output_schema: Dict[str, Any],
    model: Any,
    model_provider: Optional[str],
    language: Optional[str],
    additional_instructions: Optional[str],
    process_enable_thinking: bool,
    process_max_tokens: Optional[int],
    extract_raw_data: bool,
) -> Any:
    """Run Step 2 (schema mapping) on previously-extracted document data.

    Shared by vision_assistant_v2, kimi_assistant_v2, and the DOCX text path.
    """
    log.info("Step 2: Processing extracted data with schema and instructions...")
    extracted_data_count = len(extracted_data) if isinstance(extracted_data, list) else 1
    log.info(f"Extracted Data Total Numbers: {extracted_data_count}")

    lang_instruction = f" Please respond in {language}." if language else ""

    raw_data_instruction = ""
    if extract_raw_data:
        raw_data_instruction = (
            "\nIMPORTANT: For each field in the output, preserve the bounding box and confidence score from the extracted data. "
            "Each field value should be wrapped in an object with 'value', 'box' (bounding box coordinates [x1, y1, x2, y2]), and 'confidence' (0-1 score) properties. "
            "If bounding box or confidence data is not available for a field, omit those properties and just use the plain value."
        )

    # For vLLM strict json_schema, the output shape is enforced by guided decoding, so the
    # (large) board schema is NOT injected into the prompt — keeps Step 2 lean and fast.
    vllm_structured = (
        model_provider == "vllm" and bool(output_schema) and _vllm_structured_output_enabled()
    )
    if vllm_structured:
        schema_block = "Return the data conforming exactly to the required output structure.\n"
    else:
        schema_block = (
            f"Required Output Schema: {json.dumps(output_schema)}\n\n"
            "Please return a JSON object that follows the schema exactly and incorporates any relevant data from the extracted information.\n"
            "Please analyze the extracted data thoroughly and ensure that all relevant information is included in the final JSON response based on given json outputschema. You should read description in each field of the schema to determine which data from the extracted data is relevant to the schema.\n"
        )

    step2_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=f"You are a data processing assistant. Your task is to take previously extracted document data and format it according to a specific schema and additional instructions. Return a JSON object that strictly follows the provided schema.{lang_instruction}"
        ),
        HumanMessage(
            content=f"""Using the extracted data below, please create a JSON response that follows the specified schema and incorporates the additional instructions.{lang_instruction}

Extracted Data: {json.dumps(extracted_data, indent=2)}
Extracted Data Total Numbers: {extracted_data_count}

Additional Instructions: {additional_instructions or "None"}

{schema_block}Do not alter or interpret extracted data in any way, just use it to fill the schema.
Please return an array of JSON objects if Extracted Data Total Numbers is greater than 1, otherwise return a single JSON object.
{raw_data_instruction}
"""
        ),
    ])

    step2_messages = step2_prompt.format_messages()

    # For Bedrock native structured output the json_schema root must be an object. When we
    # expect multiple records, wrap the schema in a {"records": [...]} envelope and unwrap
    # the result afterwards. (Other providers ignore output_schema entirely.)
    expects_array = extracted_data_count > 1
    structured_schema = _wrap_schema_for_array(output_schema) if expects_array else output_schema

    step2_response = await invoke_with_structured_output(
        model, step2_messages, model_provider,
        enable_thinking=process_enable_thinking, max_tokens=process_max_tokens,
        output_schema=structured_schema, schema_name="document_step2_output",
    )

    final_content_raw = _extract_content_text(step2_response)
    final_content = parse_json_with_repair(final_content_raw)

    # Unwrap the {"records": [...]} envelope used for Bedrock structured output.
    if expects_array and isinstance(final_content, dict) and len(final_content) == 1:
        only_value = list(final_content.values())[0]
        if isinstance(only_value, list):
            final_content = only_value

    log.info("Step 2 formatting complete")
    return final_content


async def vision_assistant_v2(
    model,
    visionModel,
    url: str,
    language: Optional[str],
    additional_instructions: Optional[str],
    additional_document_instructions: Optional[str],
    output_schema: Optional[Dict[str, Any]],
    model_provider: Optional[str] = None,
    process_enable_thinking: bool = False,
    process_max_tokens: Optional[int] = None,
    extract_raw_data: bool = False,
    vision_provider: Optional[str] = None,
    enable_ocr_confidence: bool = False,
    ocr_engine: Optional[str] = None,
    confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Two-step vision assistant for document processing (V2).
    
    Step 1: Extract all data from document using vision model
    Step 2: Format extracted data according to output schema
    
    Args:
        model: LLM for step 2 formatting
        visionModel: Vision-capable LLM for step 1 extraction
        url: URL of document to process
        language: Optional language for response
        additional_instructions: Instructions for step 2 formatting
        additional_document_instructions: Instructions for step 1 extraction
        output_schema: Target schema for formatted output
        
    Returns:
        Formatted document data
    """
    # Fetch and convert document
    file_extension, base64_string = await fetch_url_as_base64_string(url)

    if file_extension == "pdf":
        file_extension = "png"
        base64_string = await pdf_base64_to_png_base64(base64_string)
    elif file_extension == "docx":
        log.info("vision_assistant_v2: Detected DOCX, extracting text via docx2txt (skipping vision step)")
        docx_text = await docx_base64_to_text(base64_string)
        if len(docx_text) > 200_000:
            log.warning(f"DOCX truncated from {len(docx_text)} to 200000 chars for Step 2 context window")
            docx_text = docx_text[:200_000]

        # Use the process model (or vision model as fallback) to structure the text
        step1_model = model or visionModel
        extracted_data = await _docx_text_to_structured_json(
            docx_text, step1_model, model_provider, additional_document_instructions,
        )

        log.info(f"DOCX Step 1 complete. Data keys: {list(extracted_data.keys()) if isinstance(extracted_data, dict) else len(extracted_data)}")

        if not output_schema:
            log.info("No outputSchema provided, returning structured data from DOCX")
            return extracted_data

        if extract_raw_data:
            log.info("extract_raw_data=True with DOCX: bounding boxes are not available for text-only extraction")

        return await _run_step2_schema_mapping(
            extracted_data=extracted_data,
            output_schema=output_schema,
            model=model,
            model_provider=model_provider,
            language=language,
            additional_instructions=additional_instructions,
            process_enable_thinking=process_enable_thinking,
            process_max_tokens=process_max_tokens,
            extract_raw_data=extract_raw_data,
        )

    log.info(f"additionalDocumentInstructions: {additional_document_instructions}")

    # Step 1: Extract all data from the image
    step1_messages = [
        SystemMessage(
            content="You are a document analysis assistant. Your task is to extract all visible text, data, and information from the provided image. Be comprehensive and include all details you can recognize. Return the information in a structured JSON format with clear categories."
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"""Please extract thoroughly all data from this document image. Include all labels, headers, descriptions, title, text, numbers, dates, tables, forms, and any other information you can recognize, including handwriting. Organize the extracted data in a comprehensive JSON structure matching the original document layout.
Notes:
Try best and pay most attention to extract numerical data or non-english letters (e.g.: chinese) if present as accurately as possible when doing OCR for handwriting because they can be hard to recognize.
For numerical data or id, please ensure accuracy and clarity and base on its label to detect the best result possible
Only do raw OCR and do not try to interpret or format the data.
Do exhaustive extraction with the document from top to bottom and left to right.
Do not alter, interpret or format the data, especially handwriting, just extract as is.
{"Extract all content including borders, boxed borders, and their visual structure as-is from the original document." if extract_raw_data else "Ignore border or boxed border, just extract the content inside the border. If box has not content, ignore it, do not make up values."}
Ignore blurry, illegible, or unreadable text, do not make up values.
Additional Instructions: {additional_document_instructions or "None"}
""",
                },
                *(
                    [
                        _format_image_content(item, file_extension, vision_provider)
                        for item in base64_string
                    ]
                    if isinstance(base64_string, list)
                    else [
                        _format_image_content(base64_string, file_extension, vision_provider)
                    ]
                ),
            ]
        ),
    ]

    step1_prompt = ChatPromptTemplate.from_messages(step1_messages)

    # Fire real-OCR concurrently with the Step-1 vision call so it overlaps the
    # network-bound LLM (OCR offloads to a worker thread). Reuses the same page
    # image(s) already rendered above — no extra render/download.
    ocr_on = enable_ocr_confidence or extract_raw_data
    ocr_task = asyncio.create_task(_run_ocr(base64_string, ocr_engine)) if ocr_on else None

    log.info("Step 1: Extracting all data from image...")
    # Don't bind response_format for Google models - they don't support it
    # The prompt itself instructs the model to return JSON
    # Set max_output_tokens for Gemini to prevent repetition loops from producing unbounded output
    if vision_provider in ("gemini", "google") and not getattr(visionModel, 'max_output_tokens', None):
        visionModel.max_output_tokens = 65536
    step1_chain = step1_prompt | visionModel
    step1_response = await step1_chain.ainvoke({})
    
    # Parse the JSON response
    raw_content = _extract_content_text(step1_response)
    
    # Handle array of content objects (Gemini can return multiple parts)
    if isinstance(raw_content, list):
        text_part = next((part.get("text") for part in raw_content if part.get("type") == "text" and "{" in part.get("text", "")), None)
        if text_part:
            raw_content = text_part
        elif raw_content:
            raw_content = raw_content[0].get("text", "") if isinstance(raw_content[0], dict) else str(raw_content[0])
    
    extracted_data = parse_json_with_repair(raw_content)
    
    log.info(f"Step 1 extraction complete. Data count: {len(extracted_data) if isinstance(extracted_data, list) else 1}")

    # If no output schema, return extracted data
    if not output_schema:
        log.info("No outputSchema provided, returning extracted data from step 1")
        final = extracted_data
    else:
        final = await _run_step2_schema_mapping(
            extracted_data=extracted_data,
            output_schema=output_schema,
            model=model,
            model_provider=model_provider,
            language=language,
            additional_instructions=additional_instructions,
            process_enable_thinking=process_enable_thinking,
            process_max_tokens=process_max_tokens,
            extract_raw_data=extract_raw_data,
        )

    if ocr_task is not None:
        ocr_tokens = await ocr_task
        final = await _ground_confidence_async(final, ocr_tokens, confidence_threshold)

    return final


async def kimi_assistant_v2(
    modelName: str,
    model,
    visionModel,
    url: str,
    language: Optional[str],
    additional_instructions: Optional[str],
    additional_document_instructions: Optional[str],
    output_schema: Optional[Dict[str, Any]],
    model_provider: Optional[str] = None,
    process_enable_thinking: bool = False,
    process_max_tokens: Optional[int] = None,
    extract_raw_data: bool = False,
    enable_ocr_confidence: bool = False,
    ocr_engine: Optional[str] = None,
    confidence_threshold: float = 0.0,
) -> Dict[str, Any]:
    """
    Two-step Kimi vision assistant for document processing (V2).
    
    Uses Kimi's vision model for OCR and extraction with special handling
    for thinking tokens and image compression.
    
    Args:
        modelName: Name of the Kimi model
        model: LLM for step 2 formatting
        visionModel: Kimi vision model for step 1 extraction
        url: URL of document to process
        language: Optional language for response
        additional_instructions: Instructions for step 2 formatting
        additional_document_instructions: Instructions for step 1 extraction
        output_schema: Target schema for formatted output
        
    Returns:
        Formatted document data
    """
    from open_webui.config import KIMI_CONFIG, OLM_CONFIG

    # Fetch and convert document
    file_extension, base64_string = await fetch_url_as_base64_string(url)

    if file_extension == "pdf":
        file_extension = "png"
        base64_string = await pdf_base64_to_png_base64(base64_string)
    elif file_extension == "docx":
        log.info("kimi_assistant_v2: Detected DOCX, extracting text via docx2txt (skipping vision step)")
        docx_text = await docx_base64_to_text(base64_string)
        if len(docx_text) > 200_000:
            log.warning(f"DOCX truncated from {len(docx_text)} to 200000 chars for Step 2 context window")
            docx_text = docx_text[:200_000]

        step1_model = model or visionModel
        extracted_data = await _docx_text_to_structured_json(
            docx_text, step1_model, model_provider, additional_document_instructions,
        )

        log.info(f"DOCX Step 1 complete. Data keys: {list(extracted_data.keys()) if isinstance(extracted_data, dict) else len(extracted_data)}")

        if not output_schema:
            log.info("No outputSchema provided, returning structured data from DOCX")
            return extracted_data

        if extract_raw_data:
            log.info("extract_raw_data=True with DOCX: bounding boxes are not available for text-only extraction")

        return await _run_step2_schema_mapping(
            extracted_data=extracted_data,
            output_schema=output_schema,
            model=model,
            model_provider=model_provider,
            language=language,
            additional_instructions=additional_instructions,
            process_enable_thinking=process_enable_thinking,
            process_max_tokens=process_max_tokens,
            extract_raw_data=extract_raw_data,
        )

    # TODO: Add image resizing/compression if needed
    compressed_base64_string = base64_string

    # Fire real-OCR concurrently with the Step-1 Kimi/OLM vision call.
    ocr_on = enable_ocr_confidence or extract_raw_data
    ocr_task = asyncio.create_task(_run_ocr(compressed_base64_string, ocr_engine)) if ocr_on else None

    log.info(f"additionalDocumentInstructions: {additional_document_instructions}")

    # Step 1: Extract using Kimi/OLM API directly
    step1_messages = [
        {
            "role": "system",
            "content": "You are a document analysis assistant. Your task is to extract all visible text, data, and information from the provided image. Be comprehensive and include all details you can recognize. Return the information in a structured JSON format with clear categories."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Please extract thoroughly all data from this document image. Include all labels, headers, descriptions, title, text, numbers, dates, tables, forms, and any other information you can recognize, including handwriting. Organize the extracted data in a comprehensive JSON structure matching the original document layout.
Notes:
Try best and pay most attention to extract numerical data or non-english letters (e.g.: chinese) if present as accurately as possible when doing OCR for handwriting because they can be hard to recognize.
For numerical data or id, please ensure accuracy and clarity and base on its label to detect the best result possible
Only do raw OCR and do not try to interpret or format the data.
Do exhaustive extraction with the document from top to bottom and left to right.
Do not alter, interpret or format the data, especially handwriting, just extract as is.
{"Extract all content including borders, boxed borders, and their visual structure as-is from the original document." if extract_raw_data else "Ignore border or boxed border, just extract the content inside the border. If box has not content, ignore it, do not make up values."}
Ignore blurry, illegible, or unreadable text, do not make up values.
Additional Instructions: {additional_document_instructions or "None"}
""",
                },
                *(
                    [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{item}",
                            },
                        }
                        for item in compressed_base64_string
                    ]
                    if isinstance(compressed_base64_string, list)
                    else [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{compressed_base64_string}",
                            },
                        }
                    ]
                ),
            ],
        }
    ]
    
    log.info(f"Step 1: Extracting all data from image using {modelName}...")
    
    # Determine which model to use
    model_to_use = KIMI_CONFIG["model"] if modelName == "Kimi-VL-A3B-Thinking-2506" else OLM_CONFIG["model"]
    log.info(f"Using model: {model_to_use}")
    step1_response = await visionModel.chat.completions.create(
        model=model_to_use,
        messages=step1_messages,
        temperature=0,
        top_p=0.1,
        response_format={"type": "json_object"},
    )
    
    # Parse response
    raw_content = step1_response.choices[0].message.content if hasattr(step1_response, 'choices') else step1_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    extracted_data = parse_json_with_repair(raw_content)
    
    log.info(f"Step 1 extraction complete")

    # If no output schema, return extracted data
    if not output_schema:
        log.info("No outputSchema provided, returning extracted data from step 1")
        final = extracted_data
    else:
        final = await _run_step2_schema_mapping(
            extracted_data=extracted_data,
            output_schema=output_schema,
            model=model,
            model_provider=model_provider,
            language=language,
            additional_instructions=additional_instructions,
            process_enable_thinking=process_enable_thinking,
            process_max_tokens=process_max_tokens,
            extract_raw_data=extract_raw_data,
        )

    if ocr_task is not None:
        ocr_tokens = await ocr_task
        final = await _ground_confidence_async(final, ocr_tokens, confidence_threshold)

    return final


# V3 Functions with enhanced PDF processing

async def split_pdf_into_chunks(
    base64_string: str,
    output_schema: Dict[str, Any],
    options: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Splits a base64 encoded PDF into chunks of images."""
    chunk_size = options.get("chunkSize", 5)
    
    log.info("Converting PDF pages to images for chunking...")
    # Use existing function asynchronously
    all_pages = await pdf_base64_to_png_base64(base64_string, page_count=None)
    
    total_pages = len(all_pages)
    log.info(f"PDF has {total_pages} pages.")
    
    chunks = []
    num_chunks = math.ceil(total_pages / chunk_size)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_pages)
        chunk_pages = all_pages[start_idx:end_idx]
        
        chunks.append({
            "chunkId": i,
            "startPage": start_idx + 1,
            "endPage": end_idx,
            "pages": chunk_pages,
            "totalChunks": num_chunks
        })
        
    log.info(f"Split PDF into {len(chunks)} chunks.")
    return chunks

async def resize_image(base64_img: str, max_dimension: int = 2000) -> str:
    """Resizes a base64 image if it exceeds max_dimension."""
    try:
        def _resize():
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data))
            width, height = img.size
            if width > max_dimension or height > max_dimension:
                ratio = min(max_dimension / width, max_dimension / height)
                new_size = (int(width * ratio), int(height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            return base64_img
        return await asyncio.to_thread(_resize)
    except Exception as e:
        log.error(f"Error resizing image: {e}")
        return base64_img

def clean_and_extract_json(response: Any, model_provider: Optional[str] = None) -> str:
    """Clean and extract JSON from AI model response."""
    response_text: Any = ""

    if hasattr(response, 'content'):
        response_text = response.content
    elif hasattr(response, 'text'):
        response_text = response.text
    elif isinstance(response, str):
        response_text = response
    else:
        response_text = json.dumps(response)

    # Some LangChain providers (e.g. multimodal/structured outputs) return
    # `content` as a list of content blocks like [{"type":"text","text":"..."}]
    # instead of a plain string. Flatten to text.
    if isinstance(response_text, list):
        response_text = "".join(
            (b.get("text", "") if isinstance(b, dict) else str(b))
            for b in response_text
        )
    elif not isinstance(response_text, str):
        response_text = str(response_text)

    if not response_text or not response_text.strip():
        log.error("Empty response received from model")
        return '{"error": "Empty response from model"}'
        
    # Remove markdown code blocks
    response_text = re.sub(r'^```json\s*', '', response_text, flags=re.IGNORECASE)
    response_text = re.sub(r'^```\s*', '', response_text, flags=re.IGNORECASE)
    response_text = re.sub(r'\s*```$', '', response_text, flags=re.IGNORECASE)
    response_text = response_text.strip()
    
    # Kimi specific cleaning
    if model_provider == 'kimi':
        response_text = re.sub(r'◁think▷.*?◁/think▷', '', response_text, flags=re.DOTALL)
        
    # Remove remaining backticks
    if '```' in response_text:
        response_text = re.sub(r'```[\s\S]*?```', '', response_text)
        
    return response_text

async def detect_mapping_key(
    extracted_data: Any,
    process_model: Any,
    additional_instructions: Optional[str],
    model_provider: Optional[str] = None
) -> str:
    """Detect mapping key using AI."""
    try:
        system_content = additional_instructions or (
            "You are a document analysis expert. Analyze the extracted data and determine the most appropriate mapping key "
            "that can uniquely identify individual documents or records.\n\n"
            "Common keys: RTV No, Invoice No, Document No, Reference No, Case No, Order No, Receipt No, Transaction ID, Serial Number, ID.\n"
            "Return ONLY the mapping key name. If no clear key found, return 'Document ID'."
        )

        user_content = f"Analyze this data and determine the best mapping key:\n\n{json.dumps(extracted_data, indent=2)}\n\nReturn only the mapping key name."

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content)
        ]

        log.info(f"detect_mapping_key: model_provider={model_provider}, model_type={type(process_model).__name__}")
        if model_provider == "vllm":
            try:
                raw = await vllm_http_invoke(process_model, messages, json_mode=False, timeout=900)
                mapping_key = re.sub(r'^["\']|["\']$', '', raw.strip()).strip()
                log.info(f"Detected mapping key (vLLM HTTP): {mapping_key}")
                return mapping_key
            except Exception as e:
                log.warning(f"vLLM direct HTTP failed for detect_mapping_key: {type(e).__name__}: {e}, falling back to LangChain")

        response = await process_model.ainvoke(messages)
        mapping_key = clean_and_extract_json(response, 'default')
        mapping_key = re.sub(r'^["\']|["\']$', '', mapping_key).strip()
        
        log.info(f"Detected mapping key: {mapping_key}")
        return mapping_key
    except Exception as e:
        log.error(f"Error detecting mapping key: {e}")
        return 'Document ID'

async def process_batch_with_ai(
    grouped_data: Dict[str, Any],
    mapping_key: str,
    output_schema: Dict[str, Any],
    ai_model: Any,
    file_url: str,
    additional_document_instructions: Optional[str],
    model_provider: Optional[str] = None,
    process_enable_thinking: bool = False,
    process_max_tokens: Optional[int] = None,
    extract_raw_data: bool = False,
    language: Optional[str] = None,
) -> List[Any]:
    """Process a batch of documents with AI."""
    lang_instruction = f" Respond in {language}." if language else ""
    system_content = (additional_document_instructions or (
        f"You are a data processing assistant. Take aggregated document data locally grouped by '{mapping_key}' "
        "and format it according to a specific schema.\n\n"
        "CRITICAL: Return ACTUAL DATA VALUES, not schema definitions. Fill schema fields with extracted content."
    )) + lang_instruction

    raw_data_instruction = ""
    if extract_raw_data:
        raw_data_instruction = (
            "5. IMPORTANT: For each field, preserve the bounding box and confidence score from the extracted data. "
            "Each field value should be wrapped in an object with 'value', 'box' (bounding box coordinates [x1, y1, x2, y2]), and 'confidence' (0-1 score) properties. "
            "If bounding box or confidence data is not available for a field, omit those properties and just use the plain value.\n"
        )

    # For vLLM strict json_schema, the shape is enforced by guided decoding, so skip the
    # (large) board schema in the prompt.
    vllm_structured = (
        model_provider == "vllm" and bool(output_schema) and _vllm_structured_output_enabled()
    )
    schema_block = "" if vllm_structured else f"Required Output Schema: {json.dumps(output_schema)}\n\n"

    user_content = (
        f"Process this grouped data and fill schema with ACTUAL DATA:\n\n"
        f"Grouped Data by {mapping_key}: {json.dumps(grouped_data, indent=2)}\n"
        f"Mapping Key: {mapping_key}\n"
        f"Number of unique documents: {len(grouped_data)}\n"
        f"File URL: {file_url or 'Not provided'}\n\n"
        f"{schema_block}"
        "CRITICAL REQUIREMENTS:\n"
        f"1. Process ALL {len(grouped_data)} unique documents.\n"
        "2. Return JSON array with ACTUAL DATA VALUES.\n"
        "3. Fill each schema field with corresponding data.\n"
        "4. Copy values VERBATIM from the extracted data. Do NOT translate or transliterate — "
        "preserve the original language and characters exactly (e.g., keep Chinese as Chinese).\n"
        f"{raw_data_instruction}"
    )
    
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ]
    
    # Batch processing always returns an array; wrap the schema in a {"records": [...]}
    # envelope for Bedrock native structured output. The existing single-key unwrap below
    # handles the {"records": [...]} response shape.
    response = await invoke_with_structured_output(
        ai_model, messages, model_provider,
        enable_thinking=process_enable_thinking, max_tokens=process_max_tokens,
        output_schema=_wrap_schema_for_array(output_schema), schema_name="document_batch_output",
    )

    response_text = clean_and_extract_json(response)
    # Using existing parser
    final_result = parse_json_with_repair(response_text)
    
    if isinstance(final_result, dict):
        # Check if wrapped in a key or is single object
        if len(final_result) == 1 and isinstance(list(final_result.values())[0], list):
             final_result = list(final_result.values())[0]
        else:
             final_result = [final_result]
             
    if not isinstance(final_result, list):
        final_result = [final_result]
        
    return final_result

async def aggregate_chunk_results(
    chunk_results: List[Dict[str, Any]],
    mapping_key: str,
    output_schema: Dict[str, Any],
    process_model: Any,
    additional_document_instructions: Optional[str],
    file_url: str,
    model_provider: Optional[str] = None,
    process_enable_thinking: bool = False,
    process_max_tokens: Optional[int] = None,
    extract_raw_data: bool = False,
    max_batch_size: int = 5,
    language: Optional[str] = None,
) -> Union[List[Any], Dict[str, Any]]:
    """Aggregates results from chunks, supports batching."""
    successful_results = [r for r in chunk_results if r["success"]]
    if not successful_results:
         raise ValueError("No successful chunk results to aggregate")
         
    all_extracted_data = []
    for chunk_res in successful_results:
        res = chunk_res["result"]
        chunk_id = chunk_res.get("chunkId", "?")
        log.debug(f"Aggregating chunk {chunk_id} result - type: {type(res).__name__}, "
                  f"preview: {json.dumps(res, default=str)[:200]}")
        if isinstance(res, list):
            all_extracted_data.extend(res)
        elif isinstance(res, dict):
            # Only unwrap when the dict is clearly an envelope like {"documents": [...]}
            if len(res) == 1:
                sole_value = next(iter(res.values()))
                if isinstance(sole_value, list):
                    all_extracted_data.extend(sole_value)
                else:
                    all_extracted_data.append(res)
            else:
                all_extracted_data.append(res)
        elif isinstance(res, str):
            log.warning(f"Chunk {chunk_id} returned string instead of dict/list, wrapping it. Value: {res[:200]}")
            all_extracted_data.append({"raw_value": res})
                
    log.info(f"Aggregating {len(all_extracted_data)} items from {len(successful_results)} chunks")
    # Log type distribution of aggregated items
    type_dist = {}
    for item in all_extracted_data:
        t = type(item).__name__
        type_dist[t] = type_dist.get(t, 0) + 1
    log.info(f"Aggregated item type distribution: {type_dist}")
    
    if not output_schema:
        return all_extracted_data

    # Group data
    grouped_data = {}
    for item in all_extracted_data:
        # Skip non-dict items (e.g., strings returned by AI)
        if not isinstance(item, dict):
            log.warning(f"Skipping non-dict item in extracted data: {type(item)}")
            if "default" not in grouped_data: grouped_data["default"] = []
            grouped_data["default"].append({"raw_value": item})
            continue
        # Heuristic search for key
        key_value = item.get(mapping_key) or item.get(mapping_key.lower()) or \
                    item.get(f"{mapping_key.lower()}_no") or item.get("rtv_no") or \
                    item.get("invoice_no") or item.get("document_no") or item.get("id")
                    
        if key_value:
            k = str(key_value)
            if k not in grouped_data: grouped_data[k] = []
            grouped_data[k].append(item)
        else:
            if "default" not in grouped_data: grouped_data["default"] = []
            grouped_data["default"].append(item)
            
    unique_docs = list(grouped_data.keys())
    log.info(f"Found {len(unique_docs)} unique documents")
    
    # Batch processing — run batches concurrently (Fix 2)
    max_concurrent_batches = 5
    all_results = []

    if len(unique_docs) > max_batch_size:
        log.info(f"Processing in batches (size {max_batch_size}, max_concurrent={max_concurrent_batches})...")
        batch_semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def _process_batch(batch_idx, batch_data):
            async with batch_semaphore:
                log.info(f"Processing batch {batch_idx + 1}")
                return await process_batch_with_ai(
                    batch_data, mapping_key, output_schema, process_model,
                    file_url, additional_document_instructions, model_provider,
                    process_enable_thinking=process_enable_thinking, process_max_tokens=process_max_tokens,
                    extract_raw_data=extract_raw_data, language=language
                )

        batch_tasks = []
        for i in range(0, len(unique_docs), max_batch_size):
            batch_keys = unique_docs[i : i + max_batch_size]
            batch_data = {k: grouped_data[k] for k in batch_keys}
            batch_tasks.append(_process_batch(i // max_batch_size, batch_data))

        batch_results = await asyncio.gather(*batch_tasks)
        for batch_result in batch_results:
            all_results.extend(batch_result)
    else:
        all_results = await process_batch_with_ai(
            grouped_data, mapping_key, output_schema, process_model,
            file_url, additional_document_instructions, model_provider,
            process_enable_thinking=process_enable_thinking, process_max_tokens=process_max_tokens,
            extract_raw_data=extract_raw_data, language=language
        )
        
    return all_results


async def process_large_pdf(
    base64_string: str,
    vision_model: Any,
    process_model: Any,
    output_schema: Dict[str, Any],
    additional_instructions: Optional[str],
    additional_document_instructions: Optional[str],
    language: Optional[str],
    file_url: str,
    options: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a large PDF by chunking and concurrent processing."""
    chunk_size = options.get("chunkSize", 5)
    max_concurrent = options.get("maxConcurrent", 3)
    model_provider = options.get("modelProvider", "kimi")
    process_provider = options.get("processProvider", model_provider)
    max_retries = options.get("maxRetries", 1)
    vision_enable_thinking = options.get("visionEnableThinking", False)
    vision_max_tokens = options.get("visionMaxTokens", None)
    process_enable_thinking = options.get("processEnableThinking", False)
    process_max_tokens = options.get("processMaxTokens", None)
    extract_raw_data = options.get("extractRawData", False)
    max_batch_size = options.get("maxBatchSize", 5)
    enable_ocr_confidence = bool(options.get("enableOcrConfidence") or extract_raw_data)
    ocr_engine_name = options.get("ocrEngine")
    confidence_threshold = options.get("confidenceThreshold") or 0.0

    try:
        log.info(f"Starting large PDF processing (chunk_size={chunk_size}, max_concurrent={max_concurrent})...")
        
        # Step 1: Split
        chunks = await split_pdf_into_chunks(base64_string, output_schema, options)
        
        # Step 2: Process Chunks
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_chunk(chunk, retry_count=0):
            async with semaphore:
                chunk_id = chunk["chunkId"]
                log.info(f"Processing chunk {chunk_id + 1}/{len(chunks)} page {chunk['startPage']}-{chunk['endPage']}")

                chunk_ocr_task = None
                try:
                    processed_pages = chunk["pages"]

                    if model_provider == "kimi":
                        resized_tasks = [resize_image(page) for page in processed_pages]
                        processed_pages = await asyncio.gather(*resized_tasks)

                    # Fire real-OCR concurrently with this chunk's vision call, over the
                    # same page images shown to the model.
                    chunk_page_numbers = list(range(chunk["startPage"], chunk["endPage"] + 1))
                    if enable_ocr_confidence:
                        chunk_ocr_task = asyncio.create_task(
                            _run_ocr(processed_pages, ocr_engine_name, chunk_page_numbers)
                        )

                    system_content = (
                        "You are a document analysis assistant. Your task is to extract all visible text, data, and information from the provided image. "
                        "Be comprehensive and include all details you can recognize. Return the information in a structured JSON format with clear categories."
                    )

                    border_instruction = (
                        "Extract all content including borders, boxed borders, and their visual structure as-is from the original document."
                        if extract_raw_data
                        else "Ignore border or boxed border, just extract the content inside the border. If box has not content, ignore it, do not make up values."
                    )
                    user_text = (
                        f"Please extract thoroughly all data from pages {chunk['startPage']}-{chunk['endPage']} (chunk {chunk_id + 1}). "
                        "Include all labels, headers, descriptions, title, text, numbers, dates, tables, forms, and any other information you can recognize, including handwriting. "
                        "Organize the extracted data in a comprehensive JSON structure matching the original document layout.\n"
                        "Notes:\n"
                        "Try best and pay most attention to extract numerical data or non-english letters (e.g.: chinese) if present as accurately as possible when doing OCR for handwriting because they can be hard to recognize.\n"
                        "For numerical data or id, please ensure accuracy and clarity and base on its label to detect the best result possible\n"
                        "Only do raw OCR and do not try to interpret or format the data.\n"
                        "Do exhaustive extraction with the document from top to bottom and left to right.\n"
                        "Do not alter, interpret or format the data, especially handwriting, just extract as is.\n"
                        f"{border_instruction}\n"
                        "Ignore blurry, illegible, or unreadable text, do not make up values.\n"
                        f"Additional Instructions: {additional_document_instructions or 'None'}"
                    )
                    
                    raw_response = ""
                    
                    if model_provider == "kimi":
                        
                        content_list = [{"type": "text", "text": user_text}]
                        for page in processed_pages:
                            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page}"}})
                        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": content_list}]
                        
                        from open_webui.config import KIMI_CONFIG
                        model_name = vision_model.model if hasattr(vision_model, 'model') else KIMI_CONFIG.get("model", "Kimi-VL-A3B-Thinking-2506")
                        
                        response = await vision_model.chat.completions.create(
                            model=model_name, messages=messages, temperature=0, top_p=0.1,
                            response_format={"type": "json_object"}, timeout=600
                        )
                        raw_response = response.choices[0].message.content
                    else:
                        image_parts = [
                            _format_image_content(page, "png", model_provider)
                            for page in processed_pages
                        ]
                        messages = [
                            SystemMessage(content=system_content),
                            HumanMessage(content=[
                                {"type": "text", "text": user_text},
                                *image_parts
                            ])
                        ]
                        # Debug: log model info and payload shape
                        log.info(f"Chunk {chunk_id} invoking model_provider={model_provider}, "
                                 f"model_type={type(vision_model).__name__}, "
                                 f"model_id={getattr(vision_model, 'model_id', getattr(vision_model, 'model', 'unknown'))}, "
                                 f"num_images={len(image_parts)}, "
                                 f"image_format={image_parts[0].get('type') if image_parts else 'none'}, "
                                 f"image_data_len={len(image_parts[0].get('source', {}).get('data', '') or image_parts[0].get('image_url', {}).get('url', '')) if image_parts else 0}")
                        # Step 1 is raw extraction; Step 2 emits the structured JSON. So when a
                        # board schema is set, don't impose (slow/unreliable) JSON mode on the
                        # vLLM Step-1 call — let it extract freely.
                        step1_json_mode = not (bool(output_schema) and model_provider == "vllm")
                        try:
                            response = await invoke_with_structured_output(vision_model, messages, model_provider, enable_thinking=vision_enable_thinking, max_tokens=vision_max_tokens, json_mode=step1_json_mode)
                        except Exception as invoke_err:
                            log.error(f"Chunk {chunk_id} invoke failed: {type(invoke_err).__name__}: {invoke_err}")
                            # Log full exception details for Bedrock
                            if hasattr(invoke_err, 'response'):
                                log.error(f"Chunk {chunk_id} response metadata: {invoke_err.response}")
                            if hasattr(invoke_err, '__cause__') and invoke_err.__cause__:
                                log.error(f"Chunk {chunk_id} caused by: {type(invoke_err.__cause__).__name__}: {invoke_err.__cause__}")
                            raise
                        raw_response = response.content
                        
                    log.debug(f"Chunk {chunk_id} raw_response type: {type(raw_response).__name__}, "
                             f"preview: {str(raw_response)[:300]}")
                    response_text = clean_and_extract_json(raw_response, model_provider)
                    log.debug(f"Chunk {chunk_id} cleaned response preview: {response_text[:300]}")
                    extracted_data = parse_json_with_repair(response_text)
                    log.info(f"Chunk {chunk_id} extraction result type: {type(extracted_data).__name__}, "
                             f"preview: {json.dumps(extracted_data, default=str)[:300]}")
                    if isinstance(extracted_data, list):
                        type_counts = {}
                        for el in extracted_data:
                            t = type(el).__name__
                            type_counts[t] = type_counts.get(t, 0) + 1
                        log.info(f"Chunk {chunk_id} list element types: {type_counts}")
                    chunk_tokens = await chunk_ocr_task if chunk_ocr_task is not None else []
                    return {"chunkId": chunk_id, "success": True, "result": extracted_data, "tokens": chunk_tokens}

                except Exception as e:
                    if chunk_ocr_task is not None:
                        chunk_ocr_task.cancel()
                    if retry_count < max_retries:
                        log.warning(f"Retrying chunk {chunk_id} (attempt {retry_count + 1}/{max_retries + 1}): {e}")
                        # Simple backoff handled in safe wrapper
                        raise e
                    else:
                        log.error(f"Error processing chunk {chunk_id} after {max_retries + 1} attempts: {e}")
                        return {"chunkId": chunk_id, "success": False, "error": str(e), "tokens": []}

        async def process_chunk_safe(chunk):
            chunk_id = chunk["chunkId"]
            current_try = 0
            while current_try <= max_retries:
                try:
                    return await process_single_chunk(chunk, current_try) 
                except Exception:
                    current_try += 1
                    if current_try <= max_retries:
                        await asyncio.sleep(1 * current_try)
            return {"chunkId": chunk_id, "success": False, "error": f"Failed after {max_retries + 1} attempts"}

        # Fix 1: Run detect_mapping_key on first completed chunk instead of waiting for all
        # Fix 3: Skip detect_mapping_key entirely for single-chunk PDFs
        if len(chunks) == 1:
            # Single chunk — no need for concurrent processing or mapping key detection
            result = await process_chunk_safe(chunks[0])
            if not result["success"]:
                raise RuntimeError(f"Single chunk failed: {result.get('error', 'Unknown')}")
            results = [result]
            mapping_key = await detect_mapping_key(result["result"], process_model, additional_instructions, process_provider) if output_schema else "Document ID"
        else:
            # Multiple chunks — start detect_mapping_key as soon as first chunk completes
            mapping_key_future = asyncio.get_event_loop().create_future()

            async def process_chunk_and_detect(chunk):
                result = await process_chunk_safe(chunk)
                # As soon as first successful chunk completes, kick off mapping key detection
                if result["success"] and not mapping_key_future.done() and output_schema:
                    async def _detect():
                        try:
                            key = await detect_mapping_key(result["result"], process_model, additional_instructions, process_provider)
                            if not mapping_key_future.done():
                                mapping_key_future.set_result(key)
                        except Exception as e:
                            if not mapping_key_future.done():
                                mapping_key_future.set_result("Document ID")
                            log.warning(f"Mapping key detection failed, using default: {e}")
                    asyncio.create_task(_detect())
                return result

            chunk_tasks = [process_chunk_and_detect(chunk) for chunk in chunks]
            results = await asyncio.gather(*chunk_tasks)

            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]

            if not successful:
                raise RuntimeError(f"All chunks failed. First error: {failed[0]['error'] if failed else 'Unknown'}")

            if output_schema:
                # Wait for mapping key (should already be done since chunks finished)
                if not mapping_key_future.done():
                    # Fallback: detect from first successful chunk
                    mapping_key = await detect_mapping_key(successful[0]["result"], process_model, additional_instructions, process_provider)
                else:
                    mapping_key = mapping_key_future.result()
            else:
                mapping_key = "Document ID"

        final = await aggregate_chunk_results(
            results, mapping_key, output_schema, process_model,
            additional_document_instructions, file_url, process_provider,
            process_enable_thinking=process_enable_thinking, process_max_tokens=process_max_tokens,
            extract_raw_data=extract_raw_data, max_batch_size=max_batch_size, language=language
        )

        if enable_ocr_confidence:
            all_tokens = []
            for r in results:
                all_tokens.extend(r.get("tokens") or [])
            log.info(f"Grounding aggregated result with {len(all_tokens)} OCR tokens "
                     f"from {len(results)} chunk(s)...")
            final = await _ground_confidence_async(final, all_tokens, confidence_threshold)

        return final

    except Exception as e:
        log.error(f"Error in large PDF processing: {e}")
        raise

async def vision_assistant_v3(
    model,
    visionModel,
    url: str,
    language: Optional[str],
    additional_instructions: Optional[str],
    additional_document_instructions: Optional[str],
    output_schema: Optional[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Enhanced vision assistant with chunked PDF processing (V3)."""
    options = options or {}
    try:
        file_extension, base64_string = await fetch_url_as_base64_string(url)
        log.info(f"vision_assistant_v3: file_extension={file_extension}, url={url[:100]}, "
                 f"output_schema type={type(output_schema).__name__}, options={options}")
        if file_extension == "pdf":
            log.info("Using enhanced PDF processing...")
            return await process_large_pdf(
                base64_string, visionModel, model, output_schema,
                additional_instructions, additional_document_instructions,
                language, url, options
            )
        else:
            return await vision_assistant_v2(
                model, visionModel, url, language,
                additional_instructions, additional_document_instructions,
                output_schema,
                model_provider=options.get("processProvider", options.get("modelProvider")),
                process_enable_thinking=options.get("processEnableThinking", False),
                process_max_tokens=options.get("processMaxTokens", None),
                extract_raw_data=options.get("extractRawData", False),
                vision_provider=options.get("modelProvider"),
                **_ocr_kwargs_from_options(options),
            )
    except Exception as e:
        log.error(f"Error in vision_assistant_v3: {e}")
        raise


async def kimi_assistant_v3(
    modelName: str,
    model,
    visionModel,
    url: str,
    language: Optional[str],
    additional_instructions: Optional[str],
    additional_document_instructions: Optional[str],
    output_schema: Optional[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced Kimi assistant with fallback strategies (V3).
    
    Tries enhanced processing first, then falls back to V2.
    
    Args:
        modelName: Name of the Kimi model
        model: LLM for step 2 formatting
        visionModel: Kimi vision model for step 1 extraction
        url: URL of document to process
        language: Optional language for response
        additional_instructions: Instructions for step 2 formatting
        additional_document_instructions: Instructions for step 1 extraction
        output_schema: Target schema for formatted output
        options: Processing options
        
    Returns:
        Formatted document data
    """
    options = options or {}
    
    try:
        file_extension, base64_string = await fetch_url_as_base64_string(url)
        
        if f"{file_extension}" == "pdf":
            # Try with enhanced processing first
            try:
                log.info('Attempting enhanced PDF processing with Kimi...')
                opts = options.copy()
                opts["chunkSize"] = opts.get("chunkSize", 3)
                opts["maxConcurrent"] = opts.get("maxConcurrent", 3)
                opts["modelProvider"] = "kimi"

                return await process_large_pdf(
                    base64_string, visionModel, model, output_schema,
                    additional_instructions, additional_document_instructions,
                    language, url, opts
                )
            except Exception as enhanced_error:
                log.error(f"Enhanced processing failed, trying fallback strategy: {enhanced_error}")
                
                # Fallback: Try with even smaller chunks and lower concurrency
                try:
                    log.info('Attempting fallback processing with smaller chunks...')
                    fallback_opts = options.copy()
                    fallback_opts["chunkSize"] = 1
                    fallback_opts["maxConcurrent"] = 1
                    fallback_opts["modelProvider"] = "kimi"
                    fallback_opts["maxRetries"] = 2
                    
                    return await process_large_pdf(
                        base64_string, visionModel, model, output_schema,
                        additional_instructions, additional_document_instructions,
                        language, url, fallback_opts
                    )
                except Exception as fallback_error:
                    log.error(f"Fallback processing also failed: {fallback_error}")
                    
                    # Final fallback: Use original kimi_assistant_v2 for the entire PDF
                    log.info('Attempting final fallback with original kimi_assistant_v2...')
                    return await kimi_assistant_v2(
                        modelName, model, visionModel, url, language,
                        additional_instructions, additional_document_instructions,
                        output_schema,
                        model_provider=options.get("processProvider", options.get("modelProvider")),
                        process_enable_thinking=options.get("processEnableThinking", False),
                        process_max_tokens=options.get("processMaxTokens", None),
                        extract_raw_data=options.get("extractRawData", False),
                        **_ocr_kwargs_from_options(options),
                    )
        else:
            return await kimi_assistant_v2(
                modelName, model, visionModel, url, language,
                additional_instructions, additional_document_instructions,
                output_schema,
                model_provider=options.get("processProvider", options.get("modelProvider")),
                process_enable_thinking=options.get("processEnableThinking", False),
                process_max_tokens=options.get("processMaxTokens", None),
                extract_raw_data=options.get("extractRawData", False),
                **_ocr_kwargs_from_options(options),
            )
    except Exception as e:
        log.error(f"Error in kimi_assistant_v3: {e}")
        raise


async def gemini_assistant_v3(
    model,
    visionModel,
    url: str,
    language: Optional[str],
    additional_instructions: Optional[str],
    additional_document_instructions: Optional[str],
    output_schema: Optional[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced Gemini assistant with chunked PDF processing (V3).
    
    Uses Gemini-specific settings for enhanced processing.
    
    Args:
        model: LLM for step 2 formatting
        visionModel: Gemini vision model for step 1 extraction
        url: URL of document to process
        language: Optional language for response
        additional_instructions: Instructions for step 2 formatting
        additional_document_instructions: Instructions for step 1 extraction
        output_schema: Target schema for formatted output
        options: Processing options
        
    Returns:
        Formatted document data
    """
    options = options or {}
    
    try:
        file_extension, base64_string = await fetch_url_as_base64_string(url)
        
        if file_extension == "pdf":
            # Use the enhanced large PDF processor with Gemini-specific settings
            log.info("Using enhanced PDF processing with Gemini...")
            opts = options.copy()
            # Optimize options for Gemini
            opts["chunkSize"] = opts.get("chunkSize", 10)
            opts["maxConcurrent"] = opts.get("maxConcurrent", 30)
            opts["modelProvider"] = "gemini" 
            
            return await process_large_pdf(
                base64_string, visionModel, model, output_schema,
                additional_instructions, additional_document_instructions,
                language, url, opts
            )
        else:
            # For non-PDF files, use the original visionAssistantV2 logic
            return await vision_assistant_v2(
                model, visionModel, url, language,
                additional_instructions, additional_document_instructions,
                output_schema,
                model_provider=options.get("processProvider", "gemini") if options else "gemini",
                process_enable_thinking=options.get("processEnableThinking", False) if options else False,
                process_max_tokens=options.get("processMaxTokens", None) if options else None,
                extract_raw_data=options.get("extractRawData", False) if options else False,
                vision_provider="google",
                **_ocr_kwargs_from_options(options or {}),
            )
    except Exception as e:
        log.error(f"Error in gemini_assistant_v3: {e}")
        raise
