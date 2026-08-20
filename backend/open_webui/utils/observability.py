"""
Structured logging + X-Request-Id correlation for chat-ai (Open WebUI fork).

Open WebUI logs through **loguru** (see utils/logger.py); a stdlib
InterceptHandler routes standard `logging` records into loguru too. So to emit
the canonical imbrace log schema we plug a loguru *format function* into the
stdout sink (mirroring loguru's own `file_format` pattern) — this upgrades every
`log.*()` / `logger.*()` call at once, no call-site changes.

Canonical single-line JSON schema (same as every other imbrace service):

    ip, request_id, date_time (ISO 8601 UTC), time (epoch ms), method_request,
    request_path, service_name, env, type_of_entity, function_of_code,
    description_message, response_time, status_code, proxy, level

Wiring:
  - utils/logger.py: the stdout sink uses ``format=loguru_canonical_format``.
  - main.py: ``add_request_context_middleware(app)`` mints/propagates the id and
    emits one access line per request.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from datetime import timezone
from typing import Any, Optional

from loguru import logger

REQUEST_ID_HEADER = "x-request-id"
PROXY_HEADER = "x-proxy"

SERVICE_NAME = os.environ.get("SERVICE_NAME") or "chat-ai"


def _normalize_env(raw: Optional[str]) -> str:
    v = (raw or "").lower()
    if "prod" in v:
        return "prodv2"
    if "stag" in v or v == "stg":
        return "staging"
    if "dev" in v or "local" in v:
        return "dev"
    return v or "dev"


ENV = os.environ.get("DEPLOY_ENV") or _normalize_env(
    os.environ.get("ENV") or os.environ.get("NODE_ENV")
)

# Map loguru/stdlib level names to the canonical tokens.
_LEVEL_MAP = {
    "DEBUG": "debug",
    "INFO": "info",
    "WARNING": "warn",
    "ERROR": "error",
    "CRITICAL": "error",
    "TRACE": "debug",
    "SUCCESS": "info",
}

# Per-request correlation context (Python's AsyncLocalStorage equivalent).
_request_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "imbrace_request_ctx", default=None
)


def get_context() -> Optional[dict]:
    return _request_ctx.get()


def set_context(ctx: dict) -> contextvars.Token:
    return _request_ctx.set(ctx)


def get_request_id() -> Optional[str]:
    ctx = _request_ctx.get()
    return ctx.get("request_id") if ctx else None


def enter_job_context(job_name: str, incoming_request_id: Optional[str] = None):
    """
    Bind a freshly-minted correlation context for a non-HTTP unit of work
    (background worker tick, cron job, queue consumer) so every log line it emits
    shares one request_id. Returns a token to pass to ``exit_job_context``.
    Pass ``incoming_request_id`` to continue a trace across an async hop.
    """
    return set_context(
        {
            "request_id": incoming_request_id or str(uuid.uuid4()),
            "ip": "",
            "method": "JOB",  # synthetic marker — not an HTTP method
            "path": job_name,  # what this run is, e.g. "board-embedding-worker"
            "proxy": "cron",  # marks the originator
            "start_time": time.time(),
        }
    )


def exit_job_context(token) -> None:
    _request_ctx.reset(token)


def _safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


# Keys we consume explicitly or that are loguru/imbrace internals — kept out of `meta`.
_RESERVED_EXTRA = {
    "function",
    "entity",
    "status_code",
    "response_time",
    "auditable",
    "__canonical__",
    "extra_json",
}


def loguru_canonical_format(record: "Any") -> str:
    """
    Loguru format function: stuff a canonical JSON line into the record's extra
    and return a template that emits it verbatim (no brace re-parsing) — the same
    trick loguru's own `file_format` uses.
    """
    ctx = _request_ctx.get() or {}
    extra = record["extra"]
    dt = record["time"].astimezone(timezone.utc)

    line = {
        "ip": ctx.get("ip", ""),
        "request_id": ctx.get("request_id", ""),
        "date_time": dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z",
        "time": int(record["time"].timestamp() * 1000),
        "method_request": ctx.get("method", ""),
        "request_path": ctx.get("path", ""),
        "service_name": SERVICE_NAME,
        "env": ENV,
        "type_of_entity": extra.get("entity") or "EMPTY",
        "function_of_code": extra.get("function") or record["function"] or "",
        "description_message": record["message"],
        "response_time": extra.get("response_time"),
        "status_code": extra.get("status_code"),
        "proxy": ctx.get("proxy", ""),
        "level": _LEVEL_MAP.get(record["level"].name, record["level"].name.lower()),
    }

    meta = {k: v for k, v in extra.items() if k not in _RESERVED_EXTRA}
    if meta:
        line["meta"] = meta
    if record["exception"] is not None:
        line["details"] = [str(record["exception"])]

    extra["__canonical__"] = json.dumps(
        {k: v for k, v in line.items() if v is not None}, default=_safe
    )
    return "{extra[__canonical__]}\n"


def add_request_context_middleware(app) -> None:
    """Register the HTTP middleware that mints/propagates the correlation id."""

    @app.middleware("http")
    async def _request_context(request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = (
            incoming if incoming and 0 < len(incoming) <= 64 else str(uuid.uuid4())
        )

        fwd = request.headers.get("x-forwarded-for")
        ip = (fwd.split(",")[0].strip() if fwd else None) or (
            request.client.host if request.client else ""
        )

        ctx = {
            "request_id": request_id,
            "ip": ip,
            "method": request.method,
            "path": request.url.path,
            "proxy": request.headers.get(PROXY_HEADER) or "client",
            "start_time": time.time(),
        }
        token = set_context(ctx)
        status = 500  # default if call_next raises
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id  # echo back
            return response
        finally:
            response_time = int((time.time() - ctx["start_time"]) * 1000)
            level = "ERROR" if status >= 500 else "WARNING" if status >= 400 else "INFO"
            logger.bind(
                function="accessLogger",
                entity="EMPTY",
                status_code=status,
                response_time=response_time,
            ).log(level, "http_request_completed")
            _request_ctx.reset(token)
