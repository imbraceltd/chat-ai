"""Helpers for provisioning Document-AI data boards from schemas and linking
them to an assistant via the databoard service.

The databoard endpoint POST {DATABOARD_HOST}/api/schemas/_link-assistant
provisions one board per schema and links it to the given assistant. It scopes
by `x-organization-id` and is NOT idempotent: re-linking the same
(assistant_id, schema_id) provisions a brand-new board every time, so callers
must only send schema_ids that are not already linked (no board_id yet).
"""

import logging
from typing import Dict, List

import aiohttp

from open_webui.config import DATABOARD_CONFIG
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("ASSISTANT_APPS", logging.INFO))

DATABOARD_HOST = DATABOARD_CONFIG.get("host", "http://localhost:8081")


def collect_owned_board_ids(document_ai: Dict) -> set:
    """Board ids this assistant already owns, read from its persisted
    document_ai.schemas. Used to tell an assistant's own boards apart from a
    board_id copied over from another assistant / the schema picker."""
    ids: set = set()
    if isinstance(document_ai, dict):
        for s in document_ai.get("schemas") or []:
            if isinstance(s, dict) and s.get("board_id"):
                ids.add(s["board_id"])
    return ids


def drop_unowned_board_ids(document_ai: Dict, owned_board_ids: set) -> None:
    """Strip `board_id` from schema entries whose board this assistant does NOT
    already own (mutates `document_ai` in place).

    Each assistant gets its own board per schema, so a board_id that wasn't
    provisioned by THIS assistant (copied from another assistant, or prefilled
    by the FE schema picker) must not short-circuit provisioning — otherwise
    provision_document_ai_boards skips the `_link-assistant` call and the
    assistant is never appended to the schema's agent_ids. Stripping it forces a
    fresh per-assistant board + a proper link. Pass an empty set on create so
    every incoming board_id is dropped (a new assistant owns nothing yet)."""
    if not isinstance(document_ai, dict):
        return
    for s in document_ai.get("schemas") or []:
        if (
            isinstance(s, dict)
            and s.get("board_id")
            and s["board_id"] not in owned_board_ids
        ):
            s.pop("board_id", None)


async def link_assistant_to_schemas(
    organization_id: str,
    assistant_id: str,
    schemas: List[Dict[str, str]],
    user_id: str = "",
    business_unit_id: str = "",
) -> List[Dict[str, str]]:
    """Provision boards from schemas and link them to an assistant.

    `schemas` is a list of {"schema_id", "data_board_name"(optional)} — each
    entry provisions one board, named by its `data_board_name` (falls back to
    the schema name when omitted). Returns a list of {"board_id", "schema_id"}
    pairs. Raises on HTTP error.
    """
    url = f"{DATABOARD_HOST}/api/schemas/_link-assistant"
    headers = {
        "x-organization-id": organization_id,
        "x-user-id": user_id or "",
        "x-business-unit-id": business_unit_id or "",
        "Content-Type": "application/json",
    }
    payload = {"assistant_id": assistant_id, "schemas": schemas}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
    if isinstance(data, dict):
        return data.get("data", []) or []
    return data or []


async def unlink_assistant_from_schemas(
    organization_id: str,
    assistant_id: str,
    schemas: List[Dict[str, str]],
    user_id: str = "",
    business_unit_id: str = "",
) -> int:
    """Tell databoard to remove this assistant from the given schemas.

    `schemas` is a list of {"schema_id", "board_id"(optional)} — each schema
    drops `assistant_id` from its agent_ids and (when board_id is given) that
    board from its databoard_ids.
    POST {DATABOARD_HOST}/api/schemas/_unlink-schema-from-assistant.
    Returns the number of schemas updated. Raises on HTTP error.
    """
    if not schemas:
        return 0
    url = f"{DATABOARD_HOST}/api/schemas/_unlink-schema-from-assistant"
    headers = {
        "x-organization-id": organization_id,
        "x-user-id": user_id or "",
        "x-business-unit-id": business_unit_id or "",
        "Content-Type": "application/json",
    }
    payload = {"assistant_id": assistant_id, "schemas": schemas}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
    if isinstance(data, dict):
        return data.get("unlinked", 0) or 0
    return 0


def _schema_board_pairs(document_ai: Dict) -> Dict[str, str]:
    """Map schema_id -> board_id (board_id may be None) from a document_ai."""
    pairs: Dict[str, str] = {}
    if isinstance(document_ai, dict):
        for s in document_ai.get("schemas") or []:
            if isinstance(s, dict) and s.get("schema_id"):
                pairs[s["schema_id"]] = s.get("board_id")
    return pairs


def carry_over_board_ids(new_document_ai: Dict, old_document_ai: Dict) -> None:
    """Restore the persisted board_id onto incoming schemas (mutates in place).

    On update the incoming document_ai may omit board_id (e.g. the channel
    service re-saves the assistant to wire its workflow, resending the original
    schemas without the board_ids that create() already provisioned). Without
    this, provision_document_ai_boards would see schemas with no board_id and
    provision a SECOND board per schema. For every incoming schema still missing
    a board_id, copy the board_id this assistant already owns for that schema_id
    so provisioning skips it. Genuinely new schemas (absent from the old
    document_ai) keep no board_id and are provisioned as normal.

    Call AFTER drop_unowned_board_ids — that strips foreign board_ids first, then
    this fills in only the assistant's own persisted boards.
    """
    if not isinstance(new_document_ai, dict):
        return
    old_pairs = _schema_board_pairs(old_document_ai)
    for s in new_document_ai.get("schemas") or []:
        if (
            isinstance(s, dict)
            and not s.get("board_id")
            and old_pairs.get(s.get("schema_id"))
        ):
            s["board_id"] = old_pairs[s["schema_id"]]


async def unlink_removed_document_ai_schemas(
    organization_id: str,
    assistant_id: str,
    old_document_ai: Dict,
    new_document_ai: Dict,
    user_id: str = "",
    business_unit_id: str = "",
) -> None:
    """Unlink schemas dropped from the assistant's document_ai on update.

    Compares the persisted (old) document_ai against the incoming (new) one and
    tells databoard to detach the assistant from every schema that is no longer
    present, carrying each schema's board_id so its per-assistant board is also
    removed from databoard_ids. Best-effort: never raises — a cleanup failure
    must not roll back the assistant update.
    """
    old_pairs = _schema_board_pairs(old_document_ai)
    new_ids = set(_schema_board_pairs(new_document_ai).keys())
    removed = [
        {"schema_id": sid, **({"board_id": bid} if bid else {})}
        for sid, bid in old_pairs.items()
        if sid not in new_ids
    ]
    if not removed:
        return
    try:
        await unlink_assistant_from_schemas(
            organization_id,
            assistant_id,
            removed,
            user_id=user_id,
            business_unit_id=business_unit_id,
        )
    except Exception as error:
        removed_ids = [r["schema_id"] for r in removed]
        log.error(
            f"Failed to unlink removed schemas {removed_ids} from assistant {assistant_id}: {error}"
        )


async def rename_board(
    organization_id: str,
    board_id: str,
    name: str,
    user_id: str = "",
    business_unit_id: str = "",
    board_category_id: str = None,
    board_deployment_access: List[str] = None,
) -> None:
    """Update an existing data board (PATCH {DATABOARD_HOST}/api/boards/:id).

    Used on update so an already-linked board picks up the latest
    data_board_name (and, when provided, its category + deployment access)
    without provisioning a duplicate. Raises on HTTP error.

    `board_category_id` is only sent when non-empty (empty = leave as-is).
    `board_deployment_access` is sent whenever it is a list (including `[]`) so
    the user can reset the board back to "all teams". Maps to the board's
    `category` / `team_ids` fields, mirroring the /databoards Board Profile.
    """
    url = f"{DATABOARD_HOST}/api/boards/{board_id}"
    headers = {
        "x-organization-id": organization_id,
        "x-user-id": user_id or "",
        "x-business-unit-id": business_unit_id or "",
        "Content-Type": "application/json",
    }
    payload = {"name": name}
    if board_category_id:
        payload["category"] = board_category_id
    if isinstance(board_deployment_access, list):
        payload["team_ids"] = board_deployment_access
    async with aiohttp.ClientSession() as session:
        async with session.patch(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()


async def provision_document_ai_boards(
    organization_id: str,
    assistant_id: str,
    document_ai: Dict,
    user_id: str = "",
    business_unit_id: str = "",
) -> bool:
    """Best-effort: link any not-yet-linked schemas to their boards and fold the
    returned board_id back onto each schema in `document_ai` (mutated in place).

    A schema that already carries a board_id is considered linked and skipped to
    avoid duplicate boards (the link endpoint is not idempotent). Returns True if
    at least one schema gained a board_id. Never raises — a link failure must not
    roll back assistant create/update, so it only logs and returns False.
    """
    if not isinstance(document_ai, dict):
        return False

    schemas = document_ai.get("schemas") or []

    # Already-linked schemas (carry a board_id): rename the existing board to the
    # latest data_board_name instead of re-provisioning. Best-effort and
    # idempotent — a no-op when the name is unchanged.
    for s in schemas:
        if (
            isinstance(s, dict)
            and s.get("board_id")
            and s.get("data_board_name")
        ):
            try:
                await rename_board(
                    organization_id,
                    s["board_id"],
                    s["data_board_name"],
                    user_id=user_id,
                    business_unit_id=business_unit_id,
                    board_category_id=s.get("board_category_id"),
                    board_deployment_access=s.get("board_deployment_access"),
                )
            except Exception as error:
                log.error(
                    f"Failed to rename board {s.get('board_id')} "
                    f"to '{s.get('data_board_name')}': {error}"
                )

    # Not-yet-linked schemas (no board_id) → provision a board each, carrying the
    # per-schema data_board_name so each board keeps the name the user set in the
    # Document-AI form (not the schema name).
    link_inputs = [
        {
            "schema_id": s["schema_id"],
            **({"data_board_name": s["data_board_name"]} if s.get("data_board_name") else {}),
            # Provision the board into the chosen category (empty → "Doc Agent" default)
            # and with the chosen deployment access (empty list → all teams).
            **({"board_category_id": s["board_category_id"]} if s.get("board_category_id") else {}),
            **(
                {"board_deployment_access": s["board_deployment_access"]}
                if isinstance(s.get("board_deployment_access"), list) and s["board_deployment_access"]
                else {}
            ),
        }
        for s in schemas
        if isinstance(s, dict) and s.get("schema_id") and not s.get("board_id")
    ]
    if not link_inputs:
        return False

    try:
        links = await link_assistant_to_schemas(
            organization_id,
            assistant_id,
            link_inputs,
            user_id=user_id,
            business_unit_id=business_unit_id,
        )
        board_by_schema = {
            link.get("schema_id"): link.get("board_id")
            for link in links
            if isinstance(link, dict)
        }
        linked = False
        for schema in schemas:
            board_id = board_by_schema.get(schema.get("schema_id"))
            if board_id:
                schema["board_id"] = board_id
                linked = True
        # Back-compat: keep a top-level board_id pointing at the first board.
        if linked and not document_ai.get("board_id"):
            document_ai["board_id"] = next(
                (b for b in board_by_schema.values() if b), None
            )
        return linked
    except Exception as error:
        linked_schema_ids = [s["schema_id"] for s in link_inputs]
        log.error(
            f"Failed to link assistant {assistant_id} to schemas {linked_schema_ids}: {error}"
        )
        return False
