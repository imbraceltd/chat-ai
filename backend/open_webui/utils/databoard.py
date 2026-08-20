from typing import List, Dict, Any
import logging
import copy
import asyncio
from open_webui.internal.mongo_db import mongodb_client, get_mongodb_session, OPENAI_DB_NAME
from datetime import datetime
from open_webui.env import SRC_LOG_LEVELS
import requests
import random
import json
import pycountry

from open_webui.config import DATABOARD_CONFIG
DATABOARD_HOST = DATABOARD_CONFIG.get("host", "http://localhost:8081")

log = logging.getLogger(__name__)

output_schema_properties = {
    "Currency": {
        "example": '{"amounts": <number>, "currency_code": "<string>"}',
        "type": "object",
        "description": "Format: Currency code; ",
        "properties": {
            "amounts": {
                "type": "number",
                "default": 0,
                "description": "The numerical value of the amount (e.g., 100.50 for $100.50).",
            },
            "currency_code": {
                "type": "string",
                "default": "HKD",
                "enum": [
                    "HKD", "VND", "MYR", "USD", "CNY", "TWD", "AUD",
                    "CAD", "CHF", "EUR", "GBP", "JPY", "NZD", "SGD",
                    "THB", "ZAR",
                ],
                "description": "The ISO 4217 three-letter currency code (e.g., 'USD' for US Dollar, 'HKD' for Hong Kong Dollar).",
            },
        },
        "required": ["amounts", "currency_code"],
    },
    "Notes": {
        "example": "<string>",
        "type": "string",
        "description": "",
    },
    "Attachment": {
        "example": '[{"type": "url","data": {"name": "<string>","url": "<string>","extension": "<string>"}}]',
        "type": "array",
        "description": "",
        "items": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "default": "url",
                },
                "data": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "the same at file url",
                        },
                        "url": {
                            "type": "string",
                            "format": "uri",
                            "description": "file url",
                        },
                    },
                    "required": ["name", "url"],
                },
            },
            "required": ["type", "data"],
        },
    },
    "Origin": {
        "example": "<string>",
        "type": "string",
        "description": "",
        "enum": ["Option 1"],
    },
    "Priority": {
        "example": "<string>",
        "type": "string",
        "description": "",
        "enum": ["Low", "Medium", "High", "Very High"],
    },
    "Country": {
        "example": '{ country_code: "<country_code>", country_name: "<country_name>" }',
        "type": "object",
        "description": "",
        "properties": {
            "country_code": {
                "type": "string",
                "default": "HK",
                "description": "The ISO 3166-1 alpha-2 country code (e.g., 'US' for United States, 'HK' for Hong Kong).",
            },
            "country_name": {
                "type": "string",
                "default": "Hong Kong",
                "description": "The full name of the country or region (e.g., 'United States', 'Hong Kong').",
            },
        },
        "required": ["country_code", "country_name"],
    },
    "Checkbox": {
        "example": "<boolean>",
        "type": "boolean",
        "description": "",
    },
    "Link": {
        "example": "<string>",
        "type": "string",
        "description": "Format: valid url; ",
    },
    "Phone": {
        "example": '{"country_code":"<country_code>","country_calling_code":"<country_calling_code>","phone":"<phone without country code>","calling_code_with_number":"<calling code with number and start with +>","national_number":"<phone in national number>"}',
        "type": "object",
        "description": "",
        "properties": {
            "country_code": {
                "type": "string",
                "default": "HK",
                "description": "The ISO 3166-1 alpha-2 country code (e.g., 'US' for United States, 'HK' for Hong Kong).",
            },
            "country_calling_code": {
                "type": "string",
                "default": "852",
                "description": "The international calling code without the '+' symbol (e.g., '1' for US/Canada, '852' for Hong Kong).",
            },
            "phone": {
                "type": "string",
                "description": "The local phone number without country code or special characters (e.g., '98765432').",
            },
            "calling_code_with_number": {
                "type": "string",
                "description": "The full international phone number with '+' symbol, country code, and local number (e.g., '+85298765432').",
            },
            "national_number": {
                "type": "string",
                "description": "The formatted local phone number without country code (same as 'phone' field in most cases)(e.g., 98765432).",
            },
        },
        "required": [
            "country_code",
            "country_calling_code",
            "phone",
            "calling_code_with_number",
            "national_number",
        ],
    },
    "Email": {
        "example": "<string>; ",
        "type": "string",
        "description": "Format: valid email; ",
    },
    "Datetime": {
        "example": "yyyy-MM-ddTHH:mm:ssZ",
        "type": "string",
        "description": "Format: datetime in ISO 8601 format, like yyyy-MM-ddTHH:mm:ssZ; ",
    },
    "Time": {
        "example": "yyyy-MM-ddTHH:mm:ssZ",
        "type": "string",
        "description": "Format: datetime in ISO 8601 format, like yyyy-MM-ddTHH:mm:ssZ; ",
    },
    "Date": {
        "example": "yyyy-MM-ddTHH:mm:ssZ",
        "type": "string",
        "description": "Format: datetime in ISO 8601 format, like yyyy-MM-ddTHH:mm:ssZ; ",
    },
    "Number": {
        "example": "<number>",
        "type": "number",
        "description": "number",
    },
    "MultipleSelection": {
        "example": '["<string>"]',
        "type": "array",
        "description": "",
        "items": {
            "type": "string",
            "enum": ["Option 1", "Option 2"],
        },
    },
    "SingleSelection": {
        "example": "<string>",
        "type": "string",
        "description": "",
        "enum": ["Option 1"],
    },
    "LongText": {
        "example": "<string>",
        "type": "string",
        "description": "",
    },
    "ShortText": {
        "example": "<string>",
        "type": "string",
        "description": "Less than 150 characters; ",
    },
    "TableInTable": {
        "type": "array",
        "description": "",
        "items": {
            "type": "object",
            "properties": {},
        },
    },
}

async def get_board_schema(organization_id: str, board_id: str) -> dict | None:
   try:
      url = f"{DATABOARD_HOST}/api/boards/{board_id}/schema"
      log.info(f"Fetching board schema from: {url}")

      response = requests.get(url)
      log.info(f"Board schema response status: {response.status_code}")

      data = response.json()
    #   log.info(f"Board schema response: {json.dumps(data, default=str)[:100]}")
      return data
   except Exception as e:
      log.error(f"Error getting board schema: {e}")
      return None



# Field type to JSON schema type mapping
FIELD_TYPES = {
    "ShortText": "string",
    "LongText": "string",
    "SingleSelection": "string",
    "MultipleSelection": "array",
    "Number": "number",
    "Time": "string",
    "Date": "string",
    "Datetime": "string",
    "Link": "string",
    "Priority": "string",
    "Assignee": "Assignee",
    "Email": "string",
    "Phone": "object",
    "RichText": "string",
    "Country": "object",
    "Notes": "string",
    "Origin": "string",
    "Attachment": "array",
    "Currency": "object",
    "Checkbox": "boolean",
    "TableInTable": "array",
}

SKIPPED_FIELD_TYPES = {"Assignee", "TableInTable"}


def _build_field_property(field: dict) -> dict | None:
    """
    Build a JSON schema property dict for a single board field.
    Returns None if the field should be skipped.
    """
    field_type = field.get("type")
    field_name = field.get("name")

    if not field_name or field_type in SKIPPED_FIELD_TYPES or field_type not in FIELD_TYPES:
        return None

    base_property_schema = output_schema_properties.get(field_type, {})
    prop = base_property_schema.copy()

    # Safely combine descriptions
    description = prop.get("description", "")
    field_description = field.get("description")
    if field_description:
        description += field_description
    prop["description"] = description

    # Handle enums
    _enum = None
    if field_type == "Country":
        _enum = [country.alpha_2 for country in pycountry.countries]
    else:
        field_data = field.get("data")
        if isinstance(field_data, list):
            _enum = [item.get("value") for item in field_data if item.get("value") is not None]

    if FIELD_TYPES.get(field_type) == "string" and isinstance(_enum, list) and len(_enum) > 0:
        prop["enum"] = _enum

    # Handle default value for identifiers
    if field.get("is_identifier"):
        prop["default"] = str(random.randint(1000, 9999))

    return prop


def _build_table_in_table_property(field: dict, child_fields: list) -> dict | None:
    """
    Build a TableInTable JSON schema property from a list of child field dicts.
    Returns None if no usable child fields are produced.
    """
    child_properties = {}
    for child_field in child_fields or []:
        child_prop = _build_field_property(child_field)
        if child_prop is not None:
            child_properties[child_field.get("name")] = child_prop

    if not child_properties:
        return None

    base = copy.deepcopy(output_schema_properties.get("TableInTable", {}))
    description = base.get("description", "")
    field_description = field.get("description")
    if field_description:
        description += field_description
    base["description"] = description
    base["items"]["properties"] = child_properties
    return base


def _resolve_table_in_table_child_board_id(field: dict) -> str | None:
    """
    Resolve the child board id for a TableInTable field, preferring the new
    settings.childBoardId shape and falling back to the legacy data[0].value/_id.
    """
    settings = field.get("settings")
    if isinstance(settings, dict):
        child_board_id = settings.get("childBoardId") or settings.get("child_board_id")
        if child_board_id:
            return child_board_id
    field_data = field.get("data")
    if isinstance(field_data, list) and len(field_data) > 0:
        return field_data[0].get("value") or field_data[0].get("_id")
    return None


async def get_board_as_function_call_schema(board_info: dict, organization_id: str = None) -> dict:
    """
    Converts a boardInfo object into a JSON Schema for AI function-calling.

    Args:
        board_info: A dictionary containing the board schema information.
        organization_id: Organization ID, required for resolving TableInTable child boards.

    Returns:
        A dictionary representing the JSON Schema.
    """
    properties = {}

    fields = board_info.get("boardSchema", {}).get("fields", [])

    # Collect TableInTable fields that still need a child-board HTTP fetch.
    # Fields that ship `child_board_fields` inline are resolved immediately.
    table_in_table_fields = []
    for field in fields:
        field_type = field.get("type")
        field_name = field.get("name")

        if not field_name:
            continue

        if field_type == "TableInTable":
            inline_child_fields = field.get("child_board_fields")
            if isinstance(inline_child_fields, list) and inline_child_fields:
                prop = _build_table_in_table_property(field, inline_child_fields)
                if prop is not None:
                    properties[field_name] = prop
                else:
                    log.warning(
                        f"Skipping TableInTable field '{field_name}': inline child_board_fields produced no usable fields"
                    )
                continue

            if not organization_id:
                log.warning(f"Skipping TableInTable field '{field_name}': no organization_id provided")
                continue
            child_board_id = _resolve_table_in_table_child_board_id(field)
            if not child_board_id:
                log.warning(f"Skipping TableInTable field '{field_name}': no child board ID found")
                continue
            table_in_table_fields.append((field_name, field, child_board_id))
            continue

        # Build property for regular fields
        prop = _build_field_property(field)
        if prop is not None:
            properties[field_name] = prop

    # Fetch all child board schemas concurrently
    if table_in_table_fields:
        child_board_ids = [cb_id for _, _, cb_id in table_in_table_fields]
        child_board_results = await asyncio.gather(
            *[get_board_schema(organization_id, cb_id) for cb_id in child_board_ids],
            return_exceptions=True,
        )

        for (field_name, field, child_board_id), child_board_info in zip(table_in_table_fields, child_board_results):
            if isinstance(child_board_info, Exception):
                log.error(f"Error fetching child board for TableInTable '{field_name}': {child_board_info}")
                continue
            if not child_board_info:
                log.warning(f"Skipping TableInTable field '{field_name}': child board fetch returned None")
                continue

            child_fields = child_board_info.get("boardSchema", {}).get("fields", [])
            prop = _build_table_in_table_property(field, child_fields)
            if prop is None:
                log.warning(f"Skipping TableInTable field '{field_name}': child board has no usable fields")
                continue
            properties[field_name] = prop

    # Add the hardcoded fallback and comment properties
    properties["ReQuEsT_HuMaN"] = {
        "type": "boolean",
        "description": "Return true if you are not confident about filling in all the required values; otherwise, return false.",
        "default": False,
        "example": "<boolean>",
    }

    properties["SaY_To_HuMaN"] = {
        "type": "string",
        "description": "Please share your comments here in at least 20 words to describe how you performed.",
        "example": "<string>",
    }

    schema = {
        "title": "output schema",
        "type": "object",
        "properties": properties,
    }

    return schema
