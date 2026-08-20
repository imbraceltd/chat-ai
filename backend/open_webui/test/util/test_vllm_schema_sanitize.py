"""Unit tests for the vLLM strict json_schema sanitizer.

Verifies that a loose board output schema is rewritten into the OpenAI/vLLM strict
json_schema subset: additionalProperties=false, full `required`, nullable optionals,
stripped unsupported keywords, preserved enums — without mutating the input.

Run: pytest backend/open_webui/test/util/test_vllm_schema_sanitize.py
"""

import copy

from open_webui.utils.document_assistant import _sanitize_schema_for_vllm


def _board_like_schema():
    # Mirrors get_board_as_function_call_schema output: loose, no root `required`,
    # carries description/default/example, a nested object with its own `required`,
    # an array-of-objects (TableInTable), an enum field, and Chinese keys.
    return {
        "title": "output schema",
        "type": "object",
        "properties": {
            "事故者姓名": {"type": "string", "description": "name", "example": "賴小宇"},
            "信心分數": {"type": "number"},
            "傷害類型": {"type": "string", "enum": ["執行職務", "上下班事故"]},
            "電話": {
                "type": "object",
                "properties": {
                    "country_code": {"type": "string"},
                    "phone": {"type": "string"},
                },
                "required": ["country_code", "phone"],
            },
            "不能工作期間": {
                "type": "array",
                "description": "segments",
                "items": {
                    "type": "object",
                    "properties": {
                        "多段核定起日": {"type": "string"},
                        "多段核定迄日": {"type": "string"},
                    },
                    "required": ["多段核定起日", "多段核定迄日"],
                },
            },
        },
    }


def test_root_object_is_strict():
    out = _sanitize_schema_for_vllm(_board_like_schema())
    assert out["type"] == "object"
    assert out["additionalProperties"] is False
    # Every property must be listed in required (strict mode).
    assert set(out["required"]) == set(out["properties"].keys())


def test_absent_optional_scalars_become_nullable():
    out = _sanitize_schema_for_vllm(_board_like_schema())
    # Root has no `required`, so its scalar fields are optional → nullable unions.
    assert out["properties"]["事故者姓名"]["type"] == ["string", "null"]
    assert out["properties"]["信心分數"]["type"] == ["number", "null"]


def test_nested_object_required_fields_stay_non_null():
    out = _sanitize_schema_for_vllm(_board_like_schema())
    phone = out["properties"]["電話"]
    assert phone["additionalProperties"] is False
    assert set(phone["required"]) == {"country_code", "phone"}
    # These were in the nested object's `required` → must stay non-null.
    assert phone["properties"]["country_code"]["type"] == "string"
    assert phone["properties"]["phone"]["type"] == "string"


def test_array_items_are_strict_and_non_null():
    out = _sanitize_schema_for_vllm(_board_like_schema())
    items = out["properties"]["不能工作期間"]["items"]
    assert out["properties"]["不能工作期間"]["type"] == "array"
    assert items["additionalProperties"] is False
    assert set(items["required"]) == {"多段核定起日", "多段核定迄日"}
    assert items["properties"]["多段核定起日"]["type"] == "string"


def test_enum_preserved_and_null_appended_when_nullable():
    out = _sanitize_schema_for_vllm(_board_like_schema())
    field = out["properties"]["傷害類型"]
    assert field["type"] == ["string", "null"]  # optional → nullable
    assert "執行職務" in field["enum"] and None in field["enum"]


def test_unsupported_keywords_stripped():
    out = _sanitize_schema_for_vllm(_board_like_schema())
    assert "title" not in out
    assert "example" not in out["properties"]["事故者姓名"]
    # description is allowed and kept (helps the model).
    assert out["properties"]["事故者姓名"].get("description") == "name"


def test_descriptions_retained_at_all_levels():
    # Descriptions must survive into the json_schema at every nesting level so the model
    # keeps field semantics even though the schema is no longer injected into the prompt.
    schema = {
        "type": "object",
        "properties": {
            "事故者姓名": {"type": "string", "description": "victim full name", "example": "賴小宇"},
            "傷害類型": {"type": "string", "enum": ["執行職務"], "description": "injury category"},
            "電話": {
                "type": "object",
                "properties": {"phone": {"type": "string", "description": "phone number"}},
                "required": ["phone"],
            },
            "不能工作期間": {
                "type": "array",
                "description": "incapacity segments",
                "items": {
                    "type": "object",
                    "properties": {"多段核定起日": {"type": "string", "description": "segment start"}},
                    "required": ["多段核定起日"],
                },
            },
        },
    }
    out = _sanitize_schema_for_vllm(schema)
    p = out["properties"]
    assert p["事故者姓名"]["description"] == "victim full name"
    assert p["傷害類型"]["description"] == "injury category"
    assert p["電話"]["properties"]["phone"]["description"] == "phone number"
    assert p["不能工作期間"]["description"] == "incapacity segments"
    assert p["不能工作期間"]["items"]["properties"]["多段核定起日"]["description"] == "segment start"
    # ...but non-strict-safe keywords are still stripped.
    assert "example" not in p["事故者姓名"]


def test_input_not_mutated():
    schema = _board_like_schema()
    snapshot = copy.deepcopy(schema)
    _sanitize_schema_for_vllm(schema)
    assert schema == snapshot
