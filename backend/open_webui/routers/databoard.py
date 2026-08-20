import logging
import asyncio
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Path, UploadFile, File, Form
from pydantic import BaseModel
import aiohttp
import json
from openpyxl import load_workbook

from open_webui.env import SRC_LOG_LEVELS
from open_webui.utils.data import convert_metadata_to_string
from open_webui.utils import rag
from open_webui.internal.mongo_db import mongodb_client
from open_webui.config import DATABOARD_CONFIG, MONGODB_CONFIG
from openai import AsyncOpenAI
import os
import io
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("DATABOARD", logging.INFO))

router = APIRouter()

# Configuration
DB_NAME = MONGODB_CONFIG.get("openai_db_name", "openai_db")
RAG_COLLECTION_NAME = "rag"
DATABOARD_HOST = DATABOARD_CONFIG.get("host", "http://localhost:8081")

MAX_ROWS = 3  # how many rows to sample
MAX_COLS = 20  # how many columns to sample
MAX_TOKENS_LIMIT = 2000  # GPT input safeguard

# ========== Type rules ==========
ALLOWED_TYPES = {
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

SKIPPED_FIELD_TYPES = ["Assignee"]

router = APIRouter()
log = logging.getLogger(__name__)


class MergePreviewRequest(BaseModel):
    files: List[str]  # list of URLs (CSV / Excel)


async def fetch_json(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            # Ensure the request succeeded
            response.raise_for_status()
            # Parse and return JSON
            return await response.json()


async def put_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.put(url, json=payload) as response:
            if response.status not in (200, 201):
                raise HTTPException(
                    status_code=response.status, detail=await response.text()
                )
            return await response.json()


async def patch_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.patch(url, json=payload) as response:
            if response.status not in (200, 201):
                raise HTTPException(
                    status_code=response.status, detail=await response.text()
                )
            return await response.json()


async def post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status not in (200, 201):
                raise HTTPException(
                    status_code=response.status, detail=await response.text()
                )
            return await response.json()


# -------------------------------
# Utility to summarize dataframe
# -------------------------------
def summarize_dataframe(df: pd.DataFrame, max_rows=MAX_ROWS, max_cols=MAX_COLS):
    df = df.iloc[:max_rows, :max_cols]  # limit rows and columns
    schema_summary = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        examples = df[col].dropna().astype(str).head(2).tolist()
        schema_summary.append({"column": col, "dtype": dtype, "examples": examples})
    return schema_summary


# -------------------------------
# Token-safe truncation helper
# -------------------------------
def truncate_json(json_text: str, max_chars: int = 2000) -> str:
    """Trim long JSON strings to safe size"""
    if len(json_text) > max_chars:
        return json_text[:max_chars] + "..."
    return json_text

def normalize_field_type(gpt_type: str) -> str:
    # fallback to ShortText if invalid
    if gpt_type not in ALLOWED_TYPES:
        return "ShortText"
    return gpt_type

# =====================================================
# 🚀 DETECT FIELDS — optimized for large files
# =====================================================
@router.post("/detect-fields")
async def detect_fields(payload: Dict[str, Any]):
    """
    Detect fields from uploaded spreadsheet URLs using AI,
    automatically merging similar/synonymous fields with aliases,
    and identifying new fields not in the board schema.
    """
    try:
        # Step 1️⃣ — Extract payload
        org_id = payload.get("org_id")
        board_id = payload.get("board_id")
        file_urls: List[str] = payload.get("files", [])
        if not org_id or not board_id or not file_urls:
            raise HTTPException(400, "org_id, board_id, and files[] are required")

        # Step 2️⃣ — Fetch board schema
        board_url = f"{DATABOARD_HOST}/api/boards/{board_id}"
        log.info(f"Fetching board schema from {board_url}")
        board_data = await fetch_json(board_url)
        existing_fields = board_data.get("data", {}).get("fields", [])
        board_field_map = {
            f["name"]: f["type"] for f in existing_fields if not f.get("hidden", False)
        }
        existing_names_lower = {n.lower() for n in board_field_map.keys()}

        board_field_summary = json.dumps(
            [{"name": n, "type": t} for n, t in board_field_map.items()], indent=2
        )

        # Step 3️⃣ — Summarize file schemas
        summaries, schemas = [], []
        for url in file_urls:
            filename = os.path.basename(url).split("?")[0]
            log.info(f"Reading file from URL: {filename}")

            if filename.endswith(".csv"):
                df = pd.read_csv(url)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(url)
            else:
                raise HTTPException(400, f"Unsupported file type: {filename}")

            schema_summary = summarize_dataframe(df)
            schemas.append({"filename": filename, "schema": schema_summary})
            summaries.append({
                "filename": filename,
                "rows": len(df),
                "columns": list(df.columns)
            })

        file_schema_json = truncate_json(json.dumps(schemas, ensure_ascii=False))

        # Step 4️⃣ — Initialize OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key)

        # Step 5️⃣ — Build GPT prompt
        prompt = f"""
You are an AI schema unifier that maps spreadsheet fields to a board schema.

### Task
Unify similar or synonymous field names across spreadsheets and board fields.
If a spreadsheet column means the same as an existing board field, merge it under `aliases`.

Examples:
- "Name", "Full Name", "Customer Name" → unify as "Name"
- "Zip Code", "Postal Code" → unify as "Zip Code"
- "Trial", "Test" → unify as "Test"
- "Mileage", "Distance" → unify as "Distance"

Existing Board Fields:
{board_field_summary}

Uploaded File Schemas:
{file_schema_json}

Allowed field types: {list(ALLOWED_TYPES.keys())}
Skip types: {SKIPPED_FIELD_TYPES}

Return only JSON:
{{
  "data": [
    {{
      "name": "Name",
      "type": "ShortText",
      "exists": true,
      "aliases": ["Full Name", "Customer Name"]
    }},
    {{
      "name": "Mileage",
      "type": "Number",
      "exists": false,
      "aliases": []
    }}
  ]
}}
"""

        # Step 6️⃣ — GPT inference
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a data schema expert. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

        # Step 7️⃣ — Parse GPT JSON safely
        try:
            detected = json.loads(response.choices[0].message.content)
        except Exception as e:
            log.error(f"Failed to parse GPT JSON: {e}")
            detected = {"data": []}

        # Step 8️⃣ — Normalize unified fields
        unified = {}
        for item in detected.get("data", []):
            if "name" not in item:
                continue
            ftype = normalize_field_type(item.get("type", "ShortText"))
            if ftype in SKIPPED_FIELD_TYPES:
                continue
            unified[item["name"]] = {
                "name": item["name"],
                "type": ftype,
                "exists": item.get("exists", True),
                "aliases": item.get("aliases", []),
            }

        # Step 9️⃣ — Add missing board fields
        for name, t in board_field_map.items():
            if name not in unified:
                unified[name] = {"name": name, "type": t, "exists": True, "aliases": []}

        # 🔟 — Extract new fields (not existing in board)
        new_fields = [
            {"name": f["name"], "type": f["type"]}
            for f in unified.values()
            if not f.get("exists", False) or f["name"].lower() not in existing_names_lower
        ]

        log.info(f"Unified fields: {len(unified)}, new fields: {len(new_fields)}")

        return {
            "summary": summaries,
            "organization_id": org_id,
            "board_id": board_id,
            "unified_fields": list(unified.values()),
            "new_fields": new_fields,
        }

    except Exception as e:
        log.exception("Error detecting fields")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# 🚀 MERGE PREVIEW — optimized for large files
# =====================================================
@router.post("/merge-preview")
async def merge_preview(payload: Dict[str, Any]):
    try:
        org_id = payload.get("org_id")
        board_id = payload.get("board_id")
        file_urls: List[str] = payload.get("files", [])
        if not org_id or not board_id or not file_urls:
            raise HTTPException(400, "org_id, board_id, and files[] are required")

        # 1️⃣ Fetch board schema
        board_url = f"{DATABOARD_HOST}/api/boards/{board_id}"
        board_data = await fetch_json(board_url)
        board_fields = [
            {"name": f["name"], "type": f["type"]}
            for f in board_data.get("data", {}).get("fields", [])
            if not f.get("hidden", False)
        ]
        board_field_names = [f["name"] for f in board_fields]
        board_summary = ", ".join(board_field_names[:20])
        existing_names_lower = {n.lower() for n in board_field_names}

        # 2️⃣ Summarize uploaded files
        summaries, schemas = [], []
        for url in file_urls:
            filename = os.path.basename(url).split("?")[0]
            log.info(f"Loading: {filename}")

            if filename.endswith(".csv"):
                df = pd.read_csv(url)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(url)
            else:
                raise HTTPException(400, f"Unsupported file type: {filename}")

            schema_summary = summarize_dataframe(df)
            schemas.append({"filename": filename, "schema": schema_summary})
            summaries.append(
                {"filename": filename, "rows": len(df), "columns": list(df.columns)}
            )

        schema_json = truncate_json(json.dumps(schemas, ensure_ascii=False))

        # 3️⃣ GPT setup
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(500, "Missing OPENAI_API_KEY")
        client = AsyncOpenAI(api_key=api_key)

        # 4️⃣ Optimized GPT prompt
        prompt = f"""
You are an expert in unifying spreadsheet schemas.

### Board Fields
{json.dumps(board_fields, indent=2)}

### Uploaded File Schemas
{schema_json}

### Rules
- Merge equivalent or synonymous columns (e.g., "Trial" ~ "Test", "Customer Name" ~ "Name").
- Prefer using existing board field names whenever possible.
- Keep all columns (union).
- Include "mappings" showing uploaded → board field relationships.
- If a column doesn’t match any board field, mark it as new and infer type.
- Return JSON:
{{
  "merged_columns": [...],
  "merged_preview": [...],
  "mappings": [{{"uploaded": "Trial", "mapped_to": "Test"}}],
  "new_fields": [{{"name": "Mileage", "type": "Number"}}]
}}
"""

        # 5️⃣ GPT call
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=1000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You unify and map spreadsheet schemas efficiently."},
                {"role": "user", "content": prompt},
            ],
        )

        # 6️⃣ Parse GPT result
        try:
            gpt_result = json.loads(response.choices[0].message.content)
        except Exception as e:
            log.error(f"Failed to parse GPT JSON: {e}")
            gpt_result = {}

        merged_columns = gpt_result.get("merged_columns", [])
        merged_preview = gpt_result.get("merged_preview", [])
        mappings = gpt_result.get("mappings", [])
        gpt_new_fields = gpt_result.get("new_fields", [])

        # 7️⃣ Normalize GPT new_fields
        normalized_new_fields = []
        for nf in gpt_new_fields:
            if "name" not in nf:
                continue
            name = nf["name"].strip()
            ftype = nf.get("type", "ShortText")
            if ftype not in ALLOWED_TYPES:
                ftype = "ShortText"
            # only keep truly new (not already in board)
            if name.lower() not in existing_names_lower:
                normalized_new_fields.append({"name": name, "type": ftype})

        # 8️⃣ Exclude mapped → board fields from new_fields
        mapped_to_board = {
            m["uploaded"].lower()
            for m in mappings
            if "mapped_to" in m and m["mapped_to"].lower() in existing_names_lower
        }

        filtered_new_fields = [
            nf for nf in normalized_new_fields
            if nf["name"].lower() not in mapped_to_board
        ]

        # 9️⃣ Also detect unmatched uploaded columns (backup fallback)
        uploaded_columns = {col for s in summaries for col in s["columns"]}
        for col in uploaded_columns:
            col_lower = col.lower()
            if (
                col_lower not in existing_names_lower
                and col_lower not in mapped_to_board
                and all(nf["name"].lower() != col_lower for nf in filtered_new_fields)
            ):
                filtered_new_fields.append({"name": col, "type": "ShortText"})

        log.info(
            f"Merged {len(merged_columns)} cols, found {len(filtered_new_fields)} new fields after mapping exclusion"
        )

        return {
            "summary": summaries,
            "merged_columns": merged_columns,
            "merged_preview": merged_preview,
            "mappings": mappings,
            "new_fields": filtered_new_fields,
        }

    except Exception as e:
        log.exception("Error merging previews")
        raise HTTPException(500, str(e))

# ========== Main logic ==========
@router.post("/ai-import-databoard")
async def import_ai_to_databoard(payload: Dict[str, Any]):
    """
    AI-driven import pipeline:
      1️⃣ Read CSV/Excel URLs -> merged preview
      2️⃣ Fetch board and field mapping
      3️⃣ Use GPT to map similar columns
      4️⃣ Create missing fields (PUT)
      5️⃣ Insert items (POST in batches of 100)
      6️⃣ Return summary with counts
    """
    try:
        org_id = payload.get("org_id")
        board_id = payload.get("board_id")
        file_urls: List[str] = payload.get("files", [])
        if not (org_id and board_id and file_urls):
            raise HTTPException(400, "org_id, board_id, and files[] are required")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(500, "Missing OPENAI_API_KEY")

        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=api_key)

        # 1️⃣ Read all files into a single DataFrame
        dfs = []
        for url in file_urls:
            if url.endswith(".csv"):
                dfs.append(pd.read_csv(url))
            elif url.endswith((".xlsx", ".xls")):
                dfs.append(pd.read_excel(url))
            else:
                raise HTTPException(400, f"Unsupported file type: {url}")

        if not dfs:
            raise HTTPException(400, "No valid data found")

        merged_df = pd.concat(dfs, ignore_index=True).fillna("")
        merged_preview = merged_df.to_dict(orient="records")
        preview_cols = list(merged_df.columns)

        # 2️⃣ Fetch board info and field map
        board_url = f"{DATABOARD_HOST}/api/boards/{board_id}"
        board_data = await fetch_json(board_url)
        field_map = {f["name"]: f["id"] for f in board_data.get("data", {}).get("fields", [])}

        # 3️⃣ GPT — semantic field name matching
        mapping_prompt = f"""
You are an AI that matches uploaded spreadsheet column names to board field names.

Board field names: {list(field_map.keys())}
Uploaded column names: {preview_cols}

Rules:
- Match semantically similar names (case-insensitive, small variations, plural/singular).
- Example: "name" → "Name", "email address" → "Email".
- If not similar, mark as is_new: true.
Return JSON:
{{
  "mappings": [
    {{"uploaded": "<uploaded_name>", "board_field": "<matched_or_same>", "is_new": false}}
  ]
}}
"""
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.1,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are a smart field name matcher for spreadsheets.",
                },
                {"role": "user", "content": mapping_prompt},
            ],
        )

        try:
            mapping_result = json.loads(response.choices[0].message.content)
            mappings = mapping_result.get("mappings", [])
        except Exception:
            mappings = [
                {"uploaded": c, "board_field": c, "is_new": c not in field_map}
                for c in preview_cols
            ]

        column_to_board = {}
        missing_cols = []
        for m in mappings:
            uploaded = m["uploaded"]
            board_field = m["board_field"]
            is_new = m.get("is_new", False)
            if not is_new and board_field in field_map:
                column_to_board[uploaded] = field_map[board_field]
            else:
                missing_cols.append(uploaded)

        # 4️⃣ Detect and add missing fields
        new_fields = []
        if missing_cols:
            type_prompt = f"""
You are an AI that decides field types for new spreadsheet columns.
Allowed types: {list(ALLOWED_TYPES.keys())}
Skip types in {SKIPPED_FIELD_TYPES}.
Columns: {missing_cols}
Return JSON: {{ "data": [{{"name": "FieldName", "type": "ShortText"}}, ...] }}
"""
            type_response = await client.chat.completions.create(
                model="gpt-4o",
                temperature=0.2,
                max_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a structured field type detector.",
                    },
                    {"role": "user", "content": type_prompt},
                ],
            )
            try:
                detected = json.loads(type_response.choices[0].message.content)
                new_fields = detected.get("data", [])
            except Exception:
                new_fields = [{"name": c, "type": "ShortText"} for c in missing_cols]

        if new_fields:
            update_url = f"{DATABOARD_HOST}/api/boards/{board_id}/fields/bulk"
            update_payload = {
                "fields": [
                    {
                        "name": nf["name"],
                        "type": nf.get("type", "ShortText"),
                        "is_unique_identifier": False,
                        "is_default": False,
                        "hidden": False,
                        "hidden_on_record": False,
                        "data": [],
                        "field_id": "",
                    }
                    for nf in new_fields
                ]
            }
            updated = await patch_json(update_url, update_payload)
            for f in updated.get("data", {}).get("fields", []):
                field_map[f["name"]] = f["id"]
                column_to_board[f["name"]] = f["id"]

        # 5️⃣ Build items with board_field_id
        all_items = []
        for row in merged_preview:
            item_fields = []
            for name, value in row.items():
                board_field_id = column_to_board.get(name) or field_map.get(name)
                if board_field_id:
                    item_fields.append(
                        {"board_field_id": board_field_id, "value": value}
                    )
            all_items.append({"fields": item_fields})

        # 6️⃣ Split into 100-row batches and insert
        create_url = f"{DATABOARD_HOST}/api/boards/{board_id}/items/bulk"
        batch_size = 100
        total_rows = len(all_items)
        success_count = 0
        error_count = 0

        for i in range(0, total_rows, batch_size):
            batch = all_items[i : i + batch_size]
            try:
                await post_json(create_url, {"items": batch})
                success_count += len(batch)
            except Exception as e:
                error_count += len(batch)
                print(f"Batch {i//batch_size+1} failed: {e}")

        # ✅ Final Summary
        return {
            "board_id": board_id,
            "organization_id": org_id,
            "total_rows_in_file": total_rows,
            "total_new_rows": success_count,
            "total_error_rows": error_count,
            "added_fields": new_fields,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
