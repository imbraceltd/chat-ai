import logging

from nemoguardrails import RailsConfig, LLMRails

from open_webui.env import (
    OPENAI_API_KEY_GUARDRAILS,
    OPENAI_MODEL_GUARDRAILS,
)
from open_webui.config import MONGODB_CONFIG
from open_webui.internal.mongo_db import query

# Import nim_llm to trigger register_llm_provider before any LLMRails creation
import open_webui.llm.utils.nemo_guardrails.nim_llm  # noqa: F401

logger = logging.getLogger(__name__)

OPENAI_DB_NAME = MONGODB_CONFIG.get("openai_db_name", "imbrace_dev")

# In-memory cache of loaded LLMRails instances, keyed by config_id
llm_rails_instances = {}


def build_guard_from_config(config_doc):
    """Generate CoLang 2.x flow content from a guardrail config document."""
    competitor_keywords = config_doc.get("competitor_keywords", [])
    custom_unsafe_patterns = config_doc.get("custom_unsafe_patterns", [])
    unsafe_categories = config_doc.get("unsafe_categories", [])

    all_blocked = []
    if competitor_keywords:
        all_blocked.append(f"Competitor keywords: {', '.join(competitor_keywords)}")
    if custom_unsafe_patterns:
        all_blocked.append(f"Unsafe patterns: {', '.join(custom_unsafe_patterns)}")
    if unsafe_categories:
        all_blocked.append(f"Unsafe categories: {', '.join(unsafe_categories)}")
    blocked_str = "\n- ".join(all_blocked) if all_blocked else "None"

    instruction = (
        "You are a safety classifier. Block any input that matches the following:\n"
        f"- {blocked_str}\n"
        "If the user utterance contains or relates to any of these, Assign 'unsafe'. Otherwise, Assign 'safe' .\n"
        "Provide your safety assessment in JSON format:\n"
        '{{\"User Safety\": \"safe\" or \"unsafe\", \"Safety Categories\": \"comma-separated list of '
        'applicable safety categories\" (omit if all safe)}}\n'
        "Output only the JSON response."
    )

    colang = f'''import core
import guardrails
import llm

flow input rails $input_text
  print "FLOW NOKIA TRIGGERED"
  $input_safe = await check user utterance $input_text
  bot say "{{{{ $input_safe }}}}"
  abort

flow check user utterance $input_text -> $input_safe
  print "FLOW utterance TRIGGERED"
  print $input_text
  $is_safe = ..."""{instruction} User utterance: {{{{ $input_text }}}} """
  print "FLOW utterance is working and returning value"
  print $is_safe
  return $is_safe

flow main
  activate llm continuation
'''
    return colang


def get_full_model_config(config_doc):
    """Create the NemoGuardrails model configuration dict."""
    competitors = ", ".join(config_doc.get("competitor_keywords", []))
    patterns = ", ".join(config_doc.get("custom_unsafe_patterns", []))
    categories = ", ".join(config_doc.get("unsafe_categories", []))
    return {
        "colang_version": "2.x",
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "model": OPENAI_MODEL_GUARDRAILS,
                "api_key": OPENAI_API_KEY_GUARDRAILS,
            }
        ],
        "instructions": [
            {
                "type": "intent_classification",
                "content": f"""
You are a safety classifier. Determine if the user input is unsafe based on these rules:
- Competitor keywords: {competitors}
- Unsafe patterns: {patterns}
- Unsafe categories: {categories}
If the input mentions any of these, or is otherwise unsafe, classify as "unsafe". Otherwise, classify as "safe".

User input: {{$message}}

Output only the intent: "unsafe" or "safe"
""",
            }
        ],
    }


def load_rail_from_config(config_id: str, config_doc: dict):
    """Create a RailsConfig + LLMRails instance and cache it."""
    colang_content = build_guard_from_config(config_doc)
    logger.info(
        "Loading NemoGuardrails config for org: %s, config_id: %s",
        config_doc.get("org_id", ""),
        config_id,
    )
    config = RailsConfig.from_content(
        colang_content=colang_content, config=get_full_model_config(config_doc)
    )
    llm_rails_instances[config_id] = LLMRails(config)
    logger.info(f"Loaded NemoGuardrails config for id: {config_id}")


def get_loaded_guardrails(config_id: str):
    """Return cached LLMRails instance for a given config_id."""
    return llm_rails_instances.get(config_id)


async def load_all_configs():
    """Load all guardrail configs into memory on startup."""
    # In Postgres mode `guardrails` is a typed relational table (no JSONB `data`
    # column), so the generic document client's `SELECT id, data` fails. Read
    # via the typed repository instead; keep the Mongo path for DB_TYPE=mongodb.
    from open_webui.internal.document_store import DB_TYPE

    if DB_TYPE == "postgresql":
        from open_webui.repository.guardrail import guardrail_repo

        configs = await guardrail_repo.list_by_model("nim-nemo")
    else:
        configs = await query(
            database_name=OPENAI_DB_NAME,
            collection_name="guardrails",
            query_dict={"model": "nim-nemo"},
        )
    loaded_count = 0
    for doc in configs:
        config_id = doc.get("guardrails_config_id")
        if config_id:
            try:
                load_rail_from_config(config_id, doc)
                loaded_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to load NemoGuardrails config {config_id}: {e}"
                )
    logger.info(f"Loaded {loaded_count} NemoGuardrails configs from MongoDB")
