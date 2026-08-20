# Chat-AI Codebase Navigation Guide

## Where to find things

### API Endpoints
- All routers live in `backend/open_webui/routers/`
- Router prefixes are registered in `backend/open_webui/main.py` (lines ~1116–1198)
- Pattern: `app.include_router(router, prefix="/api/v1/<name>")`
- Notable routers: `assistants.py`, `guardrail.py`, `databoard.py`, `rag.py`, `retrieval.py`, `files.py`, `chats.py`, `auths.py`

### External Service Calls (where each host is used)
| Service | Host Env Var | Config Location | Used In |
|---|---|---|---|
| Databoard | `DATABOARD_HOST` | `config.py:2859` | `routers/databoard.py`, `utils/databoard.py`, `llm/utils/databoard.py`, `utils/board_embedding_job.py` |
| Workflow engine | `WORKFLOW_HOST` | `config.py:2851` | `llm/utils/workflow.py` |
| Imbrace backend (legacy) | `BACKEND_HOST` | `config.py:2856` | still referenced in `config.py` but no longer called for board APIs |
| Imbrace API | `IMBRACE_API_URL` | `env.py:404` | `utils/auth.py`, `utils/models.py`, `utils/file.py` |
| Ollama | `OLLAMA_HOST` | `config.py:2840` | `llm/models/ollama.py` |
| LLM Studio | `LLM_STUDIO_HOST` | `config.py:2829` | `llm/models/llmstudio.py` |
| NIM Guardrails | `NIM_API_BASE` | `env.py:532` | `llm/utils/nemo_guardrails/` |
| Model Armor | key in env | `env.py:538` | `llm/utils/model_armor.py` |
| Bedrock | `BEDROCK_*` | `env.py:555` | `llm/utils/` |
| Fano Speech | `FANO_URL` | `config.py:2817` | `speech/fano.py` |

### Database
- **PostgreSQL** — main app DB. Env: `DATABASE_URL`. Models in `models/` (SQLAlchemy). Migrations in `migrations/versions/`
- **pgvector** — vector search. Same DB, extension. Controlled by `VECTOR_DB=pgvector`
- **MongoDB** — optional, legacy. Env: `MONGODB_OPENAI_HOST`. Activated by `DOCUMENT_DB=mongodb`
- DB factory/switching logic: `internal/document_store.py`, `repository/vector_store.py`
- Migration tool: Alembic — run from `backend/open_webui/` with `PYTHONPATH=../`

### LLM / Agent Logic
- Agent graph: `llm/agent.py`
- Tool calling: `llm/utils/tools.py`
- Guardrail logic: `llm/utils/guardrail.py`, `llm/utils/safety.py`
- Databoard AI functions: `llm/utils/databoard.py`
- Workflow trigger: `llm/utils/workflow.py`

### Configuration
- All env vars: `backend/open_webui/env.py`
- Feature config (stored in DB, editable at runtime): `backend/open_webui/config.py`
- Service host configs (DATABOARD_CONFIG, WORKFLOW_CONFIG, etc.): `config.py` bottom section (~line 2829+)
