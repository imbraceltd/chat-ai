import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from open_webui.repository.echart import echart_repo
from open_webui.llm.utils.echart import generate_echart


logger = logging.getLogger(__name__)

router = APIRouter()


class GenerateEchartRequest(BaseModel):
    question: str = Field(...,
                          description="User question for the visualization")
    file_content: Optional[str] = Field(
        None, description="File content or raw data to visualize"
    )
    thread_id: str = Field(..., description="Thread identifier")
    echart_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Pre-generated ECharts config from the caller agent. When provided, LLM generation is skipped and this config is saved directly.",
    )


class GenerateEchartResponse(BaseModel):
    echart: Optional[Dict[str, Any]]
    echart_id: Optional[str]
    thread_id: str
    error: Optional[str] = None


async def save_echart_to_history(
    thread_id: str, echart_data: Dict[str, Any], organization_id: str = ""
) -> Optional[str]:
    """
    Save echart data to the relational `echarts` table and return the generated ID.

    Delegates to `EchartRepository`, the same backend used by the reader
    (`get_echarts_by_thread_id`). The previous implementation called
    `mongodb_client.get_client()`, which only exists on the Mongo client —
    under the PostgreSQL document client it raised AttributeError, so echarts
    were generated but never persisted.
    """
    try:
        echart_id = await echart_repo.save(
            thread_id, echart_data, organization_id=organization_id
        )
        logger.info(f"Saved echart with ID {echart_id} for thread {thread_id}")
        return echart_id
    except Exception as e:
        logger.error(f"Error saving echart to history: {e}")
        return None


@router.post(
    "/echart",
    response_model=GenerateEchartResponse,
    summary="Generate and store an EChart configuration",
    tags=["agent"],
)
async def generate_and_save_echart(
    payload: GenerateEchartRequest,
    x_organization_id: Optional[str] = Header(
        default="",
        alias="x-organization-id",
        description="Organization identifier taken from request headers",
    ),
) -> GenerateEchartResponse:
    """
    Generate an Apache ECharts configuration from `question` and `file_content`,
    save it to MongoDB when applicable, and return `{ echart, echart_id, thread_id }`.

    When `echart_config` is provided by the caller agent, LLM generation is
    skipped entirely and the config is saved directly — faster and more reliable.

    This endpoint is intentionally unauthenticated.
    """
    # Fast path: caller agent already generated the echart config
    if payload.echart_config and isinstance(payload.echart_config, dict):
        logger.info("Using agent-generated echart config (skipping LLM)")
        organization_id = x_organization_id or ""
        echart_id = await save_echart_to_history(
            payload.thread_id,
            payload.echart_config,
            organization_id=organization_id,
        )
        return GenerateEchartResponse(
            echart=payload.echart_config,
            echart_id=echart_id,
            thread_id=payload.thread_id,
        )

    # Fallback: generate echart config via LLM
    try:
        echart_result = await generate_echart(payload.question, payload.file_content)
    except Exception as e:
        logger.error(f"Error generating echart: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate echart")

    if isinstance(echart_result, dict):
        organization_id = x_organization_id or ""
        echart_id = await save_echart_to_history(
            payload.thread_id,
            echart_result,
            organization_id=organization_id,
        )
        return GenerateEchartResponse(
            echart=echart_result,
            echart_id=echart_id,
            thread_id=payload.thread_id,
        )

    # "no echart required" or any non-dict response
    return GenerateEchartResponse(
        echart=None,
        echart_id=None,
        thread_id=payload.thread_id,
        error="EChart generation failed. Please retry again. If the problem still persists, please consider updating the tool input and retry",
    )
