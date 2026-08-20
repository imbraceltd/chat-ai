import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from open_webui.config import OPENAI_API_KEY
from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("TABULAR_DESCRIPTION", logging.INFO))


async def generate_tabular_description(
    df: pd.DataFrame, file_name: str, max_samples: int = 5
) -> str:
    """
    Generate an AI description for tabular data using sample records.

    Args:
        df: Pandas DataFrame containing the tabular data
        file_name: Name of the file
        max_samples: Maximum number of sample rows to include in the prompt

    Returns:
        AI-generated description of the tabular data
    """
    try:
        # Get basic info
        num_rows = len(df)
        num_cols = len(df.columns)
        columns = df.columns.tolist()

        # Get sample data (first few rows)
        sample_rows = min(max_samples, len(df))
        sample_data = df.head(sample_rows).to_dict("records")

        # Create the prompt
        prompt = f"""
Analyze this tabular dataset and provide a brief description in exactly 3 clear sentences.

File: {file_name}
Columns: {', '.join(columns)}
Rows: {num_rows}

Sample data (first {sample_rows} rows):
"""

        # Add sample data in a readable format
        for i, row in enumerate(sample_data[:3]):  # Show only first 3 rows for brevity
            prompt += f"\nRow {i+1}: {row}"

        prompt += """

Write exactly 3 sentences describing:
1. What type of data this is
2. Key columns and their purpose
3. How this data could be useful for queries
"""

        # Build the description LLM per LLM_DEFAULT_PROVIDER (override with
        # TABULAR_DESC_PROVIDER / TABULAR_DESC_MODEL).
        from open_webui.llm.utils.task_llm import build_task_llm

        llm = build_task_llm(
            "TABULAR_DESC",
            default_openai_model="gpt-4o-mini",  # cost-effective default on openai
            temperature=0.1,
            max_tokens=500,
        )

        # Create message
        message = HumanMessage(content=prompt)

        # Get response
        response = await llm.ainvoke([message])

        description = response.content.strip()

        log.info(f"Generated description for {file_name}: {description[:200]}...")

        return description

    except Exception as error:
        log.error(f"Error generating tabular description for {file_name}: {error}")

        # Fallback description
        fallback_desc = f"Tabular data from {file_name} with {len(df)} rows and {len(df.columns)} columns: {', '.join(df.columns.tolist())}"

        log.info(f"Using fallback description: {fallback_desc}")
        return fallback_desc


async def generate_tabular_description_batch(
    df: pd.DataFrame, file_name: str, batch_size: int = 10
) -> str:
    """
    Generate description for large datasets by sampling in batches.

    Args:
        df: Pandas DataFrame
        file_name: Name of the file
        batch_size: Number of rows to sample from different parts of the dataset

    Returns:
        AI-generated description
    """
    try:
        if len(df) <= 10:
            # Small dataset, use all data
            return await generate_tabular_description(
                df, file_name, max_samples=len(df)
            )

        # Large dataset, sample from different parts
        sample_indices = []

        # Get first few rows
        sample_indices.extend(range(min(3, len(df))))

        # Get some middle rows
        middle_start = len(df) // 2
        sample_indices.extend(range(middle_start, min(middle_start + 3, len(df))))

        # Get some last rows
        last_start = max(0, len(df) - 3)
        sample_indices.extend(range(last_start, len(df)))

        # Remove duplicates and sort
        sample_indices = sorted(list(set(sample_indices)))

        # Sample the data
        sample_df = df.iloc[sample_indices]

        return await generate_tabular_description(
            sample_df, file_name, max_samples=len(sample_df)
        )

    except Exception as error:
        log.error(f"Error in batch description generation for {file_name}: {error}")
        # Fallback to regular method
        return await generate_tabular_description(df, file_name)


# Export the main function
__all__ = [
    "generate_tabular_description",
    "generate_tabular_description_batch",
]
