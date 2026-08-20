import json
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import logging
from open_webui.env import (
    SRC_LOG_LEVELS,
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

class ActionOutput(BaseModel):
    """Output schema for the action parser."""
    action: str = Field(description="The action to perform based on the user prompt.")

def analyze_prompt_for_action(api_key: str, model_name: str, prompt: str, provider:str, base_url: str) -> dict:
    """
    Analyzes a user prompt using OpenAI via LangChain and returns a structured action.
    
    Args:
        api_key: OpenAI API key
        model_name: Name of the OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        prompt: User's prompt/query to analyze
        
    Returns:
        dict: A dictionary with the determined action in format {"action": "action_name"}
    """
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=ActionOutput)
    
    # Create the prompt template
    template = """
Task: Analyze the user’s input and select the single most appropriate action from:

query

get_summary

none

Guidelines:

Choose query if:

The user requests data retrieval, including:

Filtered/conditional results (e.g., “Show customers in Texas”).

Get data for data visualization (e.g., “Show me a chart of sales by month”).

Aggregated results (e.g., “Total sales in 2023,” “Average age of active users”).

Joining multiple tables or datasets.

Keywords: “filter by,” “sum,” “average,” “total,” “count,” “group by,” “show records where”.

Examples:

“List products with stock less than 100.”

“What’s the total revenue for Q4?”

Choose get_summary if:

The user asks for metadata or structural information about the dataset.

Includes: schema, column names, data types, or file properties.

Keywords: “schema,” “columns,” “data types,” “structure,” “describe the file”.

Example: “What columns are in this dataset?”

Choose none if:

The request is unrelated to data retrieval (e.g., data manipulation, explanations, exports).

Example: “How do I delete a column?”

Decision Process:

Is the user asking for data?

Yes → query (applies to both raw data and aggregated results).

No → Proceed.

Is the user asking about metadata/schema?

Yes → get_summary.

No → none.

Output: Return only the action keyword (e.g., query).
    
    User prompt: {user_prompt}
    
    """
    
    
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["user_prompt"],
    )
    
    # Format the prompt
    formatted_prompt = prompt_template.format(user_prompt=prompt)
    
    try:
        # Initialize the LLM using init_chat_model
        log.info(f"Using Ollama model: {model_name} {base_url}")
        llm = init_chat_model(
            model=f"openai:{model_name}",  # Combine provider and model
            configurable_fields = "any",
            base_url=base_url, 
            model_kwargs={
                "openai_api_key": api_key,
                "temperature": 0,
            }
        )
        
        if provider == "ollama":
            # For Ollama, we need to set the model name directly
            
            llm = init_chat_model(
                model=model_name,
                model_provider= "ollama",
                configurable_fields = "any",
                base_url=base_url, 
                model_kwargs={
                    "temperature": 0,
                    "base_url": base_url,
                    "ollama_base_url": base_url,
                }
            )
        
        # Get response
        response = llm.invoke(formatted_prompt)
        
        # Parse the response
        try:
            # First try to extract JSON from the content
            output_text = response.content
            parsed_output = parser.parse(output_text)
            log.info(f"Parsed output: {parsed_output}")
            return {"action": parsed_output.action}
        except Exception as parsing_error:
            # Fallback parsing if the model didn't return valid JSON
            # Look for keywords in the response
            output_text = response.content.lower()
            log.info(f"Raw output: {output_text}")
            if "get_summary" in output_text or "summary" in output_text:
                return {"action": "get_summary"}
            elif "query" in output_text or "query data" in output_text:
                return {"action": "query"}
            else:
                return {"action": "none"}
            
    except Exception as e:
        return {"action": "error", "error": str(e)}


class QueryOutput(BaseModel):
    """Output schema for the duckdb query."""
    duckdb_query: str = Field(description="a valid duckdb query to run on the table schemas.")    

def generate_querry(api_key: str, model_name: str, file_schemas, prompt: str, provider: str, base_url: str) -> dict:
    """
    Analyzes a user prompt and returns a DuckDB query with up to 3 retry attempts.
    
    Args:
        api_key: OpenAI API key
        model_name: Name of the model to use
        file_schemas: Schema information for the files/tables
        prompt: User's prompt/query to analyze
        provider: Model provider (e.g., "openai", "ollama")
        base_url: Base URL for the API
        
    Returns:
        dict: A dictionary with the generated DuckDB query or error information
    """
    # Create output parser
    parser = PydanticOutputParser(pydantic_object=QueryOutput)
    
    log.info("Generating DuckDB query...")
    
    # Create the base prompt template
    base_template = """
 Your task is to generate an optimized DuckDB query to create a report based on given table schemas and specific requirements. 
    Table schemas: {file_schemas}
    Requirements: {user_prompt}

    The generated Duckdb query shoule be:
    + Use correct DuckDB syntax.
    + Please generate the query without any additional text or explanation or in sql bracket.
    + If using group by, please include all non-aggregated columns in the group by clause.
    + Do not use transaction or temporary tables.
    + Do not include any explanation, break line (\n) or comments inside the script because I will copy this command into duckdb directly.
    + Please use quote for column names.
    + Applies filters, aggregations, and sorting as required.
    + Handle NULL values properly where necessary.
    + Ensure column names are correctly referenced from the table schemas.
    + Please use quote for column names;
    + If the requirement is related to calcution/aggregation, please convert the column value to numeric type safely by trimming spacing and removing any non-numeric characters before casting.
    + Please use DOUBLE type for numeric columns.
    + For columns with numeric types, please do the following instructions before aggregation:
        + Use REGEXP_REPLACE to remove all characters except digits (0-9), decimal points (.), and minus signs (-)
        + Use CAST(... AS DOUBLE) for proper aggregation.
    """
    
    # Create a template for retry attempts that includes the previous error
    retry_template = base_template + """
    
    IMPORTANT: The previous query attempt failed with the following error:
    {error_message}
    
    Please fix the issues in the query to avoid this error. Pay close attention to:
    - Column names and their correct syntax (use quotes appropriately)
    - Table names and their references
    - SQL syntax specific to DuckDB
    - Data type conversions that might be causing issues
    - NULL handling that might be problematic
    """
    
    file_schemas = json.dumps(file_schemas)
    # Format the prompt
    log.info(f"User prompt: {prompt}")
    
    max_retries = 3
    retry_count = 0
    last_error = None
    last_query = None
    
    # Retry loop
    while retry_count < max_retries:
        try:
            # For first attempt use base template, for retries use template with error context
            if retry_count == 0:
                template = base_template
                formatted_prompt = PromptTemplate(
                    template=template,
                    input_variables=["user_prompt", "file_schemas"]
                ).format(user_prompt=prompt, file_schemas=file_schemas)
            else:
                template = retry_template
                formatted_prompt = PromptTemplate(
                    template=template,
                    input_variables=["user_prompt", "file_schemas", "error_message"]
                ).format(
                    user_prompt=prompt, 
                    file_schemas=file_schemas, 
                    error_message=f"{last_error}\n\nPrevious failed query: {last_query}"
                )
            
            # Initialize the LLM using init_chat_model
            if provider == "ollama":
                log.info(f"Using Ollama model: {model_name} at {base_url}")
                llm = init_chat_model(
                    model=model_name,
                    model_provider="ollama",
                    configurable_fields="any",
                    base_url=base_url, 
                    model_kwargs={
                        "temperature": 0,
                        "base_url": base_url,
                        "ollama_base_url": base_url,
                    }
                )
            else:
                log.info(f"Using OpenAI model: {model_name}")
                llm = init_chat_model(
                    model=f"openai:{model_name}",
                    configurable_fields="any",
                    base_url=base_url, 
                    model_kwargs={
                        "openai_api_key": api_key,
                        "temperature": 0,
                    }
                )
            
            # Get response
            response = llm.invoke(formatted_prompt)
            
            # Parse the response
            try:
                # First try to extract JSON from the content
                output_text = response.content
                parsed_output = parser.parse(output_text)
                query = extract_duckdb_query(parsed_output.duckdb_query)
                last_query = query  # Store the query in case we need it for error context
                return {"duckdb_query": query}
            except Exception as parsing_error:
                # Fallback parsing if the model didn't return valid JSON
                output_text = response.content
                query = extract_duckdb_query(output_text)
                last_query = query  # Store the query in case we need it for error context
                return {"duckdb_query": query}
                
        except Exception as e:
            retry_count += 1
            last_error = str(e)
            log.warning(f"Attempt {retry_count} failed with error: {last_error}")
            
            if retry_count < max_retries:
                log.info(f"Retrying with error context... ({retry_count}/{max_retries})")
            else:
                log.error(f"All {max_retries} attempts failed. Last error: {last_error}")
                return {"action": "error", "error": f"Query generation failed after {max_retries} attempts. Last error: {last_error}", "last_query": last_query}
    
    # This should never be reached due to the return in the else clause above,
    # but adding it for completeness
    return {"action": "error", "error": f"Query generation failed after {max_retries} attempts. Last error: {last_error}", "last_query": last_query}

def extract_duckdb_query(response_text):
    """Extract DuckDB query from a string that may contain code blocks."""
    # Remove <think> sections if present
    if "<think>" in response_text and "</think>" in response_text:
        think_start = response_text.find("<think>")
        think_end = response_text.find("</think>") + len("</think>")
        response_text = response_text[:think_start] + response_text[think_end:].strip()
   
    # Check if the text contains SQL code blocks
    if "```sql" in response_text:
        # Extract the query between ```sql and ``` markers
        start_marker = "```sql\n"
        end_marker = "\n```"
        
        start_idx = response_text.find(start_marker) + len(start_marker)
        end_idx = response_text.find(end_marker, start_idx)
        
        if end_idx != -1:
            return response_text[start_idx:end_idx]
    
    # If no code block found or extraction failed, return the original text
    return response_text