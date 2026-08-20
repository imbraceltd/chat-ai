import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI # Example LLM
import base64
import httpx
import asyncio
from typing import Tuple    
import base64
import fitz 
import os
from typing import List, Optional
from typing import Dict, Any, List, Optional
from json_repair import repair_json
from open_webui.constants import MODEL_LIST
# from open_webui.utils.databoard import  fetch_url_as_base64_string, pdf_buffer_to_png_base64


MIME_TYPE_MAP = {
    "application/pdf": "pdf",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    # Add any other specific types you need to support here
}

async def fetch_url_as_base64_string(url: str) -> Tuple[str, str]:
    """Fetches a file and returns its extension and base64 content."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower().split(";")[0].strip()
            file_extension = None

            if content_type.startswith("image/"):
                file_extension = content_type.split("/")[-1].strip()
            elif content_type in MIME_TYPE_MAP:
                file_extension = MIME_TYPE_MAP[content_type]
            elif content_type in ("application/octet-stream", "binary/octet-stream", ""):
                # Fallback: detect file type from URL extension (common with Google Drive, S3, etc.)
                from urllib.parse import urlparse
                url_path = urlparse(url).path.lower()
                ext_map = {
                    ".pdf": "pdf",
                    ".docx": "docx",
                    ".doc": "docx",
                    ".xls": "xls",
                    ".xlsx": "xlsx",
                    ".png": "png",
                    ".jpg": "jpeg",
                    ".jpeg": "jpeg",
                    ".gif": "gif",
                    ".bmp": "bmp",
                    ".tiff": "tiff",
                    ".tif": "tiff",
                    ".webp": "webp",
                }
                for ext_suffix, ext_type in ext_map.items():
                    if url_path.endswith(ext_suffix):
                        file_extension = ext_type
                        break

                if not file_extension:
                    # Last resort: try to detect from file magic bytes
                    content_bytes = response.content[:8]
                    if content_bytes[:4] == b'%PDF':
                        file_extension = "pdf"
                    elif content_bytes[:4] == b'PK\x03\x04':
                        # ZIP-based format (docx, xlsx) — check for docx markers
                        file_extension = "docx"
                    elif content_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                        file_extension = "png"
                    elif content_bytes[:2] in (b'\xff\xd8',):
                        file_extension = "jpeg"

                if not file_extension:
                    raise ValueError(
                        f"Unsupported file format: {content_type}. "
                        "Could not determine file type from URL or file content. "
                        "Please ensure the URL includes a file extension (e.g., .pdf, .docx)."
                    )
            else:
                raise ValueError(f"Unsupported file format: {content_type}")

            base64_string = base64.b64encode(response.content).decode('utf-8')
            print(f"Content-Type: {content_type}, Determined Extension: {file_extension}")
            return file_extension, base64_string

    except httpx.RequestError as e:
        raise ConnectionError(f"Failed to fetch URL: {e}")
    except Exception as e:
        raise e


def _pdf_base64_to_png_base64_sync(
    base64_string: str,
    page_count: Optional[int] = None,
    dpi: int = 150
) -> List[str]:
    """Converts a Base64 PDF into a list of Base64 PNGs. This is a synchronous, CPU-bound function."""
    try:
        pdf_bytes = base64.b64decode(base64_string)
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        # If page_count is None, process ALL pages. Otherwise, enforce the limit.
        total_pages = len(pdf_document)
        loops = page_count if page_count else total_pages

        base64_list = []

        print(f"Processing PDF with {total_pages} pages. Target: {loops} pages. DPI: {dpi}")

        for page_num, page in enumerate(pdf_document):
            if page_num >= loops:
                print(f"Reached limit of {loops} pages. Stopping.")
                break

            pix = page.get_pixmap(dpi=dpi)
            img_bytes = pix.tobytes("png")
            base64_list.append(base64.b64encode(img_bytes).decode('utf-8'))

        print(f"Converted {len(base64_list)} pages successfully.")
        return base64_list
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")
        raise e

async def pdf_base64_to_png_base64(
    base64_string: str,
    page_count: Optional[int] = None
) -> List[str]:
    """Async wrapper for the synchronous PDF conversion."""
    return await asyncio.to_thread(_pdf_base64_to_png_base64_sync, base64_string, page_count)


def _docx_base64_to_text_sync(base64_string: str) -> str:
    """Convert a Base64-encoded DOCX into plain text via docx2txt.

    Synchronous, CPU-bound. Returns empty string for empty documents.
    Raises ValueError for invalid/corrupted/encrypted DOCX or unsupported
    legacy binary .doc files.
    """
    import docx2txt
    import io

    try:
        docx_bytes = base64.b64decode(base64_string)
        text = docx2txt.process(io.BytesIO(docx_bytes)) or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        print(f"DOCX extraction: {len(text)} chars, ~{len(text.splitlines())} lines")
        return text
    except Exception as e:
        print(f"An error occurred during DOCX processing: {e}")
        raise ValueError(f"Failed to extract text from DOCX: {e}") from e


async def docx_base64_to_text(base64_string: str) -> str:
    """Async wrapper for the synchronous DOCX text extraction."""
    return await asyncio.to_thread(_docx_base64_to_text_sync, base64_string)

def get_field_details(output_schema: dict) -> list:
  
    properties = output_schema.get("properties", {})
    details = [
        {
            "fieldName": key,
            "defaultValue": properties.get(key, {}).get("default", ""),
            "description": properties.get(key, {}).get("description", ""),
            "fieldSchema": {
                **output_schema,
                "properties": {key: properties.get(key)},
            },
        }
        for key in properties.keys()
    ]
    return details

import json

def create_prompt(
    model_name: str,
    url: str,
    output_schema: dict,
    field_details: list
) -> str:
    for field_prop in output_schema.get("properties", {}).values():
        field_prop.pop("example", None) # Safely remove the key if it exists

    tasks_in_string = ",".join(detail["fieldName"] for detail in field_details)

    example_fragments = []
    for detail in field_details:
        field_name = detail["fieldName"]
        example_value = output_schema.get("properties", {}).get(field_name, {}).get("example", f'"{field_name}_example"')
        example_fragments.append(f'"{field_name}": {example_value}')
    
    example_string = ",".join(example_fragments)

    prompt = (
        f"Given {{'modelName': '{model_name}', 'fileUrl': '{url}'}}, "
        f"your task is to analyze the content of the file and extract specific information({tasks_in_string}). "
        f"Then, construct a JSON object adhering to the provided schema({json.dumps(output_schema)}). "
        f"Output Instructions: Please construct a JSON object with the extracted information. "
        f"Make sure the JSON follows the provided schema structure and includes all required fields. "
        f"If any information cannot be confidently extracted, set 'Request Human' and 'ReQuEsT_HuMaN' to true and "
        f"provide a detailed comment in 'SaY_To_HuMaN' explaining the problem. "
        f"Example Output Structure: ```json{{{example_string}}}```"
    )
    return prompt

def get_system_message_direct(
    model_name: str | None,
    url: str,
    language: str | None,
    additional_instructions: str | None,
) -> str:
    """
    Generates a system message by replacing placeholders in a template string.
    This is a direct translation of the JavaScript logic.
    """
    
    # The ternary operator `condition ? true_val : false_val` is written
    # `true_val if condition else false_val` in Python.
    additional_text = (
        f" Additionally, here are the instructions from the user: {additional_instructions}"
        if additional_instructions
        else ""
    )

    message = (
        """You are an AI assistant tasked with converting input documents into JSON format based on a specified output schema. Please refrain from including any introductions or summaries, and provide your response in <language>. Your objective is to accurately convert the provided input into the required output format as per the given schema. For fields that need a file URL, use <url>, and for those that ask for the AI model name, respond with <modelName>.
            If there is instruction that is requested for comment and recommendations, please follow the rules and provide the answer of the comment and recommendation, its value should match that of the 'SaY_To_HuMaN' field.
            <additionalInstructions>"""
        .replace("<modelName>", model_name or "undefined")
        .replace("<url>", url)
        .replace("<language>", language or "English")
        .replace("<additionalInstructions>", additional_text)
    )
    return message

async def vision_assistant(
    model,
    model_name: str,
    url: str,
    language: str,
    additional_instructions: str,
    output_schema: dict
) -> dict:
    """
    Orchestrates a call to a vision-capable AI model using LangChain.
    """
    # 1. Bind the model to force JSON output
    json_model = model.bind(
        response_format={"type": "json_object"}
    )
    file_extension, base64_content = await fetch_url_as_base64_string(url)
    print(f"Fetched file with extension: {file_extension}")

    
    if file_extension == "pdf":
        print("Detected PDF file, converting pages to PNG images...")
        # FIX #1: Run the synchronous, blocking function in a separate thread
        base64_content = await pdf_base64_to_png_base64(base64_content)
        file_extension = "png"

    print(f"Base64 content is present: {base64_content}")
    
    system_message_content = get_system_message_direct(
        model_name=model_name,
        url=url,
        language=language,
        additional_instructions=additional_instructions
    )
    print(f"System Message: {system_message_content}")

    # 4. Construct the HumanMessage content with text and image(s)
    human_message_content = [
        {
            "type": "text",
            "text": f"Extract data based on the file, return a JSON with outputSchema: {json.dumps(output_schema)}",
        }
    ]

    # This logic handles both single images and multiple images (from PDF pages)
    image_list = base64_content if isinstance(base64_content, list) else [base64_content]
    
    for item in image_list:
        human_message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{file_extension};base64,{item}"}
        })
        
    # 5. Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message_content),
        HumanMessage(content=human_message_content),
    ])

    print(f"Extract data based on the file, return a JSON with outputSchema: {json.dumps(output_schema)}")

    chain = prompt | json_model
    
    print("Invoking AI model chain...")
    response = chain.invoke({})

    print(f"Vision Assistant Response: {response}")

    if hasattr(response, 'content'):
        return json.loads(response.content)
    else:
  
        return response


async def openai_assistant(
    model_name: str,
    url: str,
    language: Optional[str],
    additional_instructions: Optional[str],
    output_schema: Dict[str, Any],
) -> Dict[str, Any]:
    request_url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        # 1. Fetch file and convert to Base64
        file_extension, base64_content = await fetch_url_as_base64_string(url)

        # 2. If it's a PDF, convert pages to PNG images
        if file_extension.lower() == "pdf":
            base64_content = await pdf_base64_to_png_base64(base64_content)
            file_extension = "png" # The content is now PNG

        # 3. Construct the message content list (text + image(s))
        message_content = [
            {
                "type": "text",
                "text": f"Given {{'modelName': '{model_name}', 'fileUrl': '{url}'}}, your task is to analyze the content of the file and extract specific information. Then, construct a JSON object adhering to the provided schema.",
            }
        ]
        
        image_list = base64_content if isinstance(base64_content, list) else [base64_content]
        for item in image_list:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/{file_extension};base64,{item}"},
            })

        # 4. Construct the full request payload
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": output_schema, "name": "outputSchema"},
            },
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                request_url,
                headers=headers,
                content=json.dumps(payload) # Send the raw JSON string body
            )
            response.raise_for_status()

        # 6. Parse the response
        data = response.json()
        print(f"Choice from API: {data.get('choices', [{}])[0]}")

        content_string = data.get("choices", [{}])[0].get("message", {}).get("content")
        if not content_string:
            raise ValueError("No content found in OpenAI API response.")
            
        json_object = json.loads(content_string)
        print(f"Parsed JSON Object: {json_object}")

        return json_object

    except httpx.RequestError as e:
        raise ConnectionError(f"HTTP request to OpenAI failed: {e}")
    except Exception as e:
        print(f"An error occurred in openai_assistant: {e}")
        raise e

async def lmdeploy_assistant(
    model_name: str,
    url: str,
    language: Optional[str],
    additional_instructions: Optional[str],
    output_schema: Dict[str, Any],
    tools: Optional[List[Any]]
) -> Dict[str, Any]:
    """
    Communicates with a self-hosted LMDeploy service to perform document AI tasks.

    Args:
        model_name: The name of the LMDeploy model to use.
        url: The URL of the document/image to process.
        language: (Included to match signature) The language of the document.
        additional_instructions: (Included to match signature) Extra instructions.
        output_schema: The desired JSON schema for the output.
        tools: (Included to match signature) Any tools for the model.

    Returns:
        A dictionary containing the structured data extracted by the model.
    """
    # 1. Construct the request URL from environment variables with a fallback
    base_url = os.environ.get("ON_PREMISE_OPENVL_BASE_URL", "http://localhost:23333")
    request_url = f"{base_url}/v1/chat/interactive"
    
    headers = {"Content-Type": "application/json"}

    # 2. Prepare the prompt using helper functions
    field_details = get_field_details(output_schema)
    prompt = create_prompt(
        model_name=model_name,
        url=url,
        output_schema=output_schema,
        field_details=field_details
    )
    print(f"Generated Prompt: {prompt}")

    # 3. Construct the request body payload
    # `null` in JavaScript is `None` in Python
    payload = {
        "prompt": prompt,
        "image_url": url,
        "session_id": -1,
        "interactive_mode": False,
        "stream": False,
        "stop": None,
        "request_output_len": None,
        "top_p": 0.8,
        "top_k": 40,
        "temperature": 0.8,
        "repetition_penalty": 1,
        "ignore_eos": False,
        "skip_special_tokens": True,
        "cancel": False,
        "adapter_name": None,
        "seed": 0,
        "min_new_tokens": None,
        "min_p": 0,
    }

    try:
        # 4. Make the asynchronous API call using httpx
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                request_url,
                headers=headers,
                content=json.dumps(payload)
            )
            response.raise_for_status() # Raise exception for 4xx/5xx errors

        data = response.json()
        print(f"Raw API Response Data: {data}")

        # 5. Clean, repair, and parse the JSON response
        text_content = data.get("text", "")
        
        # Chain .replace() and .strip() to clean the string
        cleaned_json_string = (
            text_content.replace("```json", "")
            .replace("```", "")
            .replace("\r\n", "")
            .replace("\n", "")
            .replace("\r", "")
            .strip()
        )
        
        # Use the jsonrepair library to fix any malformed JSON
        repaired_string = repair_json(cleaned_json_string)
        
        # Parse the repaired string into a Python dictionary
        json_object = json.loads(repaired_string)
        print(f"Repaired and Parsed JSON Object: {json_object}")

        return json_object

    except httpx.RequestError as e:
        raise ConnectionError(f"HTTP request to LMDeploy failed: {e}")
    except Exception as e:
        print(f"An error occurred in lmdeploy_assistant: {e}")
        raise e
    
async def get_models():
    try:
        env = os.environ.get("ENV")
        models = []
        if env in ["dev3", "aistg3", "nokia"]:
        # Equivalent to spreading arrays with `...`
            print(f"Environment '{env}' detected. Returning a subset of models.")
            models = (
                MODEL_LIST["OllamaModels"]
                + MODEL_LIST["NokiaModels"]
                + MODEL_LIST["LmdeployModels"]
            )
        else:
            print("No specific environment detected. Returning all models.")
            models = [model for model_list in MODEL_LIST.values() for model in model_list]
          
        vision_models = [
            model for model in models if model.get("is_vision_available") is True
        ]
        return {
            "success": True,
            "data": {"models": vision_models},
        }
    except Exception as e:
        print(f"An error occurred while fetching models: {e}")
        raise e