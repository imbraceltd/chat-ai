import time
import logging
import sys
import requests
import httpx
from aiocache import cached
import itertools

from typing import Dict, Any
from open_webui.routers import openai, ollama
from open_webui.functions import get_function_models


from open_webui.models.functions import Functions
from open_webui.models.models import Models


from open_webui.utils.plugin import load_function_module_by_id
from open_webui.utils.access_control import has_access

from fastapi import HTTPException, Request,status

 

from open_webui.config import (
    DEFAULT_ARENA_MODEL,
)

from open_webui.env import SRC_LOG_LEVELS, GLOBAL_LOG_LEVEL
from open_webui.models.users import UserModel

from open_webui.constants import ERROR_MESSAGES

from open_webui.env import IMBRACE_API_URL, IMBRACE_BACKNED_PRIVATE, IMBRACE_PLATFORM_HOST
from open_webui.utils import models_imbrace as models_imbrace 

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


async def get_all_base_models(request: Request, user: UserModel = None):
    function_models = []
    openai_models = []
    ollama_models = []

    if request.app.state.config.ENABLE_OPENAI_API:
        openai_models = await openai.get_all_models(request, user=user)
        openai_models = openai_models["data"]

    if request.app.state.config.ENABLE_OLLAMA_API:
        ollama_models = await ollama.get_all_models(request, user=user)
        ollama_models = [
            {
                "id": model["model"],
                "name": model["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
                "ollama": model,
                "tags": model.get("tags", []),
            }
            for model in ollama_models["models"]
        ]

    function_models = await get_function_models(request)
    models = function_models + openai_models + ollama_models

    return models


async def get_all_models(request, user: UserModel = None):
    models = await get_all_base_models(request, user=user)

    # If there are no models, return an empty list
    if len(models) == 0:
        return []

    # Add arena models
    if request.app.state.config.ENABLE_EVALUATION_ARENA_MODELS:
        arena_models = []
        if len(request.app.state.config.EVALUATION_ARENA_MODELS) > 0:
            arena_models = [
                {
                    "id": model["id"],
                    "name": model["name"],
                    "info": {
                        "meta": model["meta"],
                    },
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "arena",
                    "arena": True,
                }
                for model in request.app.state.config.EVALUATION_ARENA_MODELS
            ]
        else:
            # Add default arena model
            arena_models = [
                {
                    "id": DEFAULT_ARENA_MODEL["id"],
                    "name": DEFAULT_ARENA_MODEL["name"],
                    "info": {
                        "meta": DEFAULT_ARENA_MODEL["meta"],
                    },
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "arena",
                    "arena": True,
                }
            ]
        models = models + arena_models

    global_action_ids = [
        function.id for function in Functions.get_global_action_functions()
    ]
    enabled_action_ids = [
        function.id
        for function in Functions.get_functions_by_type("action", active_only=True)
    ]

    custom_models = Models.get_all_models()
    for custom_model in custom_models:
        if custom_model.base_model_id is None:
            for model in models:
                if custom_model.id == model["id"] or (
                    model.get("owned_by") == "ollama"
                    and custom_model.id
                    == model["id"].split(":")[
                        0
                    ]  # Ollama may return model ids in different formats (e.g., 'llama3' vs. 'llama3:7b')
                ):
                    if custom_model.is_active:
                        model["name"] = custom_model.name
                        model["info"] = custom_model.model_dump()

                        action_ids = []
                        if "info" in model and "meta" in model["info"]:
                            action_ids.extend(
                                model["info"]["meta"].get("actionIds", [])
                            )

                        model["action_ids"] = action_ids
                    else:
                        models.remove(model)

        elif custom_model.is_active and (
            custom_model.id not in [model["id"] for model in models]
        ):
            owned_by = "openai"
            pipe = None
            action_ids = []

            for model in models:
                if (
                    custom_model.base_model_id == model["id"]
                    or custom_model.base_model_id == model["id"].split(":")[0]
                ):
                    owned_by = model.get("owned_by", "unknown owner")
                    if "pipe" in model:
                        pipe = model["pipe"]
                    break

            if custom_model.meta:
                meta = custom_model.meta.model_dump()
                if "actionIds" in meta:
                    action_ids.extend(meta["actionIds"])

            models.append(
                {
                    "id": f"{custom_model.id}",
                    "name": custom_model.name,
                    "object": "model",
                    "created": custom_model.created_at,
                    "owned_by": owned_by,
                    "info": custom_model.model_dump(),
                    "preset": True,
                    **({"pipe": pipe} if pipe is not None else {}),
                    "action_ids": action_ids,
                }
            )

    # Process action_ids to get the actions
    def get_action_items_from_module(function, module):
        actions = []
        if hasattr(module, "actions"):
            actions = module.actions
            return [
                {
                    "id": f"{function.id}.{action['id']}",
                    "name": action.get("name", f"{function.name} ({action['id']})"),
                    "description": function.meta.description,
                    "icon_url": action.get(
                        "icon_url", function.meta.manifest.get("icon_url", None)
                    ),
                }
                for action in actions
            ]
        else:
            return [
                {
                    "id": function.id,
                    "name": function.name,
                    "description": function.meta.description,
                    "icon_url": function.meta.manifest.get("icon_url", None),
                }
            ]

    def get_function_module_by_id(function_id):
        if function_id in request.app.state.FUNCTIONS:
            function_module = request.app.state.FUNCTIONS[function_id]
        else:
            function_module, _, _ = load_function_module_by_id(function_id)
            request.app.state.FUNCTIONS[function_id] = function_module

    for model in models:
        action_ids = [
            action_id
            for action_id in list(set(model.pop("action_ids", []) + global_action_ids))
            if action_id in enabled_action_ids
        ]

        model["actions"] = []
        for action_id in action_ids:
            action_function = Functions.get_function_by_id(action_id)
            if action_function is None:
                raise Exception(f"Action not found: {action_id}")

            function_module = get_function_module_by_id(action_id)
            model["actions"].extend(
                get_action_items_from_module(action_function, function_module)
            )
    log.debug(f"get_all_models() returned {len(models)} models")

    request.app.state.MODELS = {model["id"]: model for model in models}
    return models


def check_model_access(user, model):
    if model.get("arena"):
        if not has_access(
            user.id,
            type="read",
            access_control=model.get("info", {})
            .get("meta", {})
            .get("access_control", {}),
        ):
            raise Exception("Model not found")
    else:
        model_info = Models.get_model_by_id(model.get("id"))
        if not model_info:
            raise Exception("Model not found")
        elif not (
            user.id == model_info.user_id
            or has_access(
                user.id, type="read", access_control=model_info.access_control
            )
        ):
            raise Exception("Model not found")

def get_agents(request:Request):
    # Get token from request header
    token = request.headers.get("X-Access-Token")
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )
    
    try:
        # Prepare headers for the API request
        headers = {
            'x-access-token': token,
            'Content-Type': 'application/json'
        }
        
        # Make request to Imbrace API with the updated endpoint
        # log.info(f"Fetching templates from Imbrace API with token: {token}")
        response = requests.get(
            f"{IMBRACE_API_URL}/v2/templates",  # Updated API endpoint
            headers=headers,
            timeout=10
        )
        
        # Handle response based on status code
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            log.error(f"Imbrace API authentication failed: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token"
            )
        else:
            log.error(f"Imbrace API request failed: Status {response.status_code}, Response: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch templates from Imbrace API: {response.text}"
            )
    except requests.RequestException as e:
        log.exception(f"Request to Imbrace API failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connection to Imbrace API failed: {str(e)}"
        )
    except Exception as e:
        log.exception(f"Unexpected error in get_agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
        
def get_agent_details(request: Request, assistant_id: str):
    """
    Fetches details of a specific assistant from the Imbrace API.
    
    Args:
        request: FastAPI request object containing headers
        assistant_id: ID of the assistant to retrieve
        
    Returns:
        dict: JSON response containing assistant details
        
    Raises:
        HTTPException: If authentication fails or API request fails
    """
    # Get token from request header
    token = request.headers.get("X-Access-Token")
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )
    
    try:
        # Prepare headers for the API request
        headers = {
            'x-access-token': token,
            'Content-Type': 'application/json'
        }
        
        # Make request to Imbrace API
        response = requests.get(
            f"{IMBRACE_API_URL}/v3/ai/assistants/{assistant_id}",
            headers=headers,
            timeout=10
        )
        
        # Handle response based on status code
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            log.error(f"Imbrace API authentication failed: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token"
            )
        elif response.status_code == 404:
            log.error(f"Assistant not found: {assistant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assistant with ID {assistant_id} not found"
            )
        else:
            log.error(f"Imbrace API request failed: Status {response.status_code}, Response: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to fetch assistant details from Imbrace API: {response.text}"
            )
    except requests.RequestException as e:
        log.exception(f"Request to Imbrace API failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connection to Imbrace API failed: {str(e)}"
        )
    except Exception as e:
        log.exception(f"Unexpected error in get_agent_details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
async def get_org_ai_settings(org_id: str) -> Dict[str, Any]:
    # platform-service replaces legacy backend; org_id passed via header.
    options = {
        "method": "GET",
        "url": f"{IMBRACE_PLATFORM_HOST}/v1/organizations/ai-settings",
        "headers": {
            "Content-Type": "application/json",
            "x-organization-id": org_id,
        },
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.request(**options)
            response.raise_for_status()  # Raise an exception for 4xx/5xx errors
            return response.json()
    except httpx.RequestError as e:
        print(f"Error calling workflow service: {e}")
        # No config available -> treat as "not configured" (allow all), distinct from [] which means "block all"
        return {}
    

async def get_models_imbrace(user_context, vision: bool = False, filter: bool = True):
    try:
        print(f"User context: {user_context}")
        model_list = await models_imbrace.get_models()
        print(f"Model list: {model_list}")
        org_id = user_context["organization_id"]
        if not org_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization ID is required.",
            )

        org_settings = await get_org_ai_settings(org_id)

        all_models_flat = model_list
        allowed_models = org_settings.get("allowed_models")

        # Path A: No filtering — only when filter is disabled OR the key is absent ("not configured").
        # An explicit [] means "block all" and must fall through to Path B.
        if not filter or allowed_models is None:
            models = []
            unique_models_dict = {model['name']: model for model in all_models_flat}
            all_unique_models = list(unique_models_dict.values())

            final_models = all_unique_models
            if vision:
                final_models = [m for m in all_unique_models if m.get("is_vision_available")]

            models.extend(final_models)
            return {"success": True, "message": "All models retrieved.", "data": models}

        # Path B: Filter based on organization's allowed models — [] blocks every model.
        models = []
        models_by_name = {m['name']: m for m in all_models_flat}

        for model_name in allowed_models:
            matching_model = models_by_name.get(model_name)

            # Check vision flag only if the query requires it
            if vision and (not matching_model or not matching_model.get("is_vision_available")):
                continue

            if matching_model:
                models.append(matching_model)
            else: # If model in allowed_models isn't in master list, add with default caps
                models.append({
                    "name": model_name,
                    "is_toolCall_available": True, # Or some other default
                    "is_vision_available": False,
                })

        return {"success": True, "message": "Models retrieved.", "data": models}

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": "Request Failed"},
        )

def update_agent_details(request: Request, assistant_id: str, data: dict):
    """
    Updates details of a specific assistant in the Imbrace API.
    
    Args:
        request: FastAPI request object containing headers
        assistant_id: ID of the assistant to update
        data: Dictionary containing updated assistant details
    Returns:
        dict: JSON response containing updated assistant details
    Raises:
        HTTPException: If authentication fails or API request fails
    """
    # Get token from request header
    token = request.headers.get("X-Access-Token")
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.INVALID_TOKEN,
        )
    
    try:
        # Prepare headers for the API request
        headers = {
            'x-access-token': token,
            'Content-Type': 'application/json'
        }
        
        # Make request to Imbrace API
        response = requests.put(
            f"{IMBRACE_API_URL}/api/v3/ai/assistant_apps/{assistant_id}",
            headers=headers,
            json=data
        )
        
        # Handle response based on status code
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            log.error(f"Imbrace API authentication failed: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token"
            )
        elif response.status_code == 404:
            log.error(f"Assistant not found: {assistant_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Assistant with ID {assistant_id} not found"
            )
        else:
            log.error(f"Imbrace API request failed: Status {response.status_code}, Response: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update assistant details in Imbrace API: {response.text}"
            )
    except requests.RequestException as e:
        log.exception(f"Request to Imbrace API failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connection to Imbrace API failed: {str(e)}"
        )
    except Exception as e:
        log.exception(f"Unexpected error in update_agent_details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
