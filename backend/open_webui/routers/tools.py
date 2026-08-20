import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import time

import black

from open_webui.tools.main import create_simple_tool_list
from open_webui.models.tools import (
    ToolForm,
    ToolMeta,
    ToolModel,
    ToolResponse,
    ToolUserResponse,
    Tools,
)
from open_webui.utils.plugin import load_tool_module_by_id, replace_imports
from open_webui.config import CACHE_DIR
from open_webui.constants import ERROR_MESSAGES
from fastapi import APIRouter, Depends, HTTPException, Request, status
from open_webui.utils.tools import (
    get_tool_specs,
    get_tool_servers_data,
    convert_openapi_to_tool_payload,
)
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.access_control import has_access, has_permission
from open_webui.env import SRC_LOG_LEVELS

from pydantic import BaseModel, ConfigDict
from open_webui.utils.mcp.client import MCPClient
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA
import aiohttp
import yaml
import json

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


router = APIRouter()

############################
# GetTools
############################


@router.get("/", response_model=list[ToolUserResponse])
async def get_tools(request: Request, user=Depends(get_verified_user)):

    if not request.app.state.TOOL_SERVERS:
        # If the tool servers are not set, we need to set them
        # This is done only once when the server starts
        # This is done to avoid loading the tool servers every time

        request.app.state.TOOL_SERVERS = await get_tool_servers_data(
            request.app.state.config.TOOL_SERVER_CONNECTIONS
        )

    tools = Tools.get_tools()
    for server in request.app.state.TOOL_SERVERS:
        tools.append(
            ToolUserResponse(
                **{
                    "id": f"server:{server['idx']}",
                    "user_id": f"server:{server['idx']}",
                    "name": server["openapi"]
                    .get("info", {})
                    .get("title", "Tool Server"),
                    "meta": {
                        "description": server["openapi"]
                        .get("info", {})
                        .get("description", ""),
                    },
                    "access_control": request.app.state.config.TOOL_SERVER_CONNECTIONS[
                        server["idx"]
                    ]
                    .get("config", {})
                    .get("access_control", None),
                    "updated_at": int(time.time()),
                    "created_at": int(time.time()),
                }
            )
        )

    if user.role != "admin":
        tools = [
            tool
            for tool in tools
            if tool.user_id == user.id
            or has_access(user.id, "read", tool.access_control)
        ]

    return tools


############################
# GetToolsByConfig
############################


class ToolServerConfigInput(BaseModel):
    url: str
    path: Optional[str] = None
    auth_type: Optional[str] = "bearer"
    key: Optional[str] = None
    type: Optional[str] = "openapi"
    enabled: Optional[bool] = True
    headers: Optional[Dict[str, str]] = None

    model_config = ConfigDict(extra="allow")


class ToolSummary(BaseModel):
    name: str
    description: Optional[str] = ""


@router.post("/config/tools", response_model=List[ToolSummary])
async def get_tools_by_config(
    request: Request,
    config: ToolServerConfigInput,
):
    if config.enabled is False:
        return []

    raw_url = config.url or ""
    server_type = (config.type or "openapi").lower()

    if server_type == "mcp":
        base_url = raw_url  # keep trailing slash if provided to avoid MCP redirects
    else:
        base_url = raw_url.rstrip("/")
    if not base_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tool server url is required",
        )

    raw_path = config.path or ("" if server_type == "mcp" else "openapi.json")
    normalized_path = raw_path.lstrip("/") if raw_path else ""

    if server_type == "mcp":
        fetch_url = base_url
    else:
        fetch_url = f"{base_url}/{normalized_path}" if normalized_path else base_url

    token = ""
    auth_type = (config.auth_type or "").lower()
    if auth_type == "bearer":
        token = config.key or ""
    elif auth_type == "session":
        state_token = getattr(request.state, "token", None)
        token = getattr(state_token, "credentials", "") if state_token else ""

    headers = config.headers or {}

    request_headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if headers:
        request_headers.update(headers)
    if token and "Authorization" not in request_headers:
        request_headers["Authorization"] = f"Bearer {token}"

    if server_type == "mcp":
        client = MCPClient()
        try:
            await client.connect(url=fetch_url, headers=request_headers or None)
            raw_specs = await client.list_tool_specs() or []
        except Exception as err:  # pragma: no cover - depends on external server
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(err),
            ) from err
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

        results: List[ToolSummary] = []
        for spec in raw_specs:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name")
            if not name:
                continue
            results.append(
                ToolSummary(
                    name=name,
                    description=spec.get("description", ""),
                )
            )
        return results

    # Default to OpenAPI-compatible servers
    try:
        timeout = aiohttp.ClientTimeout(
            total=AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA
        )
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(fetch_url, headers=request_headers) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"HTTP {response.status} while fetching tool server spec: {error_body}",
                    )

                text_content = await response.text()
    except HTTPException:
        raise
    except Exception as err:  # pragma: no cover - depends on external server
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        ) from err

    try:
        try:
            openapi_spec: Dict[str, Any] = json.loads(text_content)
        except json.JSONDecodeError:
            openapi_spec = yaml.safe_load(text_content) or {}
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse tool server spec: {err}",
        ) from err

    specs = convert_openapi_to_tool_payload(openapi_spec)

    results: List[ToolSummary] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not name:
            continue
        results.append(
            ToolSummary(
                name=name,
                description=spec.get("description", ""),
            )
        )

    return results


############################
# GetToolList
############################


@router.get("/list", response_model=list[ToolUserResponse])
async def get_tool_list(user=Depends(get_verified_user)):
    if user.role == "admin":
        tools = Tools.get_tools()
    else:
        tools = Tools.get_tools_by_user_id(user.id, "write")
    return tools


############################
# ExportTools
############################


@router.get("/export", response_model=list[ToolModel])
async def export_tools(user=Depends(get_admin_user)):
    tools = Tools.get_tools()
    return tools


############################
# CreateNewTools
############################


@router.post("/create", response_model=Optional[ToolResponse])
async def create_new_tools(
    request: Request,
    form_data: ToolForm,
    user=Depends(get_verified_user),
):
    if user.role != "admin" and not has_permission(
        user.id, "workspace.tools", request.app.state.config.USER_PERMISSIONS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    if not form_data.id.isidentifier():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only alphanumeric characters and underscores are allowed in the id",
        )

    form_data.id = form_data.id.lower()

    tools = Tools.get_tool_by_id(form_data.id)
    if tools is None:
        try:
            form_data.content = replace_imports(form_data.content)
            tool_module, frontmatter = load_tool_module_by_id(
                form_data.id, content=form_data.content
            )
            form_data.meta.manifest = frontmatter

            TOOLS = request.app.state.TOOLS
            TOOLS[form_data.id] = tool_module
            
            specs = get_tool_specs(TOOLS[form_data.id])
            tools = Tools.insert_new_tool(user.id, form_data, specs)

            tool_cache_dir = CACHE_DIR / "tools" / form_data.id
            tool_cache_dir.mkdir(parents=True, exist_ok=True)

            if tools:
                return tools
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT("Error creating tools"),
                )
        except Exception as e:
            log.exception(f"Failed to load the tool by id {form_data.id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ID_TAKEN,
        )


############################
# GetToolsById
############################


@router.get("/id/{id}", response_model=Optional[ToolModel])
async def get_tools_by_id(id: str, user=Depends(get_verified_user)):
    tools = Tools.get_tool_by_id(id)

    if tools:
        if (
            user.role == "admin"
            or tools.user_id == user.id
            or has_access(user.id, "read", tools.access_control)
        ):
            return tools
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdateToolsById
############################


@router.post("/id/{id}/update", response_model=Optional[ToolModel])
async def update_tools_by_id(
    request: Request,
    id: str,
    form_data: ToolForm,
    user=Depends(get_verified_user),
):
    tools = Tools.get_tool_by_id(id)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    # Is the user the original creator, in a group with write access, or an admin
    if (
        tools.user_id != user.id
        and not has_access(user.id, "write", tools.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    try:
        form_data.content = replace_imports(form_data.content)
        tool_module, frontmatter = load_tool_module_by_id(id, content=form_data.content)
        form_data.meta.manifest = frontmatter

        TOOLS = request.app.state.TOOLS
        TOOLS[id] = tool_module

        specs = get_tool_specs(TOOLS[id])

        updated = {
            **form_data.model_dump(exclude={"id"}),
            "specs": specs,
        }

        log.debug(updated)
        tools = Tools.update_tool_by_id(id, updated)

        if tools:
            return tools
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT("Error updating tools"),
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# DeleteToolsById
############################


@router.delete("/id/{id}/delete", response_model=bool)
async def delete_tools_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not has_access(user.id, "write", tools.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.UNAUTHORIZED,
        )

    result = Tools.delete_tool_by_id(id)
    if result:
        TOOLS = request.app.state.TOOLS
        if id in TOOLS:
            del TOOLS[id]

    return result


############################
# GetToolValves
############################


@router.get("/id/{id}/valves", response_model=Optional[dict])
async def get_tools_valves_by_id(id: str, user=Depends(get_verified_user)):
    tools = Tools.get_tool_by_id(id)
    if tools:
        try:
            valves = Tools.get_tool_valves_by_id(id)
            return valves
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# GetToolValvesSpec
############################


@router.get("/id/{id}/valves/spec", response_model=Optional[dict])
async def get_tools_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if tools:
        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "Valves"):
            Valves = tools_module.Valves
            return Valves.schema()
        return None
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


############################
# UpdateToolValves
############################


@router.post("/id/{id}/valves/update", response_model=Optional[dict])
async def update_tools_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if not tools:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if (
        tools.user_id != user.id
        and not has_access(user.id, "write", tools.access_control)
        and user.role != "admin"
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    if id in request.app.state.TOOLS:
        tools_module = request.app.state.TOOLS[id]
    else:
        tools_module, _ = load_tool_module_by_id(id)
        request.app.state.TOOLS[id] = tools_module

    if not hasattr(tools_module, "Valves"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )
    Valves = tools_module.Valves

    try:
        form_data = {k: v for k, v in form_data.items() if v is not None}
        valves = Valves(**form_data)
        Tools.update_tool_valves_by_id(id, valves.model_dump())
        return valves.model_dump()
    except Exception as e:
        log.exception(f"Failed to update tool valves by id {id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(str(e)),
        )


############################
# ToolUserValves
############################


@router.get("/id/{id}/valves/user", response_model=Optional[dict])
async def get_tools_user_valves_by_id(id: str, user=Depends(get_verified_user)):
    tools = Tools.get_tool_by_id(id)
    if tools:
        try:
            user_valves = Tools.get_user_valves_by_id_and_user_id(id, user.id)
            return user_valves
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(str(e)),
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.get("/id/{id}/valves/user/spec", response_model=Optional[dict])
async def get_tools_user_valves_spec_by_id(
    request: Request, id: str, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)
    if tools:
        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "UserValves"):
            UserValves = tools_module.UserValves
            return UserValves.schema()
        return None
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )


@router.post("/id/{id}/valves/user/update", response_model=Optional[dict])
async def update_tools_user_valves_by_id(
    request: Request, id: str, form_data: dict, user=Depends(get_verified_user)
):
    tools = Tools.get_tool_by_id(id)

    if tools:
        if id in request.app.state.TOOLS:
            tools_module = request.app.state.TOOLS[id]
        else:
            tools_module, _ = load_tool_module_by_id(id)
            request.app.state.TOOLS[id] = tools_module

        if hasattr(tools_module, "UserValves"):
            UserValves = tools_module.UserValves

            try:
                form_data = {k: v for k, v in form_data.items() if v is not None}
                user_valves = UserValves(**form_data)
                Tools.update_user_valves_by_id_and_user_id(
                    id, user.id, user_valves.model_dump()
                )
                return user_valves.model_dump()
            except Exception as e:
                log.exception(f"Failed to update user valves by id {id}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=ERROR_MESSAGES.DEFAULT(str(e)),
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=ERROR_MESSAGES.NOT_FOUND,
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

def print_tools_folder_contents():
    """Print the content of all Python files in the tools folder."""
    # Get the tools directory path
    tools_dir = Path("open_webui/tools")
    
    tool_list = create_simple_tool_list()
    # log.info(f"Tools List: {tool_list}")
    
    if not tools_dir.exists():
        print(f"Tools directory not found at {tools_dir}")
        return
    
    print(f"Reading files in {tools_dir.absolute()}\n")
    print("=" * 80)
    
    # Get all Python files in the directory
    py_files = [f for f in tools_dir.glob("*.py") if f.is_file()]
    
    if not py_files:
        print("No Python files found in the tools directory.")
        return
    
    # Print the content of each file
    for file_path in py_files:
        print(f"\n\n{'=' * 80}")
        print(f"FILE: {file_path.name}")
        tool_id = file_path.stem
        print(f"FILE NAME WITHOUT EXTENSION: {file_path.stem}")
        print(f"{'=' * 80}\n")
        
        # Find the tool in the array by matching tool_id
        matching_tool = next((tool for tool in tool_list if tool['tool_id'] == tool_id), None)
        
        if matching_tool and matching_tool.get("description") is not None:
            # print(f"Tool ID: {matching_tool['tool_id']}")
            # print(f"Tool Desc - {matching_tool['description']}")
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    formatted_code = black.format_str(content, mode=black.Mode())
                    # print(formatted_code)
                    existing_tool = Tools.get_tool_by_id(tool_id)
                    if existing_tool is None:
                        try:
                            new_tool = ToolForm(
                                id=tool_id,
                                name=tool_id,
                                content=formatted_code,
                                meta=ToolMeta(description=matching_tool["description"]),
                                access_control=None
                            )
                            new_tool.content = replace_imports(new_tool.content)
                            tool_module, frontmatter = load_tool_module_by_id(
                                tool_id, content=new_tool.content
                            )
                            new_tool.meta.manifest = frontmatter
                            
                            specs = get_tool_specs(tool_module)

                            log.info(f"Before loading the tool:")
                            existing_tool = Tools.insert_new_tool("", new_tool, specs)

                            tool_cache_dir = CACHE_DIR / "tools" / tool_id
                            tool_cache_dir.mkdir(parents=True, exist_ok=True)
                            if existing_tool:
                                log.info(f"Tool {tool_id} loaded successfully.")
                        except Exception as e:
                            log.exception(f"Failed to load the tool by id {tool_id}: {e}")
                    elif existing_tool:
                        try:
                            new_tool = ToolForm(
                                    id=tool_id,
                                    name=tool_id,
                                    content=formatted_code,
                                    meta=ToolMeta(description=matching_tool["description"]),
                                    access_control=None
                                )
                            new_tool.content = replace_imports(new_tool.content)
                            tool_module, frontmatter = load_tool_module_by_id(id, content=new_tool.content)
                            new_tool.meta.manifest = frontmatter

                            specs = get_tool_specs(tool_module)

                            updated = {
                                **new_tool.model_dump(exclude={"id"}),
                                "specs": specs,
                            }

                            existing_tool = Tools.update_tool_by_id(tool_id, updated)
                            if existing_tool:
                                log.info(f"Tool {tool_id} updated successfully.")
                        except Exception as e:
                            log.exception(f"Failed to update the tool by id {tool_id}: {e}")
                            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"\n\n{'=' * 80}")
    print(f"Total files processed: {len(py_files)}")

print("Starting to read files in the tools directory...\n")
print_tools_folder_contents()