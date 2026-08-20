import inspect
import logging
import re
import inspect
import aiohttp
import asyncio
import yaml
import json
import requests
import ffmpeg
import aiofiles
import os

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing import (
    Any,
    Awaitable,
    Callable,
    get_type_hints,
    get_args,
    get_origin,
    Dict,
    List,
    Tuple,
    Union,
    Optional,
    Type,
)
from functools import update_wrapper, partial


from fastapi import Request
from pydantic import BaseModel, Field, create_model

from langchain_core.utils.function_calling import (
    convert_to_openai_function as convert_pydantic_model_to_openai_function_spec,
)


from open_webui.models.tools import Tools
from open_webui.models.users import UserModel
from open_webui.utils.plugin import load_tool_module_by_id
from open_webui.utils.mcp.client import MCPClient
from open_webui.env import AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA

import copy

log = logging.getLogger(__name__)


def get_async_tool_function_and_apply_extra_params(
    function: Callable, extra_params: dict
) -> Callable[..., Awaitable]:
    sig = inspect.signature(function)
    extra_params = {k: v for k, v in extra_params.items() if k in sig.parameters}
    partial_func = partial(function, **extra_params)

    if inspect.iscoroutinefunction(function):
        update_wrapper(partial_func, function)
        return partial_func
    else:
        # Make it a coroutine function
        async def new_function(*args, **kwargs):
            return partial_func(*args, **kwargs)

        update_wrapper(new_function, function)
        return new_function


def get_tools(
    request: Request, tool_ids: list[str], user: UserModel, extra_params: dict
) -> dict[str, dict]:
    tools_dict = {}

    for tool_id in tool_ids:
        tool = Tools.get_tool_by_id(tool_id)
        if tool is None:
            if tool_id.startswith("server:"):
                server_idx = int(tool_id.split(":")[1])
                tool_server_connection = (
                    request.app.state.config.TOOL_SERVER_CONNECTIONS[server_idx]
                )
                tool_server_data = None
                for server in request.app.state.TOOL_SERVERS:
                    if server["idx"] == server_idx:
                        tool_server_data = server
                        break
                assert tool_server_data is not None
                specs = tool_server_data.get("specs", [])

                for spec in specs:
                    function_name = spec["name"]

                    auth_type = tool_server_connection.get("auth_type", "bearer")
                    token = None

                    if auth_type == "bearer":
                        token = tool_server_connection.get("key", "")
                    elif auth_type == "session":
                        token = request.state.token.credentials

                    def make_tool_function(function_name, token, tool_server_data):
                        async def tool_function(**kwargs):
                            print(
                                f"Executing tool function {function_name} with params: {kwargs}"
                            )
                            return await execute_tool_server(
                                token=token,
                                url=tool_server_data["url"],
                                name=function_name,
                                params=kwargs,
                                server_data=tool_server_data,
                            )

                        return tool_function

                    tool_function = make_tool_function(
                        function_name, token, tool_server_data
                    )

                    callable = get_async_tool_function_and_apply_extra_params(
                        tool_function,
                        {},
                    )

                    tool_dict = {
                        "tool_id": tool_id,
                        "callable": callable,
                        "spec": spec,
                    }

                    # TODO: if collision, prepend toolkit name
                    if function_name in tools_dict:
                        log.warning(
                            f"Tool {function_name} already exists in another tools!"
                        )
                        log.warning(f"Discarding {tool_id}.{function_name}")
                    else:
                        tools_dict[function_name] = tool_dict
            else:
                continue
        else:
            module = request.app.state.TOOLS.get(tool_id, None)
            if module is None:
                module, _ = load_tool_module_by_id(tool_id)
                request.app.state.TOOLS[tool_id] = module

            extra_params["__id__"] = tool_id

            # Set valves for the tool
            if hasattr(module, "valves") and hasattr(module, "Valves"):
                valves = Tools.get_tool_valves_by_id(tool_id) or {}
                module.valves = module.Valves(**valves)
            if hasattr(module, "UserValves"):
                extra_params["__user__"]["valves"] = module.UserValves(  # type: ignore
                    **Tools.get_user_valves_by_id_and_user_id(tool_id, user.id)
                )

            for spec in tool.specs:
                # TODO: Fix hack for OpenAI API
                # Some times breaks OpenAI but others don't. Leaving the comment
                for val in spec.get("parameters", {}).get("properties", {}).values():
                    if val["type"] == "str":
                        val["type"] = "string"

                # Remove internal reserved parameters (e.g. __id__, __user__)
                spec["parameters"]["properties"] = {
                    key: val
                    for key, val in spec["parameters"]["properties"].items()
                    if not key.startswith("__")
                }

                # convert to function that takes only model params and inserts custom params
                function_name = spec["name"]
                tool_function = getattr(module, function_name)
                callable = get_async_tool_function_and_apply_extra_params(
                    tool_function, extra_params
                )

                # TODO: Support Pydantic models as parameters
                if callable.__doc__ and callable.__doc__.strip() != "":
                    s = re.split(":(param|return)", callable.__doc__, 1)
                    spec["description"] = s[0]
                else:
                    spec["description"] = function_name

                tool_dict = {
                    "tool_id": tool_id,
                    "callable": callable,
                    "spec": spec,
                    # Misc info
                    "metadata": {
                        "file_handler": hasattr(module, "file_handler")
                        and module.file_handler,
                        "citation": hasattr(module, "citation") and module.citation,
                    },
                }

                # TODO: if collision, prepend toolkit name
                if function_name in tools_dict:
                    log.warning(
                        f"Tool {function_name} already exists in another tools!"
                    )
                    log.warning(f"Discarding {tool_id}.{function_name}")
                else:
                    tools_dict[function_name] = tool_dict

    return tools_dict


def parse_description(docstring: str | None) -> str:
    """
    Parse a function's docstring to extract the description.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        str: The description.
    """

    if not docstring:
        return ""

    lines = [line.strip() for line in docstring.strip().split("\n")]
    description_lines: list[str] = []

    for line in lines:
        if re.match(r":param", line) or re.match(r":return", line):
            break

        description_lines.append(line)

    return "\n".join(description_lines)


def parse_docstring(docstring):
    """
    Parse a function's docstring to extract parameter descriptions in reST format.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        dict: A dictionary where keys are parameter names and values are descriptions.
    """
    if not docstring:
        return {}

    # Regex to match `:param name: description` format
    param_pattern = re.compile(r":param (\w+):\s*(.+)")
    param_descriptions = {}

    for line in docstring.splitlines():
        match = param_pattern.match(line.strip())
        if not match:
            continue
        param_name, param_description = match.groups()
        if param_name.startswith("__"):
            continue
        param_descriptions[param_name] = param_description

    return param_descriptions


def convert_function_to_pydantic_model(func: Callable) -> type[BaseModel]:
    """
    Converts a Python function's type hints and docstring to a Pydantic model,
    including support for nested types, default values, and descriptions.

    Args:
        func: The function whose type hints and docstring should be converted.
        model_name: The name of the generated Pydantic model.

    Returns:
        A Pydantic model class.
    """
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    parameters = signature.parameters

    docstring = func.__doc__

    description = parse_description(docstring)
    function_descriptions = parse_docstring(docstring)

    field_defs = {}
    for name, param in parameters.items():

        type_hint = type_hints.get(name, Any)
        default_value = param.default if param.default is not param.empty else ...

        description = function_descriptions.get(name, None)

        if description:
            field_defs[name] = type_hint, Field(default_value, description=description)
        else:
            field_defs[name] = type_hint, default_value

    model = create_model(func.__name__, **field_defs)
    model.__doc__ = description

    return model


def get_functions_from_tool(tool: object) -> list[Callable]:
    return [
        getattr(tool, func)
        for func in dir(tool)
        if callable(
            getattr(tool, func)
        )  # checks if the attribute is callable (a method or function).
        and not func.startswith(
            "__"
        )  # filters out special (dunder) methods like init, str, etc. — these are usually built-in functions of an object that you might not need to use directly.
        and not inspect.isclass(
            getattr(tool, func)
        )  # ensures that the callable is not a class itself, just a method or function.
    ]


def get_tool_specs(tool_module: object) -> list[dict]:
    function_models = map(
        convert_function_to_pydantic_model, get_functions_from_tool(tool_module)
    )

    specs = [
        convert_pydantic_model_to_openai_function_spec(function_model)
        for function_model in function_models
    ]

    return specs


def resolve_schema(schema, components):
    """
    Recursively resolves a JSON schema using OpenAPI components.
    """
    if not schema:
        return {}

    if "$ref" in schema:
        ref_path = schema["$ref"]
        ref_parts = ref_path.strip("#/").split("/")
        resolved = components
        for part in ref_parts[1:]:  # Skip the initial 'components'
            resolved = resolved.get(part, {})
        return resolve_schema(resolved, components)

    resolved_schema = copy.deepcopy(schema)

    # Recursively resolve inner schemas
    if "properties" in resolved_schema:
        for prop, prop_schema in resolved_schema["properties"].items():
            resolved_schema["properties"][prop] = resolve_schema(
                prop_schema, components
            )

    if "items" in resolved_schema:
        resolved_schema["items"] = resolve_schema(resolved_schema["items"], components)

    return resolved_schema


def convert_openapi_to_tool_payload(openapi_spec):
    """
    Converts an OpenAPI specification into a custom tool payload structure.

    Args:
        openapi_spec (dict): The OpenAPI specification as a Python dict.

    Returns:
        list: A list of tool payloads.
    """
    tool_payload = []

    for path, methods in openapi_spec.get("paths", {}).items():
        for method, operation in methods.items():
            tool = {
                "type": "function",
                "name": operation.get("operationId"),
                "description": operation.get(
                    "description", operation.get("summary", "No description available.")
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            }

            # Extract path and query parameters
            for param in operation.get("parameters", []):
                param_name = param["name"]
                param_schema = param.get("schema", {})
                tool["parameters"]["properties"][param_name] = {
                    "type": param_schema.get("type"),
                    "description": param_schema.get("description", ""),
                }
                if param.get("required"):
                    tool["parameters"]["required"].append(param_name)

            # Extract and resolve requestBody if available
            request_body = operation.get("requestBody")
            if request_body:
                content = request_body.get("content", {})
                json_schema = content.get("application/json", {}).get("schema")
                if json_schema:
                    resolved_schema = resolve_schema(
                        json_schema, openapi_spec.get("components", {})
                    )

                    if resolved_schema.get("properties"):
                        tool["parameters"]["properties"].update(
                            resolved_schema["properties"]
                        )
                        if "required" in resolved_schema:
                            tool["parameters"]["required"] = list(
                                set(
                                    tool["parameters"]["required"]
                                    + resolved_schema["required"]
                                )
                            )
                    elif resolved_schema.get("type") == "array":
                        tool["parameters"] = resolved_schema  # special case for array

            tool_payload.append(tool)

    return tool_payload


async def get_tool_server_data(
    token: str,
    url: str,
    server_type: str = "openapi",
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fetch metadata/specifications for a configured tool server."""

    server_type = (server_type or "openapi").lower()
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
            await client.connect(url=url, headers=request_headers or None)
            raw_specs = await client.list_tool_specs() or []
            specs: List[Dict[str, Any]] = []
            for spec in raw_specs:
                name = spec.get("name") if isinstance(spec, dict) else None
                if not name:
                    continue
                specs.append(
                    {
                        "type": "function",
                        "name": name,
                        "description": spec.get("description", ""),
                        "parameters": spec.get("parameters", {}),
                    }
                )

            return {
                "type": "mcp",
                "url": url,
                "connection_url": url,
                "headers": request_headers,
                "specs": specs,
                "info": {},
            }
        except Exception as err:
            log.exception(f"Could not fetch MCP tool server spec from {url}")
            raise Exception(str(err))
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

    try:
        timeout = aiohttp.ClientTimeout(total=AIOHTTP_CLIENT_TIMEOUT_TOOL_SERVER_DATA)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(url, headers=request_headers) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise Exception(
                        f"HTTP {response.status} while fetching tool server spec: {error_body}"
                    )

                if url.lower().endswith((".yaml", ".yml")):
                    text_content = await response.text()
                    res = yaml.safe_load(text_content)
                else:
                    text_content = await response.text()
                    try:
                        res = json.loads(text_content)
                    except json.JSONDecodeError:
                        res = yaml.safe_load(text_content)
    except Exception as err:
        log.exception(f"Could not fetch tool server spec from {url}")
        raise Exception(str(err))

    openapi_spec = res or {}
    data = {
        "type": "openapi",
        "url": url,
        "connection_url": url,
        "headers": request_headers,
        "openapi": openapi_spec,
        "info": openapi_spec.get("info", {}),
        "specs": convert_openapi_to_tool_payload(openapi_spec),
    }

    log.debug("Fetched tool server data for %s (%s)", url, server_type)
    return data


async def get_tool_servers_data(
    servers: List[Dict[str, Any]], session_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    # Prepare list of enabled servers along with their original index
    server_entries = []
    for idx, server in enumerate(servers):
        if not server.get("config", {}).get("enable"):
            continue

        server_type = (server.get("type", "openapi") or "openapi").lower()
        raw_url = server.get("url") or ""
        base_url = raw_url if server_type == "mcp" else raw_url.rstrip("/")
        path = (server.get("path", "openapi.json") or "").lstrip("/")

        if not base_url:
            continue

        full_url = base_url
        if server_type != "mcp" and path:
            full_url = f"{base_url}/{path}"

        auth_type = server.get("auth_type", "bearer")
        token: Optional[str] = None
        headers: Dict[str, str] = {}

        if auth_type == "bearer":
            token = server.get("key", "")
        elif auth_type == "session":
            token = session_token
        elif auth_type == "none":
            token = None

        server_entries.append((idx, server, full_url, token or "", server_type, headers))

    # Create async tasks to fetch data
    tasks = [
        get_tool_server_data(token, url, server_type=server_type, headers=headers or None)
        for (_, _, url, token, server_type, headers) in server_entries
    ]

    # Execute tasks concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Build final results with index and server metadata
    results = []
    for (idx, server, url, _, server_type, _), response in zip(server_entries, responses):
        if isinstance(response, Exception):
            log.error("Failed to connect to %s tool server (%s)", url, server_type)
            continue

        results.append(
            {
                "idx": idx,
                "url": server.get("url"),
                "connection_url": response.get("connection_url", response.get("url", url)),
                "type": response.get("type", server_type),
                "headers": response.get("headers", {}),
                "openapi": response.get("openapi"),
                "info": response.get("info"),
                "specs": response.get("specs", []),
            }
        )

    return results


async def execute_tool_server(
    token: Optional[str],
    url: str,
    name: str,
    params: Dict[str, Any],
    server_data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    """Execute a tool against a configured tool server."""

    server_data = server_data or {}
    server_type = (server_data.get("type") or "openapi").lower()
    params = params or {}

    if server_type == "mcp":
        connection_url = (
            server_data.get("connection_url")
            or url
            or server_data.get("url")
        )

        if not connection_url:
            return {"error": "Missing connection URL for MCP server."}

        request_headers: Dict[str, str] = {}
        if isinstance(server_data.get("headers"), dict):
            request_headers.update(server_data.get("headers"))
        if headers:
            request_headers.update(headers)
        if token and "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {token}"

        client: Optional[MCPClient] = server_data.get("client")  # type: ignore[assignment]
        created_client = False

        if client is None:
            client = MCPClient()
            created_client = True
            try:
                await client.connect(url=connection_url, headers=request_headers or None)
            except Exception as err:
                log.exception("Failed to connect to MCP server %s: %s", connection_url, err)
                return {"error": str(err)}

        try:
            tool_name = server_data.get("tool_name_map", {}).get(name, name)
            return await client.call_tool(tool_name, params)
        except Exception as err:
            log.exception("MCP tool execution error for %s: %s", name, err)
            return {"error": str(err)}
        finally:
            if created_client:
                try:
                    await client.disconnect()
                except Exception:
                    pass

    try:
        openapi = server_data.get("openapi", {})
        paths = openapi.get("paths", {})

        matching_route = None
        for route_path, methods in paths.items():
            for http_method, operation in methods.items():
                if isinstance(operation, dict) and operation.get("operationId") == name:
                    matching_route = (route_path, methods)
                    break
            if matching_route:
                break

        if not matching_route:
            raise Exception(f"No matching route found for operationId: {name}")

        route_path, methods = matching_route

        method_entry = None
        for http_method, operation in methods.items():
            if operation.get("operationId") == name:
                method_entry = (http_method.lower(), operation)
                break

        if not method_entry:
            raise Exception(f"No matching method found for operationId: {name}")

        http_method, operation = method_entry

        path_params: Dict[str, Any] = {}
        query_params: Dict[str, Any] = {}
        body_params: Dict[str, Any] = {}

        for param in operation.get("parameters", []):
            param_name = param["name"]
            param_in = param["in"]
            if param_name in params:
                if param_in == "path":
                    path_params[param_name] = params[param_name]
                elif param_in == "query":
                    query_params[param_name] = params[param_name]

        base_url = url or server_data.get("connection_url") or server_data.get("url")
        if not base_url:
            raise Exception("No base URL configured for tool server")

        final_url = f"{base_url}{route_path}"
        for key, value in path_params.items():
            final_url = final_url.replace(f"{{{key}}}", str(value))

        if query_params:
            query_string = "&".join(f"{k}={v}" for k, v in query_params.items())
            final_url = f"{final_url}?{query_string}"

        if operation.get("requestBody", {}).get("content"):
            if params:
                body_params = params
            else:
                raise Exception(
                    f"Request body expected for operation '{name}' but none found."
                )

        request_headers: Dict[str, str] = {"Content-Type": "application/json"}
        if isinstance(server_data.get("headers"), dict):
            request_headers.update(server_data.get("headers"))
        if headers:
            request_headers.update(headers)
        if token and "Authorization" not in request_headers:
            request_headers["Authorization"] = f"Bearer {token}"

        async with aiohttp.ClientSession() as session:
            request_method = getattr(session, http_method.lower())

            if http_method in ["post", "put", "patch", "delete"]:
                async with request_method(
                    final_url, json=body_params, headers=request_headers
                ) as response:
                    if response.status >= 400:
                        text = await response.text()
                        raise Exception(f"HTTP error {response.status}: {text}")
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        return await response.text()
            else:
                async with request_method(final_url, headers=request_headers) as response:
                    if response.status >= 400:
                        text = await response.text()
                        raise Exception(f"HTTP error {response.status}: {text}")
                    try:
                        return await response.json()
                    except aiohttp.ContentTypeError:
                        return await response.text()

    except Exception as err:
        log.exception(f"API Request Error: {err}")
        return {"error": str(err)}

def get_tool_server_url(url: Optional[str], path: str) -> str:
    """
    Build the full URL for a tool server, given a base url and a path.
    """
    if "://" in path:
        # If it contains "://", it's a full URL
        return path
    if not path.startswith("/"):
        # Ensure the path starts with a slash
        path = f"/{path}"
    return f"{url}{path}"

async def download_file(url: str, output_path: str):
    """
    Downloads a file from URL to output_path using streaming approach.
    Similar to the JavaScript version that uses responseType: "stream".
    """
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        
        print(f"Downloading from {url} to {output_path}")
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                # Use streaming approach similar to JavaScript
                async with aiofiles.open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                
                print(f"Successfully downloaded file to: {output_path}")
                        
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        print(f"URL: {url}")
        print(f"Output path: {output_path}")
        raise e


def convert_to_16khz(input_path, output_path):
    try:
        (
            ffmpeg
            .input(input_path)
            .output(output_path, ar=16000, ac=1) # ar = audio rate (frequency), ac = audio channels
            .run(overwrite_output=True, quiet=True) # Execute the command
        )
        print(f"Successfully converted {input_path} to {output_path}")
        return output_path
    except ffmpeg.Error as e:
        print("FFmpeg Error:", e.stderr.decode() if e.stderr else "Unknown error")
        raise e