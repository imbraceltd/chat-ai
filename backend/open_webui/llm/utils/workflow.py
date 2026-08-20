import requests
import re
from typing import Dict, List, Any, Optional
from open_webui.config import WORKFLOW_CONFIG
WORKFLOW_HOST = WORKFLOW_CONFIG.get("host", "http://localhost:8001")
AP_WORKFLOW_HOST = WORKFLOW_CONFIG.get("ap_host", "http://localhost:9983")
AP_WORKFLOW_WEBHOOK_URL = WORKFLOW_CONFIG.get("ap_webhook_url", "http://localhost:9983")

def set_workflow_input(details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create workflow input configuration with default values.
    
    Args:
        details: Dictionary containing workflow configuration details
        
    Returns:
        Dictionary containing workflow configuration
    """
    if details is None:
        details = {}
    
    # Set default values
    default_details = {
        "organizationId": "",
        "workflowName": "",
        "textExpression": "",
        "threadIdExpression": "",
        "assistantId": "",
        "credentialId": "",
        "credentialName": "",
        "tags": [],
        "method": "",
        "sendMessage": False,
    }
    
    # Update defaults with provided details
    merged_details = {**default_details, **details}
    
    return {
        "name": merged_details["workflowName"],
        "nodes": [
            {
                "parameters": {"icsTitle": "Start"},
                "name": "Start",
                "type": "n8n-nodes-base.start",
                "typeVersion": 1,
                "position": [60, 240],
            },
            {
                "parameters": {
                    "icsTitle": "Manage AI-Assistants",
                    "assistant_id": merged_details["assistantId"],
                    "advanced": merged_details["sendMessage"],
                    "sendMessage": merged_details["sendMessage"],
                },
                "name": "Manage AI-Assistants",
                "type": "n8n-nodes-base.assistantRequest",
                "typeVersion": 1,
                "position": [240, 240],
            },
        ],
        "connections": {
            "Start": {
                "main": [[{"node": "Manage AI-Assistants", "type": "main", "index": 0}]],
            },
        },
        "settings": {},
        "tags": merged_details["tags"],
        "active": merged_details["method"] == "POST",
    }


def get_workflow_error(openai_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and format workflow error from OpenAI response.
    
    Args:
        openai_response: Response object from OpenAI API
        
    Returns:
        Formatted error dictionary
    """
    if not openai_response or not openai_response.get("response"):
        return openai_response
    
    response = openai_response.get("response", {})
    data = response.get("data", {})
    status = response.get("status", 500)
    detail = data.get("detail")
    
    if detail:
        return {
            "message": detail.get("message") if isinstance(detail, dict) else str(detail),
            "status": status
        }
    
    return openai_response


def get_workflow_settings(workflows_list: List, org_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve workflow settings from the workflow service.
    
    Args:
        workflows_list: List of workflow IDs
        org_id: Organization ID
        config: Configuration object containing workflow host information
        
    Returns:
        List of workflow settings
    """
    if not workflows_list or len(workflows_list) <= 0:
        return []
    
    # Prepare request parameters
    url = f"{WORKFLOW_HOST}/api/v1/workflows/all"
    params = {
        "limit": "-1",
        "skip": "0",
        "ids": ",".join(str(i) for i in workflows_list) 
    }
    headers = {
        "content-type": "application/json",
        "x-organization-id": org_id,
    }
    
    print(workflows_list)
    
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        data = response.json()
        
        settings = []
        for workflow in data.get("data", []):
            # Extract AI settings from workflow
            workflow_data = {
                **workflow.get("settings", {}).get("ai", {})
            }
            
            # Generate function name from workflow ID and name
            workflow_name = workflow.get("name", "").strip()
            # Replace non-alphanumeric characters with underscores
            clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', workflow_name)
            function_name = f"{workflow.get('id')}_{clean_name}"
            
            workflow_data["name"] = function_name
            
            # Ensure function object exists and set name
            if "function" not in workflow_data:
                workflow_data["function"] = {}
            workflow_data["function"]["name"] = function_name
            
            # Add organization_id to function if it's empty or missing
            if not workflow_data["function"].get("organization_id"):
                workflow_data["function"]["organization_id"] = org_id


            
            settings.append(workflow_data)
        
        return settings if settings else []
        
    except requests.exceptions.RequestException as error:
        print(f"Error fetching workflow settings: {error}")
        return []
    except Exception as error:
        print(f"Unexpected error: {error}")
        return []
    

def get_workflow_settings_v2(workflows_list: List, org_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve workflow settings from the workflow service.
    
    Args:
        workflows_list: List of workflow IDs
        org_id: Organization ID
        config: Configuration object containing workflow host information
        
    Returns:
        List of workflow settings
    """
    if not workflows_list or len(workflows_list) <= 0:
        return []

    ids_set = set(str(i) for i in workflows_list)
    headers = {
        "content-type": "application/json",
        "x-organization-id": org_id,
    }

    print(workflows_list)

    try:
        all_flows = []
        cursor = None
        while True:
            params = {"limit": "100"}
            if cursor:
                params["cursor"] = cursor
            response = requests.get(
                f"{AP_WORKFLOW_HOST}/api/v1/flows",
                params=params,
                headers=headers,
            )
            response.raise_for_status()
            page = response.json()
            for flow in page.get("data", []):
                if str(flow.get("id")) in ids_set:
                    all_flows.append(flow)
            cursor = page.get("next")
            if not cursor:
                break

        settings = []
        for workflow in all_flows:
            # Extract AI settings from workflow
            metadata = workflow.get("metadata", {})
            workflow_data = {
                **metadata.get("settings", {}).get("ai", {})
            }
            
            # Generate function name from workflow ID and name
            workflow_name = workflow.get("displayName", "").strip()
            # Replace non-alphanumeric characters with underscores
            clean_name = re.sub(r'[^a-zA-Z0-9]+', '_', workflow_name)
            function_name = f"{workflow.get('id')}_{clean_name}"
            
            workflow_data["name"] = function_name
            
            # Ensure function object exists and set name
            if "function" not in workflow_data:
                workflow_data["function"] = {}
            workflow_data["function"]["name"] = function_name
            
            # Add organization_id to function if it's empty or missing
            if not workflow_data["function"].get("organization_id"):
                workflow_data["function"]["organization_id"] = org_id


            
            settings.append(workflow_data)
        
        return settings if settings else []
        
    except requests.exceptions.RequestException as error:
        print(f"Error fetching workflow settings: {error}")
        return []
    except Exception as error:
        print(f"Unexpected error: {error}")
        return []


def trigger(organization_id: str = "", workflow_id: str = "", method: str = "POST", data: Dict[str, Any] = None) -> Any:
    """
    Trigger a workflow execution.
    
    Args:
        organization_id: Organization ID
        workflow_id: Workflow ID to trigger
        method: HTTP method (default: POST)
        data: Data to send with the trigger request
        
    Returns:
        Response data from the workflow trigger or error message
    """
    if data is None:
        data = {}
    
    try:
        url = f"{WORKFLOW_HOST}/api/v1/workflows/{workflow_id}/trigger"
        
        # Prepare request payload
        payload = {**data, "method": method}
        
        headers = {
            "Content-Type": "application/json",
            "x-organization-id": organization_id
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Check if response has content before trying to parse JSON
        response_text = response.text if response.text else ""
        if response_text.strip():
            try:
                return response.json()
            except ValueError as json_error:
                print(f"Error parsing JSON response: {json_error}")
                print(f"Response text: {response.text}")
                return "There is no returned response from the tool call. Please try again later"
        else:
            print("Empty response from workflow trigger")
            return "The tool call succeeded but returned no content."
        
    except requests.exceptions.RequestException as error:
        print(f"Error triggering workflow: {error}")
        return "Tool call failed. Please ask user to wait a moment and try again later."
    except Exception as error:
        print(f"Unexpected error triggering workflow: {error}")
        return "The tool call failed. Please ask user to wait a moment and try again later."


def trigger_v2(organization_id: str = "", workflow_id: str = "", method: str = "POST", data: Dict[str, Any] = None) -> Any:
    """
    Trigger a workflow execution.
    
    Args:
        organization_id: Organization ID
        workflow_id: Workflow ID to trigger
        method: HTTP method (default: POST)
        data: Data to send with the trigger request
        
    Returns:
        Response data from the workflow trigger or error message
    """
    if data is None:
        data = {}
    
    try:
        url = f"{AP_WORKFLOW_WEBHOOK_URL}/{workflow_id}/sync"

        # Prepare request payload
        payload = {**data, "method": method}
        
        headers = {
            "Content-Type": "application/json",
            "x-organization-id": organization_id
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=300)
        response.raise_for_status()

        # Check if response has content before trying to parse JSON
        response_text = response.text if response.text else ""
        if response_text.strip():
            try:
                response_data = response.json()
                print("trigger response:", response_data)
                return response_data
            except ValueError as json_error:
                print(f"Error parsing JSON response: {json_error}")
                print(f"Response text: {response.text}")
                return "There is no returned response from the tool call. Please try again later"
        else:
            print("Empty response from workflow trigger")
            return "The tool call succeeded but returned no content."

    except requests.exceptions.RequestException as error:
        print(f"Error triggering workflow: {error}")
        return "Tool call failed. Please ask user to wait a moment and try again later."
    except Exception as error:
        print(f"Unexpected error triggering workflow: {error}")
        return "Tool call failed. Please ask user to wait a moment and try again later."