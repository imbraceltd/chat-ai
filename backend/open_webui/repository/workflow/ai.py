from open_webui.routers.helper import set_workflow_input
import requests
import logging
from open_webui.env import IMBRACE_BACKNED_PRIVATE


log = logging.getLogger(__name__)


def get_all(organizationId='', filterOptions={}):
    try:
        url = f"{IMBRACE_BACKNED_PRIVATE}/v1/organization/{organizationId}/n8n/workflows"
        input = set_workflow_input(organizationId, filterOptions)

        headers = {
            'x-org-id': organizationId,
            'Content-Type': 'application/json'
        }

        response = requests.get(url, input, headers=headers)
        return response

    except Exception as e:
        log.error(f"Error getting workflows: {e}")
        raise e


def update_workflow_name(organizationId, workflow):
    try:
        url = f"{IMBRACE_BACKNED_PRIVATE}/v1/organization/{organizationId}/n8n/workflows/{workflow['id']}"
        headers = {
            'x-org-id': organizationId,
            'Content-Type': 'application/json'
        }

        response = requests.patch(url, workflow, headers=headers)
        return response

    except Exception as e:
        log.error(f"Error updating workflow name: {e}")
        raise e


def update(workflowId, workflowDetails):
    try:
        if workflowDetails is None:
            workflowDetails = {}
        url = f"{IMBRACE_BACKNED_PRIVATE}/v1/organization/{workflowDetails.organizationId}/n8n/workflows/{workflowId}"
        input = set_workflow_input(
            workflowDetails.organizationId, workflowDetails)

        headers = {
            'x-org-id': workflowDetails.organizationId,
            'Content-Type': 'application/json'
        }

        response = requests.patch(url, input, headers=headers)
        return response

    except Exception as e:
        log.error(
            f"An unexpected error occurred while updating workflow {workflowId}: {e}")
        raise e
