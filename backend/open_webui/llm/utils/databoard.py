import json
import copy
import requests
from typing import Dict, Any, Optional, List
import random
import pycountry

from open_webui.config import DATABOARD_CONFIG

DATABOARD_HOST = DATABOARD_CONFIG.get("host", "http://localhost:8081")

class DataboardService:
    def __init__(self, backend_host: Optional[str] = None):
        self.backend_host = backend_host or DATABOARD_HOST
        self.databoard_host = DATABOARD_HOST
        
        # Type mapping from JavaScript
        self.types = {
            "ShortText": "string",
            "LongText": "string",
            "SingleSelection": "string",
            "MultipleSelection": "array",
            "Number": "number",
            "Time": "string",
            "Date": "string",
            "Datetime": "string",
            "Link": "string",
            "Priority": "string",
            "Assignee": "Assignee",  # TODO: non js type
            "Email": "string",
            "Phone": "object",
            "RichText": "string",
            "Country": "object",
            "Notes": "string",
            "Origin": "string",
            "Attachment": "array",
            "Currency": "object",
            "Checkbox": "boolean",
            "TableInTable": "array",
        }

        self.skipped_field_types = ["Assignee", "TableInTable"]

        # Import output_schema_properties from the primary utils module
        from open_webui.utils.databoard import output_schema_properties
        self.output_schema_properties = output_schema_properties
        self.country_codes = [country.alpha_2 for country in pycountry.countries]

    def get_board_schema(self, org_id: str = "", board_id: str = "") -> Optional[Dict[str, Any]]:
        """Fetch board schema from the API."""
        try:
            url = f"{self.backend_host}/api/boards/{board_id}/schema"
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            print(f"Error fetching board schema: {error}")
            raise error

    def _build_field_property(self, field: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build a JSON schema property dict for a single board field."""
        field_type = field.get("type")
        field_name = field.get("name")

        if not field_name or field_type in self.skipped_field_types or field_type not in self.types:
            return None

        base_schema = self.output_schema_properties.get(field_type, {})
        prop = base_schema.copy()

        description = prop.get("description", "")
        field_description = field.get("description")
        if field_description:
            description += field_description
        prop["description"] = description

        _enum = None
        if field_type == "Country":
            _enum = self.country_codes
        else:
            field_data = field.get("data", [])
            if isinstance(field_data, list):
                _enum = [item.get("value") for item in field_data if item.get("value") is not None]

        if self.types.get(field_type) == "string" and isinstance(_enum, list) and len(_enum) > 0:
            prop["enum"] = _enum

        if field.get("is_identifier"):
            prop["default"] = str(random.randint(1000, 9999))

        return prop

    def _build_table_in_table_property(self, field: Dict[str, Any], child_fields: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build a TableInTable JSON schema property from child field dicts."""
        child_properties = {}
        for child_field in child_fields or []:
            child_prop = self._build_field_property(child_field)
            if child_prop is not None:
                child_properties[child_field.get("name")] = child_prop

        if not child_properties:
            return None

        base = copy.deepcopy(self.output_schema_properties.get("TableInTable", {}))
        description = base.get("description", "")
        field_description = field.get("description")
        if field_description:
            description += field_description
        base["description"] = description
        base["items"]["properties"] = child_properties
        return base

    def _resolve_table_in_table_child_board_id(self, field: Dict[str, Any]) -> Optional[str]:
        """Resolve the child board id, preferring settings.childBoardId over legacy data[0]."""
        settings = field.get("settings")
        if isinstance(settings, dict):
            child_board_id = settings.get("childBoardId") or settings.get("child_board_id")
            if child_board_id:
                return child_board_id
        field_data = field.get("data")
        if isinstance(field_data, list) and len(field_data) > 0:
            return field_data[0].get("value") or field_data[0].get("_id")
        return None

    def get_board_as_function_call_schema(self, board_info: Dict[str, Any], organization_id: str = None) -> Dict[str, Any]:
        """Convert board info to function call schema."""
        properties = {}
        fields = board_info.get("boardSchema", {}).get("fields", [])

        # Process each field
        for field in fields:
            field_type = field.get("type")
            field_name = field.get("name")

            if not field_name:
                continue

            if field_type == "TableInTable":
                inline_child_fields = field.get("child_board_fields")
                if isinstance(inline_child_fields, list) and inline_child_fields:
                    prop = self._build_table_in_table_property(field, inline_child_fields)
                    if prop is not None:
                        properties[field_name] = prop
                    continue

                if not organization_id:
                    continue
                child_board_id = self._resolve_table_in_table_child_board_id(field)
                if not child_board_id:
                    continue

                try:
                    child_board_info = self.get_board_schema(organization_id, child_board_id)
                    if not child_board_info:
                        continue
                except Exception:
                    continue

                child_fields = child_board_info.get("boardSchema", {}).get("fields", [])
                prop = self._build_table_in_table_property(field, child_fields)
                if prop is None:
                    continue
                properties[field_name] = prop
                continue

            prop = self._build_field_property(field)
            if prop is not None:
                properties[field_name] = prop

        # Add connector fallback property
        properties["ReQuEsT_HuMaN"] = {
            "type": "boolean",
            "description": "Return true if you are not confident about filling in all the required values; otherwise, return false.",
            "default": False,
            "example": "<boolean>",
        }

        # Add connector comments property
        properties["SaY_To_HuMaN"] = {
            "type": "string",
            "description": "Please share your comments here in at least 20 words to describe how you performed.",
            "example": "<string>",
        }

        schema = {
            "title": "output schema",
            "type": "object",
            "properties": properties,
        }

        print(json.dumps({"schema": schema}, indent=2))
        return schema

    def execute_pipeline(self, org_id: str = "", board_id: str = "", payload: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Execute pipeline with given payload."""
        if payload is None:
            payload = {}
            
        try:
            url = f"{self.backend_host}/api/boards/{board_id}/aggregate"
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            print(f"Error executing pipeline: {error}")
            return None

    def get_folder_subfolders(
        self,
        folder_ids: List[str],
        recursive: bool = True,
        ignore_assistant: bool = True,
        identity_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve folder structure including subfolders and file listings.

        Args:
            folder_ids: List of folder IDs to query.
            recursive: Whether to include nested subfolders recursively.
            ignore_assistant: Whether to ignore assistant-specific folders.
            identity_headers: Caller identity headers (x-organization-id,
                x-user-id, x-access-token) forwarded to data-board so it can
                resolve the caller's team scope and filter the requested ids.
                Without them data-board fails open to the whole org.

        Returns:
            API response dict with folder structure, file counts and file names,
            or None on error.
        """
        try:
            url = f"{self.databoard_host}/api/folders/subfolders"
            payload = {
                "ids": folder_ids,
                "recursive": True,
                "ignore_assistant": ignore_assistant,
            }
            response = requests.post(
                url, json=payload, headers=identity_headers or None, timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            print(f"Error fetching folder subfolders: {error}")
            return None

    def upload(self, organization_id: str, file_buffer: bytes, file_name: str) -> Optional[str]:
        """Upload file to the databoard service."""
        endpoint = f"/v1/board/_fileupload/{organization_id}"
        url = f"{self.backend_host}{endpoint}"

        files = {"file": (file_name, file_buffer)}
        headers = {"accept": "application/json, text/plain, */*"}

        try:
            response = requests.post(url, files=files, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("url")
        except requests.RequestException as error:
            print(f"Error uploading file: {error}")
            return None


# Factory function to create service instance
def create_databoard_service(backend_host: Optional[str] = None) -> DataboardService:
    """Create a databoard service instance."""
    return DataboardService(backend_host)


# For backward compatibility, you can also create module-level functions
def get_board_schema(org_id: str = "", board_id: str = "", backend_host: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Module-level function to get board schema."""
    service = DataboardService(backend_host)
    return service.get_board_schema(org_id, board_id)


def get_board_as_function_call_schema(board_info: Dict[str, Any], backend_host: Optional[str] = None, organization_id: str = None) -> Dict[str, Any]:
    """Module-level function to get board as function call schema."""
    service = DataboardService(backend_host)
    return service.get_board_as_function_call_schema(board_info, organization_id=organization_id)


def execute_pipeline(org_id: str = "", board_id: str = "", payload: Dict[str, Any] = None, backend_host: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Module-level function to execute pipeline."""
    service = DataboardService(backend_host)
    return service.execute_pipeline(org_id, board_id, payload)


def upload(organization_id: str, file_buffer: bytes, file_name: str, backend_host: Optional[str] = None) -> Optional[str]:
    """Module-level function to upload file."""
    service = DataboardService(backend_host)
    return service.upload(organization_id, file_buffer, file_name)