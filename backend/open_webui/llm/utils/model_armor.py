import aiohttp
import json
from typing import Dict, Any, Optional, List
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from open_webui.env import (
    MODEL_ARMOR_PRIVATE_KEY,
    MODEL_ARMOR_PRIVATE_KEY_ID,
)
import logging

logger = logging.getLogger(__name__)


class ModelArmorService:
    def __init__(self, backend_host: Optional[str] = None):
        self.project_id = "model-armor-poc"
        self.location = "us-central1"
        self.backend_host = (
            backend_host or "https://modelarmor.us-central1.rep.googleapis.com"
        )
        self.session = None  # aiohttp session will be initialized in async context
        self.credentials = service_account.Credentials.from_service_account_info(
            {
                "type": "service_account",
                "project_id": "model-armor-poc",
                "private_key_id": MODEL_ARMOR_PRIVATE_KEY_ID,
                "private_key": MODEL_ARMOR_PRIVATE_KEY,
                "client_email": "model-armor-poc@model-armor-poc.iam.gserviceaccount.com",
                "client_id": "100308294558476962398",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/model-armor-poc%40model-armor-poc.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com",
            },
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    async def __aenter__(self):
        """Initialize aiohttp session."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Close aiohttp session."""
        await self.session.close()

    async def get_access_token(self) -> Optional[str]:
        """
        Retrieves and refreshes the access token using Google OAuth 2.0.
        Returns the access token, or None if an error occurs.
        """
        try:
            if self.credentials.expired or not self.credentials.valid:
                self.credentials.refresh(Request())
            return self.credentials.token
        except Exception as error:
            logger.error(f"Error retrieving access token: {error}")
            return None

    def extract_template_id(self, resource_name: str) -> Optional[str]:
        """
        Extracts the template_id from a Google Cloud resource name.
        Args:
            resource_name: Resource name in the format
                           'projects/{project_id}/locations/{location}/templates/{template_id}'
        Returns:
            The template_id (e.g., 'imbrace_dev'), or None if the format is invalid
        """
        try:
            if not isinstance(resource_name, str):
                raise ValueError(
                    f"Invalid resource_name type: expected str, got {type(resource_name)}"
                )
            parts = resource_name.strip().split("/")
            if len(parts) >= 6 and parts[-2] == "templates":
                return parts[-1]
            raise ValueError(f"Invalid resource name format: {resource_name}")
        except Exception as error:
            logger.error(f"Error extracting template_id: {error}")
            return None

    async def get_headers(self) -> Dict[str, str]:
        """
        Generates headers with the access token for API requests.
        Returns:
            Headers dictionary with Authorization and Content-Type.
        """
        try:
            token = await self.get_access_token()
            if not token:
                raise ValueError("Failed to retrieve access token")
            return {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        except Exception as error:
            logger.error(f"Error generating headers: {error}")
            raise

    async def list_templates(self) -> List[Dict[str, Any]]:
        """
        Lists all templates for the project and location.
        Returns:
            The API response data as a list of templates.
        """
        try:
            headers = await self.get_headers()
            url = f"{self.backend_host}/v1/projects/{self.project_id}/locations/{self.location}/templates"
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as error:
            logger.error(f"Error listing templates: {error}")
            raise

    async def create_template(
        self, data: Dict[str, Any], name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Creates a new template with the specified data and template ID.
        Args:
            data: The template configuration data.
            name: The template ID or resource name.
        Returns:
            The API response data.
        """
        try:
            if not isinstance(data, dict):
                raise ValueError("Invalid data: expected a dictionary")
            if not isinstance(name, str):
                raise ValueError("Invalid name: expected a string")
            
            template_id = self.extract_template_id(name) or name
            headers = await self.get_headers()
            url = f"{self.backend_host}/v1/projects/{self.project_id}/locations/{self.location}/templates?template_id={template_id}"
            
            # Enhanced logging
            logger.info("=== MODEL ARMOR CREATE TEMPLATE REQUEST ===")
            logger.info("URL: %s", url)
            logger.info("Template ID: %s", template_id)
            logger.info("Request payload: %s", json.dumps(data, indent=2))
            
            async with self.session.post(url, headers=headers, json=data) as response:
                response_text = await response.text()
                
                logger.info("=== MODEL ARMOR CREATE TEMPLATE RESPONSE ===")
                logger.info("Status Code: %s", response.status)
                logger.info("Response Body: %s", response_text)
                
                if response.status >= 400:
                    logger.error("❌ Google Model Armor API error (%s)", response.status)
                    logger.error("Request URL: %s", url)
                    logger.error("Request payload: %s", json.dumps(data, indent=2))
                    logger.error("Error response: %s", response_text)
                    
                    # Try to parse error as JSON
                    try:
                        error_json = json.loads(response_text)
                        logger.error("Parsed error JSON: %s", json.dumps(error_json, indent=2))
                    except:
                        pass
                
                response.raise_for_status()
                
                # Parse and return response
                try:
                    result = json.loads(response_text) if response_text else {}
                    logger.info("✅ Model Armor template created successfully")
                    return result
                except json.JSONDecodeError as e:
                    logger.error("Failed to parse response as JSON: %s", e)
                    raise ValueError(f"Invalid JSON response: {response_text}")
                    
        except aiohttp.ClientError as error:
            logger.error("❌ HTTP error creating template: %s", error)
            logger.error("Error type: %s", type(error).__name__)
            raise
        except Exception as error:
            logger.error("❌ Unexpected error creating template: %s", error)
            logger.error("Error type: %s", type(error).__name__)
            logger.error("Full traceback:", exc_info=True)
            raise

    async def read_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a template by ID.
        Args:
            template_id: The template ID.
        Returns:
            The API response data.
        """
        try:
            if not isinstance(template_id, str):
                raise ValueError("Invalid template_id: expected a string")
            headers = await self.get_headers()
            url = f"{self.backend_host}/v1/projects/{self.project_id}/locations/{self.location}/templates/{template_id}"
            async with self.session.get(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as error:
            logger.error(f"Error reading template {template_id}: {error}")
            raise

    async def update_template(
        self, template_id: str, data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Updates a template by ID with the specified data.
        Args:
            template_id: The template ID.
            data: The updated template configuration data.
        Returns:
            The API response data.
        """
        try:
            if not isinstance(template_id, str):
                raise ValueError("Invalid template_id: expected a string")
            if not isinstance(data, dict):
                raise ValueError("Invalid data: expected a dictionary")
            headers = await self.get_headers()
            url = f"{self.backend_host}/v1/projects/{self.project_id}/locations/{self.location}/templates/{template_id}"
            async with self.session.patch(url, headers=headers, json=data) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as error:
            logger.error(f"Error updating template {template_id}: {error}")
            raise

    async def delete_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Deletes a template by ID.
        Args:
            template_id: The template ID.
        Returns:
            The API response data.
        """
        try:
            if not isinstance(template_id, str):
                raise ValueError("Invalid template_id: expected a string")
            logger.info(f"Deleting template with ID: {template_id}")
            headers = await self.get_headers()
            url = f"{self.backend_host}/v1/projects/{self.project_id}/locations/{self.location}/templates/{template_id}"
            async with self.session.delete(url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as error:
            logger.error(f"Error deleting template {template_id}: {error}")
            raise

    async def sanitize_prompt(
        self, template_id: str, prompt_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Sanitizes a user prompt using the specified template.
        Args:
            template_id: The template ID.
            prompt_data: The prompt data to sanitize.
        Returns:
            The API response data.
        """
        try:
            if not isinstance(template_id, str):
                raise ValueError("Invalid template_id: expected a string")
            if not isinstance(prompt_data, dict):
                raise ValueError("Invalid prompt_data: expected a dictionary")
            headers = await self.get_headers()
            url = f"{self.backend_host}/v1/projects/{self.project_id}/locations/{self.location}/templates/{template_id}:sanitizeUserPrompt"
            async with self.session.post(
                url, headers=headers, json=prompt_data
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as error:
            logger.error(f"Error sanitizing prompt for template {template_id}: {error}")
            raise

    async def sanitize_response(
        self, template_id: str, response_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Sanitizes a model response using the specified template.
        Args:
            template_id: The template ID.
            response_data: The response data to sanitize.
        Returns:
            The API response data.
        """
        try:
            if not isinstance(template_id, str):
                raise ValueError("Invalid template_id: expected a string")
            if not isinstance(response_data, dict):
                raise ValueError("Invalid response_data: expected a dictionary")
            headers = await self.get_headers()
            url = f"{self.backend_host}/v1/projects/{self.project_id}/locations/{self.location}/templates/{template_id}:sanitizeModelResponse"
            async with self.session.post(
                url, headers=headers, json=response_data
            ) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as error:
            logger.error(
                f"Error sanitizing response for template {template_id}: {error}"
            )
            raise


# Factory function
def create_model_armor_service(backend_host: Optional[str] = None) -> ModelArmorService:
    return ModelArmorService(backend_host)


# Module-level async functions for convenience
async def list_templates(backend_host: Optional[str] = None) -> List[Dict[str, Any]]:
    async with ModelArmorService(backend_host) as service:
        return await service.list_templates()


async def create_template(
    data: Dict[str, Any], name: str, backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with ModelArmorService(backend_host) as service:
        return await service.create_template(data, name)


async def read_template(
    template_id: str, backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with ModelArmorService(backend_host) as service:
        return await service.read_template(template_id)


async def update_template(
    template_id: str, data: Dict[str, Any], backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with ModelArmorService(backend_host) as service:
        return await service.update_template(template_id, data)


async def delete_template(
    template_id: str, backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with ModelArmorService(backend_host) as service:
        return await service.delete_template(template_id)


async def sanitize_prompt(
    template_id: str, prompt_data: Dict[str, Any], backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with ModelArmorService(backend_host) as service:
        return await service.sanitize_prompt(template_id, prompt_data)


async def sanitize_response(
    template_id: str, response_data: Dict[str, Any], backend_host: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    async with ModelArmorService(backend_host) as service:
        return await service.sanitize_response(template_id, response_data)
