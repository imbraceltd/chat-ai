import aioboto3
import logging
from typing import Dict, Any, Optional, List
from open_webui.env import (
    AWS_BEDROCK_REGION,
    AWS_BEDROCK_GUARDRAIL_VERSION,
    BEDROCK_ACCESS_KEY_ID,
    BEDROCK_SECRET_ACCESS_KEY,
)

logger = logging.getLogger(__name__)


class BedrockGuardrailService:
    """AWS Bedrock Guardrails client wrapper for async operations."""

    def __init__(
        self,
        region_name: Optional[str] = None,
        default_version: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
    ):
        """
        Initialize Bedrock Guardrail service.

        Args:
            region_name: AWS region for Bedrock API calls
            default_version: Default guardrail version to fetch
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
        """
        self.region_name = region_name or AWS_BEDROCK_REGION
        self.default_version = default_version or AWS_BEDROCK_GUARDRAIL_VERSION
        self.access_key_id = access_key_id or BEDROCK_ACCESS_KEY_ID
        self.secret_access_key = secret_access_key or BEDROCK_SECRET_ACCESS_KEY
        self.session = aioboto3.Session(
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name,
        )

    async def __aenter__(self):
        """Initialize async context."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Close async context."""
        pass

    async def list_guardrails(
        self, org_id: Optional[str] = None, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch all Bedrock guardrails.

        Args:
            org_id: Organization ID (not used for Bedrock, for API compatibility)
            max_results: Maximum number of guardrails to return per page

        Returns:
            List of guardrail configurations
        """
        try:
            guardrails = []
            async with self.session.client(
                "bedrock", region_name=self.region_name
            ) as bedrock_client:
                paginator = bedrock_client.get_paginator("list_guardrails")
                async for page in paginator.paginate(
                    PaginationConfig={"MaxItems": max_results}
                ):
                    for guardrail in page.get("guardrails", []):
                        # Normalize to match existing guardrail format
                        normalized = {
                            "guardrails_config_id": guardrail.get("id"),
                            "name": guardrail.get("name"),
                            "description": guardrail.get("description", ""),
                            "model": "bedrock",
                            "org_id": org_id,
                            "created_at": (
                                guardrail.get("createdAt").isoformat()
                                if guardrail.get("createdAt")
                                else None
                            ),
                            "updated_at": (
                                guardrail.get("updatedAt").isoformat()
                                if guardrail.get("updatedAt")
                                else None
                            ),
                            "status": guardrail.get("status"),
                            "version": guardrail.get("version"),
                            "arn": guardrail.get("arn"),
                        }
                        guardrails.append(normalized)

            logger.info(f"Retrieved {len(guardrails)} Bedrock guardrails")
            return guardrails
        except Exception as error:
            logger.error(f"Error listing Bedrock guardrails: {error}")
            return []

    async def get_guardrail(
        self, guardrail_id: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a single guardrail by ID.

        Args:
            guardrail_id: The guardrail identifier or ARN
            version: Guardrail version (defaults to configured version)

        Returns:
            Guardrail configuration or None if not found
        """
        try:
            async with self.session.client(
                "bedrock", region_name=self.region_name
            ) as bedrock_client:
                response = await bedrock_client.get_guardrail(
                    guardrailIdentifier=guardrail_id,
                    guardrailVersion=version or self.default_version,
                )

                # Log the full AWS response for debugging
                logger.info(f"Raw AWS Bedrock guardrail response for {guardrail_id}:")
                logger.info(f"  Name: {response.get('name')}")
                logger.info(f"  Status: {response.get('status')}")
                logger.info(f"  Version: {response.get('version')}")
                logger.info(f"  Content Policy: {response.get('contentPolicy')}")
                logger.info(f"  Word Policy: {response.get('wordPolicy')}")
                logger.info(
                    f"  Sensitive Info Policy: {response.get('sensitiveInformationPolicy')}"
                )
                logger.info(f"  Topic Policy: {response.get('topicPolicy')}")
                logger.info(
                    f"  Blocked Input Messaging: {response.get('blockedInputMessaging')}"
                )
                logger.info(
                    f"  Blocked Outputs Messaging: {response.get('blockedOutputsMessaging')}"
                )
                logger.info(f"  Full Response Keys: {list(response.keys())}")

                # Normalize to match existing guardrail format
                normalized = {
                    "guardrails_config_id": response.get("guardrailId"),
                    "name": response.get("name"),
                    "description": response.get("description", ""),
                    "model": "bedrock",
                    "created_at": (
                        response.get("createdAt").isoformat()
                        if response.get("createdAt")
                        else None
                    ),
                    "updated_at": (
                        response.get("updatedAt").isoformat()
                        if response.get("updatedAt")
                        else None
                    ),
                    "status": response.get("status"),
                    "version": response.get("version"),
                    "arn": response.get("guardrailArn"),
                    # Additional Bedrock-specific fields
                    "blocked_input_messaging": response.get("blockedInputMessaging"),
                    "blocked_outputs_messaging": response.get(
                        "blockedOutputsMessaging"
                    ),
                    "content_policy": response.get("contentPolicy"),
                    "word_policy": response.get("wordPolicy"),
                    "sensitive_information_policy": response.get(
                        "sensitiveInformationPolicy"
                    ),
                    "topic_policy": response.get("topicPolicy"),
                }

                logger.info(f"Retrieved Bedrock guardrail: {guardrail_id}")
                return normalized
        except bedrock_client.exceptions.ResourceNotFoundException:
            logger.warning(f"Bedrock guardrail not found: {guardrail_id}")
            return None
        except Exception as error:
            logger.error(f"Error getting Bedrock guardrail {guardrail_id}: {error}")
            return None

    async def apply_guardrail(
        self,
        guardrail_id: str,
        content: str,
        source: str = "INPUT",
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply guardrail to content for safety assessment.

        Args:
            guardrail_id: The guardrail identifier or ARN
            content: Content to check
            source: "INPUT" or "OUTPUT"
            version: Guardrail version (defaults to configured version)

        Returns:
            Safety assessment results
        """
        try:
            async with self.session.client(
                "bedrock-runtime", region_name=self.region_name
            ) as bedrock_runtime:
                response = await bedrock_runtime.apply_guardrail(
                    guardrailIdentifier=guardrail_id,
                    guardrailVersion=version or self.default_version,
                    source=source,
                    content=[{"text": {"text": content}}],
                )

                # Log the full AWS safety check response for debugging
                logger.info(
                    f"Raw AWS Bedrock safety check response for {guardrail_id}:"
                )
                logger.info(f"  Action: {response.get('action')}")
                logger.info(f"  Outputs: {response.get('outputs')}")
                logger.info(f"  Assessments: {response.get('assessments')}")
                logger.info(f"  Usage: {response.get('usage')}")
                logger.info(f"  Full Response Keys: {list(response.keys())}")

                # Parse the response
                action = response.get("action")  # GUARDRAIL_INTERVENED or NONE
                outputs = response.get("outputs", [])
                assessments = response.get("assessments", [])

                # Check if guardrail blocked the content
                is_safe = action != "GUARDRAIL_INTERVENED"

                # Extract violated categories from assessments
                violated_categories = []
                if not is_safe and assessments:
                    for assessment in assessments:
                        # Check content policy filters
                        if "contentPolicy" in assessment:
                            for filter_item in assessment["contentPolicy"].get(
                                "filters", []
                            ):
                                if filter_item.get("action") == "BLOCKED":
                                    filter_type = filter_item.get("type", "Unknown")
                                    violated_categories.append(
                                        filter_type.replace("_", " ").title()
                                    )

                        # Check word policy filters
                        if "wordPolicy" in assessment:
                            for match in assessment["wordPolicy"].get(
                                "customWords", []
                            ):
                                if match.get("action") == "BLOCKED":
                                    violated_categories.append("Custom Word Filter")

                        # Check sensitive info policy
                        if "sensitiveInformationPolicy" in assessment:
                            for pii in assessment["sensitiveInformationPolicy"].get(
                                "piiEntities", []
                            ):
                                if pii.get("action") == "BLOCKED":
                                    violated_categories.append(
                                        f"PII: {pii.get('type', 'Unknown')}"
                                    )

                        # Check topic policy
                        if "topicPolicy" in assessment:
                            for topic in assessment["topicPolicy"].get("topics", []):
                                if topic.get("action") == "BLOCKED":
                                    violated_categories.append(
                                        f"Topic: {topic.get('name', 'Unknown')}"
                                    )

                result = {
                    "action": action,
                    "is_safe": is_safe,
                    "violated_categories": violated_categories,
                    "outputs": outputs,
                    "assessments": assessments,
                    "usage": response.get("usage", {}),
                }

                logger.info(
                    f"Applied Bedrock guardrail {guardrail_id}: action={action}"
                )
                return result
        except Exception as error:
            logger.error(f"Error applying Bedrock guardrail {guardrail_id}: {error}")
            raise


# Factory function
def create_bedrock_guardrail_service(
    region_name: Optional[str] = None, default_version: Optional[str] = None
) -> BedrockGuardrailService:
    """Create a BedrockGuardrailService instance."""
    return BedrockGuardrailService(
        region_name=region_name, default_version=default_version
    )


# Module-level async functions for convenience
async def list_guardrails(
    org_id: Optional[str] = None,
    max_results: int = 50,
    region_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List all Bedrock guardrails."""
    async with create_bedrock_guardrail_service(region_name=region_name) as service:
        return await service.list_guardrails(org_id=org_id, max_results=max_results)


async def get_guardrail(
    guardrail_id: str,
    version: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Get a specific Bedrock guardrail."""
    async with create_bedrock_guardrail_service(region_name=region_name) as service:
        return await service.get_guardrail(guardrail_id=guardrail_id, version=version)


async def apply_guardrail(
    guardrail_id: str,
    content: str,
    source: str = "INPUT",
    version: Optional[str] = None,
    region_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a Bedrock guardrail to content."""
    async with create_bedrock_guardrail_service(region_name=region_name) as service:
        return await service.apply_guardrail(
            guardrail_id=guardrail_id,
            content=content,
            source=source,
            version=version,
        )
