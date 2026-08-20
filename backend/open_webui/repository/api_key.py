import logging
from fastapi import HTTPException, status  # Import HTTPException
from open_webui.internal.mongo_db import mongodb_client
from open_webui.config import OPENAI_API_KEY

log = logging.getLogger(__name__)

API_KEYS_COLLECTION = "openai_payments"
DEFAULT_IMBRACE_KEY = "iMBrace key"


class ApiKeyRepository:
    def __init__(self):
        self.db_client = mongodb_client
        self.db_name = "openai_db"  # Example DB name
        self.collection_name = API_KEYS_COLLECTION

    # Corrected the return type hint to 'str'
    async def check_key_is_paid(self, organization_id: str, api_key: str) -> str:
        """
        Checks if the key is valid for the org. Returns a valid key on success
        or raises an HTTPException on failure.
        """
        try:
            # If the provided key is not the default, assume it's valid and return it
            if api_key != DEFAULT_IMBRACE_KEY:
                return api_key

            # If it is the default key, check the organization's payment status
            query = {"organization_id": organization_id}
            data = await self.db_client.query_one(
                database_name=self.db_name,
                collection_name=self.collection_name,
                query=query
            )

            if not data:
                # Raise an exception that FastAPI will turn into a 404 Not Found response
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Organization ID not found."
                )

            paid = data.get("paid")
            if paid:
                # Return the globally configured OpenAI key for paid orgs
                return OPENAI_API_KEY

            # FIXED: Raise an exception that FastAPI will turn into a 402 Payment Required
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="This service requires a paid subscription."
            )

        except HTTPException:
            # Re-raise HTTP exceptions directly
            raise
        except Exception as e:
            log.error(f"Error checking key is paid: {str(e)}")
            # Raise a generic server error for any other unexpected issues
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while verifying credentials."
            )
