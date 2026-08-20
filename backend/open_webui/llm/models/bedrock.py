"""
AWS Bedrock Provider Implementation
Implements LLMProvider interface for AWS Bedrock models using langchain-aws
"""

import json
import logging
from typing import Dict, List, Any
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from ..agent import LLMProvider
from ..utils.models import get_models_from_env
from strands.models import BedrockModel
from strands import Agent
from openai import OpenAI
from langchain_aws import ChatBedrockConverse



from langchain_aws import BedrockEmbeddings, ChatBedrock

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProvider):
    """AWS Bedrock-specific implementation of LLM provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = "bedrock"
        self.region = config.get("bedrock", {}).get("region", "us-west-2")
        self.access_key = config.get("bedrock", {}).get("access_key", "")
        self.secret_key = config.get("bedrock", {}).get("secret_key", "")
        self.model = config.get("bedrock", {}).get("model", "qwen.qwen3-32b-v1:0")
        
        # Embedding model id, driven by env RAG_BEDROCK_EMBEDDING_MODEL via
        # CONFIG["bedrock"]["embedding_model"]. Default Titan v1 (1536-dim)
        # matches the rag.embedding/embedding_v2 columns and the backfill.
        self.embedding_model_name = config.get("bedrock", {}).get(
            "embedding_model", "amazon.titan-embed-text-v1"
        )
        
        # Initialize AWS credentials
        self._setup_aws_credentials()

    def _setup_aws_credentials(self):
        """Setup AWS credentials for Bedrock access"""
        self.aws_credentials = {}
        if self.access_key and self.secret_key:
            self.aws_credentials = {
                "aws_access_key_id": self.access_key,
                "aws_secret_access_key": self.secret_key,
                "region_name": self.region
            }
        else:
            # Use default credentials chain (environment, IAM role, etc.)
            self.aws_credentials = {"region_name": self.region}

    def create_embedding_model(self):
        """Create Bedrock embedding model"""
        try:
            # Pass an explicit boto3 client built from the configured Bedrock
            # keys. Without this, BedrockEmbeddings falls back to the default
            # credential chain (the EC2 IAM role), which lacks
            # bedrock:InvokeModel and fails query embedding with AccessDenied.
            # When keys are absent, client=None lets boto3 use the default chain.
            client = None
            if self.access_key and self.secret_key:
                client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                )
            return BedrockEmbeddings(
                model_id=self.embedding_model_name,
                region_name=self.aws_credentials.get("region_name", "us-west-2"),
                client=client,
            )
        except Exception as e:
            logger.error(f"Failed to create Bedrock embedding model: {str(e)}")
            raise

    def create_retriever_model(self, model: str, temperature: float = 0.1):
        """Create Bedrock retriever model"""
        try:
            return  ChatBedrock(
                model_id=self.model,
                temperature=temperature,
                **self.aws_credentials
            )
        except Exception as e:
            logger.error(f"Failed to create Bedrock retriever model: {str(e)}")
            raise

    def create_llm(
        self,
        model: str,
        available_models: Dict[str, List[Dict]],
        streaming: bool = True,
        temperature: float = 0.1,
    ):
        """Create Bedrock LLM for main chat"""
        try:
            logger.info(f"Creating Bedrock LLM with model: {model}, streaming: {streaming}")
#             session = boto3.Session(
#                         aws_access_key_id=self.access_key,
#                         aws_secret_access_key=self.secret_key,
#                         region_name=self.region,
# )
#             bedrock_model = BedrockModel(
#                 model_id=model or self.model,
#                 temperature=temperature,
#                 top_p=0.1,
#                 boto_session=session,
#                 streaming=True,
#             )
#             return Agent(
#                 model=bedrock_model
#             )
            llm = ChatBedrockConverse(
                model_id=model or self.model,
                temperature=temperature,
                top_p=temperature,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
            )
            # Store model ID as a custom attribute using object.__setattr__ to bypass Pydantic validation
            # This allows us to access the model ID later without relying on internal fields
            object.__setattr__(llm, 'model', model or self.model)
            return llm
        except Exception as e:
            logger.error(f"Failed to create Bedrock LLM: {str(e)}")
            raise

    async def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get available Bedrock models from AWS API"""
        try:
            # Initialize Bedrock client
            bedrock_client = boto3.client(
                'bedrock',
                **self.aws_credentials
            )

            # Fetch foundation models from Bedrock
            response = bedrock_client.list_foundation_models()
            
            # Filter and format models for chat completion
            bedrock_models = []
            for model in response.get('modelSummaries', []):
                # Only include models that support text generation and are active
                if ('TEXT' in model.get('inputModalities', []) and 
                    'TEXT' in model.get('outputModalities', []) and
                    model.get('modelLifecycle', {}).get('status') == 'ACTIVE'):
                    
                    bedrock_models.append({
                        "name": model['modelId'],
                        "provider": "bedrock",
                        "description": model.get('modelName', ''),
                        "provider_name": model.get('providerName', '')
                    })

            # Sort models by name for consistent ordering
            bedrock_models.sort(key=lambda x: x["name"])

            logger.info(f"Fetched {len(bedrock_models)} Bedrock models from API")
            return {"bedrockModels": bedrock_models}

        except NoCredentialsError:
            logger.error("AWS credentials not found for Bedrock")
            # Return default models if credentials are not available
            return self._get_default_models()
        except ClientError as e:
            logger.error(f"Failed to fetch Bedrock models: {str(e)}")
            # Return default models if API call fails
            return self._get_default_models()
        except Exception as e:
            logger.error(f"Unexpected error fetching Bedrock models: {str(e)}")
            return self._get_default_models()

    async def get_models_from_provider(self) -> Dict[str, List[Dict]]:
        """
        Strictly fetch available Bedrock models from AWS API.

        - On success: return models fetched from AWS.
        - On any error: raise the exception (no local fallback).
        """
        try:
            logger.info(
                "BedrockProvider.get_models_from_provider: listing foundation models in region %s",
                self.region,
            )
            bedrock_client = boto3.client("bedrock", **self.aws_credentials)
            response = bedrock_client.list_foundation_models()

            bedrock_models: List[Dict[str, Any]] = []
            for model in response.get("modelSummaries", []):
                if (
                    "TEXT" in model.get("inputModalities", [])
                    and "TEXT" in model.get("outputModalities", [])
                    and model.get("modelLifecycle", {}).get("status") == "ACTIVE"
                ):
                    model_id = model["modelId"]
                    # Determine capabilities for this model
                    is_vision_available = "IMAGE" in model.get(
                        "outputModalities", []
                    )
                    # Heuristic flag: whether this model supports "thinking"/reasoning mode
                    is_support_thinking = self._is_support_thinking_model_id(model_id)
                    bedrock_models.append(
                        {
                            "name": model_id,
                            "provider": "bedrock",
                            "description": model.get("modelName", ""),
                            "provider_name": model.get("providerName", ""),
                            "is_toolCall_available": True,
                            "is_vision_available": is_vision_available,
                            "is_support_thinking": is_support_thinking,
                            "is_shown": False,
                        }
                    )

            bedrock_models.sort(key=lambda x: x["name"])

            logger.info(
                "BedrockProvider.get_models_from_provider: fetched %d models from AWS",
                len(bedrock_models),
            )
            return {"bedrockModels": bedrock_models}
        except Exception as e:
            logger.error(
                "BedrockProvider.get_models_from_provider: failed to fetch models from AWS: %s",
                e,
            )
            raise

    def _is_tool_call_model_id(self, model_id: str) -> bool:
        """
        Heuristically determine if a Bedrock model supports tool calling
        based on its model_id string.
        """
        if not model_id:
            return False

        model_id_lower = model_id.lower()

        # Claude 3+ models support tool calling
        if any(
            claude_model in model_id_lower
            for claude_model in ["claude-3", "claude-3-5"]
        ):
            return True

        # Titan models typically support tool calling
        if "titan" in model_id_lower:
            return True

        # Llama 3+ / 4+ models typically support tool calling
        if any(
            substr in model_id_lower
            for substr in ["llama3", "llama-3", "llama4", "llama-4"]
        ):
            return True

        # Mistral models typically support tool calling
        if "mistral" in model_id_lower:
            return True

        # OpenAI GPT-OSS models
        if "gpt-oss" in model_id_lower:
            return True

        # Qwen models
        if "qwen" in model_id_lower:
            return True
        
        if "minimax" in model_id_lower:
            return True

        # Default to False for unknown models to be safe
        return False

    def _is_support_thinking_model_id(self, model_id: str) -> bool:
        """
        Heuristically determine if a Bedrock model supports "thinking"/reasoning mode.
        """
        if not model_id:
            return False

        model_id_lower = model_id.lower()

        # DeepSeek-R1 and other reasoning-focused models
        thinking_patterns = [
            "deepseek.r1",
            "deepseek-r1",
            "thinking",
            "reasoning",
        ]
        return any(p in model_id_lower for p in thinking_patterns)

    def _get_default_models(self) -> Dict[str, List[Dict]]:
        """Fallback to predefined list of common Bedrock models or MODELS env var"""
        models_from_env = get_models_from_env("bedrock")
        if models_from_env:
            bedrock_models_raw = [
                {
                    "name": name,
                    "provider": "bedrock",
                    "description": name.split(".")[-1].replace("-", " ").title() if "." in name else name,
                    "provider_name": name.split(".")[0].title() if "." in name else "Bedrock"
                }
                for name in models_from_env
            ]
        else:
            bedrock_models_raw = []
        bedrock_models: List[Dict[str, Any]] = []
        for model in bedrock_models_raw:
            name = model["name"]
            bedrock_models.append(
                {
                    **model,
                    "is_toolCall_available": True,
                    "is_vision_available": False,
                    "is_support_thinking": self._is_support_thinking_model_id(name),
                    "is_shown": False,
                }
            )

        return {"bedrockModels": bedrock_models}

    def is_tool_call_available(self, llm) -> bool:
        """Bedrock models always support tool calling"""
        return True
