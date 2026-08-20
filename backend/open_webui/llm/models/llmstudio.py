"""
LMStudio Provider Implementation
Clean implementation without if/else branching for other providers
"""

import logging
from typing import Dict, List, Any
from ..agent import LLMProvider
from ..utils.models import get_models_from_env
import aiohttp

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LMStudioProvider(LLMProvider):
    """LMStudio-specific implementation of LLM provider"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = "lmstudio"
        self.host = config.get("lmstudio", {}).get("host", "http://localhost:1234")
        self.api_type = config.get("lmstudio", {}).get("api_type", "gpt-3.5-turbo")
        self.embedding_model = config.get("lmstudio", {}).get("embedding_model", "text-embedding-nomic-embed-text-v1.5")
        self.api_key = config.get("openApi", {}).get("api_key", "not-needed")
    
    def create_embedding_model(self):
        """Create LMStudio embedding model (uses OpenAI-compatible API)"""
      
        
        return OpenAIEmbeddings(
            openai_api_key=self.api_key,
            model=self.embedding_model,
            openai_api_base=self.host,
            # LMStudio specific configuration
            default_headers={
                "Content-Type": "application/json"
            }
        )
    
    def create_retriever_model(self, model: str, temperature: float = 0.1):
        """Create LMStudio retriever model (uses OpenAI-compatible API)"""
        
        
        return ChatOpenAI(
            openai_api_base=self.host,
            openai_api_key=self.api_key,
            model=model or self.api_type,
            temperature=temperature,
            top_p=0.1
        )
    
    def create_llm(self, model: str, available_models: Dict[str, List[Dict]], streaming: bool = True, temperature: float = 0.1):
        """Create LMStudio LLM for main chat (uses OpenAI-compatible API)"""
    
        
        # Check if model is available in LMStudio
        lmstudio_models = available_models.get("lmstudioModels", [])
        if any(m.get("name") == model for m in lmstudio_models):
            selected_model = model
        else:
            selected_model = self.api_type
        
        return ChatOpenAI(
            openai_api_base=self.host,
            openai_api_key=self.api_key,
            model=selected_model,
            temperature=temperature,
            top_p=0.1,
            streaming=streaming
        )
    
    async def get_available_models(self) -> Dict[str, List[Dict]]:
        """Get available LMStudio models from API"""
        try:
            # Fetch models from LMStudio OpenAI-compatible API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/v1/models", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Format models from LMStudio API response
                        lmstudio_models: List[Dict[str, Any]] = []
                        for model in data.get("data", []):
                            model_id = model.get("id", "") or ""
                            lmstudio_models.append(
                                {
                                    "name": model_id,
                                    "provider": "lmstudio",
                                    "is_toolCall_available": True,
                                    "is_vision_available": False,
                                    "is_support_thinking": False,
                                    "is_shown": False,
                                }
                            )
                        
                        # Sort models by name for consistent ordering
                        lmstudio_models.sort(key=lambda x: x["name"])
                        
                        return {
                            "lmstudioModels": lmstudio_models
                        }
                    else:
                        logger.warning(f"LMStudio API returned status {response.status}")
                        return {
                            "lmstudioModels": []
                        }
                        
        except Exception as e:
            logger.error(f"Failed to fetch LMStudio models: {str(e)}")
            # Fallback to default models if API call fails
            models_from_env = get_models_from_env("lmstudio")
            if models_from_env:
                log_names = models_from_env
            else:
                log_names = [
                    "gpt-3.5-turbo",
                    "llama-2-7b-chat",
                    "codellama-7b-instruct",
                    "mistral-7b-instruct",
                    "local-model",
                ]
            
            default_models: List[Dict[str, Any]] = []
            for name in log_names:
                default_models.append(
                    {
                        "name": name,
                        "provider": "lmstudio",
                        "is_toolCall_available": True,
                        "is_vision_available": False,
                        "is_support_thinking": False,
                        "is_shown": False,
                    }
                )

            return {
                "lmstudioModels": default_models
            }

    async def get_models_from_provider(self) -> Dict[str, List[Dict]]:
        """
        Strictly fetch available LMStudio models from the provider API.

        - On success: return models fetched from API.
        - On any error: raise the exception (no local fallback).
        """
        try:
            logger.info(
                "LMStudioProvider.get_models_from_provider: fetching models from %s",
                self.host,
            )

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/v1/models", headers=headers
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(
                            f"LMStudio API returned status {response.status}: {text}"
                        )

                    data = await response.json()
                    lmstudio_models: List[Dict[str, Any]] = []
                    for model in data.get("data", []):
                        model_id = model.get("id", "") or ""
                        lmstudio_models.append(
                            {
                                "name": model_id,
                                "provider": "lmstudio",
                                # Treat LMStudio models as tool-capable by default; they use
                                # an OpenAI-compatible API and support tools for most models.
                                "is_toolCall_available": True,
                                # Vision capability is not generally advertised; default to False.
                                "is_vision_available": False,
                                # Reasoning/"thinking" support depends on the loaded model;
                                # expose this as False by default and allow UI to override.
                                "is_support_thinking": False,
                                "is_shown": False,
                            }
                        )

                    lmstudio_models.sort(key=lambda x: x["name"])

                    return {"lmstudioModels": lmstudio_models}
        except Exception as e:
            logger.error(
                "LMStudioProvider.get_models_from_provider: failed to fetch models from %s: %s",
                self.host,
                e,
            )
            raise
    
    def is_tool_call_available(self, llm) -> bool:
        """LMStudio models always support tool calling."""
        return True
