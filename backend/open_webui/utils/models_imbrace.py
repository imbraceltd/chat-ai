import httpx
import asyncio
from typing import List, Dict, Any
import logging
from open_webui.env import SRC_LOG_LEVELS
import os
from open_webui.config import OLLAMA_CONFIG, LLM_STUDIO_CONFIG
from open_webui.constants import MODEL_LIST

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS.get("RAG", logging.INFO))

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai,ollama").split(",")


print(f"LLM_PROVIDER: {LLM_PROVIDER}")


def map_ollama_models(ollama_api_response: dict) -> list:
    mapped_models = []
    for model in ollama_api_response.get("models", []):
        # Example mapping logic
        is_vision = "vision" in model.get("name", "")
        # Prefix all Ollama model names with "imbrace/"
        model_name = f"imbrace/{model.get('name')}"
        mapped_models.append(
            {
                "name": model_name,
                "is_toolCall_available": True,  # Assumption
                "is_vision_available": is_vision,
            }
        )
    return mapped_models


class ModelService:
    @staticmethod
    async def get_ollama_models() -> List[Dict[str, Any]]:
        """Fetches and processes models from an Ollama server."""
        url = f"{OLLAMA_CONFIG['host']}/api/tags"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            if not data or "models" not in data:
                return []

            # Map the response to the desired format
            return map_ollama_models(data)
        except httpx.RequestError as e:
            print(f"Error fetching Ollama models: {e}")
            return []

    @staticmethod
    async def get_lmstudio_models() -> List[Dict[str, Any]]:
        """Fetches and processes models from an LM Studio server."""
        url = f"{LLM_STUDIO_CONFIG['host']}/models"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

            # A list comprehension is the Pythonic equivalent of .map()
            lm_studio_models = [
                {
                    "name": model.get("id"),
                    "is_toolCall_available": True,  # Assumption based on original code
                    "is_vision_available": False,  # Assumption
                }
                for model in data.get("data", [])
            ]
            return lm_studio_models
        except httpx.RequestError as e:
            print(f"Error fetching LM Studio models: {e}")
            return []

    @staticmethod
    async def get_cloud_models() -> List[Dict[str, Any]]:
        """Returns a combined list of statically defined cloud models."""
        # The `+` operator concatenates lists, equivalent to the spread `...` syntax
        return (
            MODEL_LIST["OpenAIModels"]
            + MODEL_LIST["GoogleModels"]
            + MODEL_LIST["GroqModels"]
        )

    @staticmethod
    async def get_bedrock_models() -> List[Dict[str, Any]]:
        """Returns a combined list of statically defined Bedrock models."""
        return MODEL_LIST["BedrockModels"]

    @staticmethod
    async def get_models() -> List[Dict[str, Any]]:
        """
        The main method to fetch all models based on the LLM_PROVIDER config.
        """
        models = []

        # `in` is the Python equivalent of `array.includes()`
        if "ollama" in LLM_PROVIDER or "all" in LLM_PROVIDER:
            ollama_models = await ModelService.get_ollama_models()
            if ollama_models:
                # .extend() is the Pythonic way to append all items from another list
                models.extend(ollama_models)

        if "lmstudio" in LLM_PROVIDER or "all" in LLM_PROVIDER:
            lm_studio_models = await ModelService.get_lmstudio_models()
            models.extend(lm_studio_models)

        if "openai" in LLM_PROVIDER or "all" in LLM_PROVIDER:
            cloud_models = await ModelService.get_cloud_models()
            models.extend(cloud_models)
         
        if "bedrock" in LLM_PROVIDER or "all" in LLM_PROVIDER:
            bedrock_models = await ModelService.get_bedrock_models()
            models.extend(bedrock_models)

        return models


imbrace_models = ModelService()


async def get_ollama_models() -> List[Dict[str, Any]]:
    return await imbrace_models.get_ollama_models()


async def get_lmstudio_models() -> List[Dict[str, Any]]:
    return await imbrace_models.get_lmstudio_models()


async def get_cloud_models() -> List[Dict[str, Any]]:
    return await imbrace_models.get_cloud_models()


async def get_models() -> List[Dict[str, Any]]:
    return await imbrace_models.get_models()


# export
__all__ = ["get_ollama_models", "get_lmstudio_models", "get_cloud_models", "get_models"]
