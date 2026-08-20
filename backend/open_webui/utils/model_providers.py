"""
Model provider factory functions for Document AI processing.

This module provides factory functions for creating LLM and vision model instances
for different providers: OpenAI, Google Gemini, Ollama, LMStudio, Kimi, and OLM.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

from openai import OpenAI
import httpx

from open_webui.config import (
    OPENAI_API_KEY,
    GOOGLE_API_KEY,
    KIMI_API_KEY,
    KIMI_CONFIG,
    OLM_API_KEY,
    OLM_CONFIG,
    OLLAMA_URL,
    LMSTUDIO_HOST,
)

log = logging.getLogger(__name__)


def ollama_llm(model_name: str) -> Optional[Any]:
    """
    Create a ChatOllama instance for Ollama models.
    
    Args:
        model_name: Name of the Ollama model
        
    Returns:
        ChatOllama instance or None if not available
    """
    if ChatOllama is None:
        log.warning("ChatOllama not available. Install langchain-ollama package.")
        return None
        
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_URL,
        temperature=0.1,
    )


def google_llm(model_name: str, api_key: str = GOOGLE_API_KEY) -> ChatGoogleGenerativeAI:
    """
    Create a ChatGoogleGenerativeAI instance for Google models.

    Args:
        model_name: Name of the Google model
        api_key: Google API key (defaults to GOOGLE_API_KEY from env)

    Returns:
        ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=api_key,
    )


def lmstudio_llm(model_name: str) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance configured for LMStudio.
    
    Args:
        model_name: Name of the LMStudio model
        
    Returns:
        ChatOpenAI instance configured for LMStudio
    """
    return ChatOpenAI(
        base_url=LMSTUDIO_HOST,
        model=model_name,
        api_key=OPENAI_API_KEY or "not-needed",
        temperature=0,
    )


def kimi_vision_model() -> OpenAI:
    """
    Create an OpenAI client instance for Kimi vision model.
    
    Returns:
        OpenAI client configured for Kimi
    """
    log.info(f'Initializing Kimi vision model with config: {KIMI_CONFIG}')
    
    base_url = KIMI_CONFIG["host"]
    model_name = KIMI_CONFIG["model"]
    
    # Create HTTP client with appropriate timeout
    http_client = httpx.Client(
        timeout=httpx.Timeout(300.0),  # 5 minutes timeout
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
    )
    
    openai_config = {
        "base_url": base_url,
        "http_client": http_client,
        "max_retries": 2,
    }
    
    # Only add API key if it exists
    if KIMI_API_KEY:
        openai_config["api_key"] = KIMI_API_KEY
    else:
        log.info('No KIMI_API_KEY provided, using without authentication')
        openai_config["api_key"] = "not-needed"
    
    client = OpenAI(
        base_url=base_url,
        http_client=http_client,
        max_retries=2,
    )
    # Store model name for reference
    client.model = model_name
    
    return client


class OlmClient:
    """
    Custom OLM client that mimics OpenAI API structure.
    """
    
    def __init__(self, base_url: str, model: str, timeout: int = 300000, 
                 max_retries: int = 2, api_key: str = "testing"):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout / 1000  # Convert to seconds
        self.max_retries = max_retries
        self.api_key = api_key
        
        # Create nested structure to match OpenAI client
        self.chat = ChatCompletions(self)
        
    
class ChatCompletions:
    """Chat completions endpoint for OLM client."""
    
    def __init__(self, client):
        self.client = client
    
    async def create(self, model: str, messages: List[Dict], **kwargs):
        """
        Create a chat completion using OLM/Ollama API.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            **kwargs: Additional parameters
            
        Returns:
            Response object with choices
        """
        import httpx
        
        # Remove response_format if it exists - Ollama doesn't support it
        kwargs_clean = {k: v for k, v in kwargs.items() if k != 'response_format'}
        
        async with httpx.AsyncClient(timeout=self.client.timeout) as http_client:
            try:
                # Try OpenAI-compatible endpoint first (Ollama with llama.cpp)
                response = await http_client.post(
                    f"{self.client.base_url}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        **kwargs_clean
                    },
                    headers={
                        "Authorization": f"Bearer {self.client.api_key}",
                        "Content-Type": "application/json",
                    }
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # If v1 endpoint fails, try without v1 prefix
                log.warning(f"v1 endpoint failed, trying without prefix: {e}")
                try:
                    response = await http_client.post(
                        f"{self.client.base_url}/chat/completions",
                        json={
                            "model": model,
                            "messages": messages,
                            **kwargs_clean
                        },
                        headers={
                            "Authorization": f"Bearer {self.client.api_key}",
                            "Content-Type": "application/json",
                        }
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e2:
                    log.error(f"Error calling OLM/Ollama API: {e2}")
                    raise
            except Exception as e:
                log.error(f"Error calling OLM/Ollama API: {e}")
                raise


def olm_vision_model() -> OlmClient:
    """
    Create an OLM client instance for OLM OCR model.
    
    Returns:
        OlmClient configured for OLM
    """
    log.info(f'Initializing OLM vision model with custom client: {OLM_CONFIG}')
    
    client = OlmClient(
        base_url=OLM_CONFIG["host"],
        model=OLM_CONFIG["model"],
        timeout=300000,  # 5 minutes timeout
        max_retries=2,
        api_key=OLM_API_KEY or "testing",
    )
    
    return client


def google_vision_model(model_name: str = "gemini-2.5-flash", api_key: str = GOOGLE_API_KEY) -> ChatGoogleGenerativeAI:
    """
    Create a ChatGoogleGenerativeAI instance for Google vision models.

    Args:
        model_name: Name of the Google vision model
        api_key: Google API key (defaults to GOOGLE_API_KEY from env)

    Returns:
        ChatGoogleGenerativeAI instance
    """
    # Map pro to correct model name

    log.info(f"Using Google Vision model: {model_name}")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        top_p=0.1,
        google_api_key=api_key,
    )


def openai_llm(model_name: str) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance for OpenAI models.
    
    Args:
        model_name: Name of the OpenAI model
        
    Returns:
        ChatOpenAI instance
    """
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=model_name,
        temperature=0,
        max_tokens=3000,
        timeout=60,
    )


class ModelService:
    """
    Service for retrieving available models from different providers.
    """
    
    @staticmethod
    async def get_models() -> List[Dict[str, Any]]:
        """
        Get list of available models from all configured providers.
        
        Returns:
            List of model dictionaries with name, provider, and capabilities
        """
        models = []
        
        # Add OpenAI models
        models.extend([
            {
                "name": "gpt-4o",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "gpt-4o-mini",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "gpt-4o-2024-08-06",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "o1-2024-12-17",
                "provider": "openai",
                "is_toolCall_available": False,
                "is_vision_available": False,
            },
            {
                "name": "gpt-4.1",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "gpt-4.1-nano",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
                                   {
                "name": "gpt-4.1-mini",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
                                   {
                "name": "gpt-5-mini",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
                                   {
                "name": "gpt-5-nano",
                "provider": "openai",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
        ])
        
        # Add Google models
        models.extend([
            {
                "name": "gemini-2.0-flash-exp",
                "provider": "google",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "gemini-1.5-pro",
                "provider": "google",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "gemini-2.5-pro",
                "is_toolCall_available": True,
                "is_vision_available": True,
                "provider": "google",
            },
            {
                "name": "gemini-2.5-flash",
                "is_toolCall_available": True,
                "is_vision_available": True,
                "provider": "google",
            },
        ])
        
        # Add Bedrock models
        models.extend([
            {
                "name": "qwen.qwen3-vl-235b-a22b",
                "provider": "bedrock",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "moonshotai.kimi-k2.5",
                "provider": "bedrock",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "amazon.nova-pro-v1:0",
                "provider": "bedrock",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
            {
                "name": "mistral.ministral-3-14b-instruct",
                "provider": "bedrock",
                "is_toolCall_available": True,
                "is_vision_available": True,
            },
        ])

        # Add Kimi model if configured
        if KIMI_CONFIG["host"]:
            models.append({
                "name": KIMI_CONFIG["model"],
                "provider": "kimi",
                "is_toolCall_available": False,
                "is_vision_available": True,
            })
        
        # Add OLM model if configured
        if OLM_CONFIG["host"]:
            models.append({
                "name": OLM_CONFIG["model"],
                "provider": "kimi",  # Use kimi provider for compatibility
                "is_toolCall_available": False,
                "is_vision_available": True,
            })
        
        # Try to get Ollama models if available
        if ChatOllama is not None and OLLAMA_URL:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{OLLAMA_URL}/api/tags")
                    if response.status_code == 200:
                        ollama_data = response.json()
                        for model in ollama_data.get("models", []):
                            models.append({
                                "name": model.get("name"),
                                "provider": "ollama",
                                "is_toolCall_available": True,
                                "is_vision_available": False,
                            })
            except Exception as e:
                log.warning(f"Could not fetch Ollama models: {e}")
        
        # Try to get LMStudio models if available
        if LMSTUDIO_HOST:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{LMSTUDIO_HOST}/models")
                    if response.status_code == 200:
                        lmstudio_data = response.json()
                        for model in lmstudio_data.get("data", []):
                            models.append({
                                "name": model.get("id"),
                                "provider": "lmstudio",
                                "is_toolCall_available": True,
                                "is_vision_available": False,
                            })
            except Exception as e:
                log.warning(f"Could not fetch LMStudio models: {e}")
        
        return models
