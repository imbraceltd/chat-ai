import json
import logging
from typing import Dict, Any

import aiohttp

log = logging.getLogger(__name__)


def create_guardrail_service(provider_type: str, config: Dict[str, Any]) -> "BaseGuardrailProvider":
    """Factory: create the right guardrail provider based on type."""
    providers = {
        "openai": OpenAIModerationProvider,
        "nim-nemo": NIMNemoProvider,
        "llamaguard": LlamaGuardProvider,
        "ollama": OllamaGuardrailProvider,
        "custom": CustomHTTPProvider,
    }
    cls = providers.get(provider_type, CustomHTTPProvider)
    return cls(config)


class BaseGuardrailProvider:
    """Base class for all guardrail providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get("api_key", "")
        self.timeout = config.get("timeout", 30)

    def _bearer_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def check_safety(self, message: str, guardrail_config_id: str) -> Dict[str, Any]:
        raise NotImplementedError

    async def validate_connection(self) -> bool:
        raise NotImplementedError

    def _fail_closed(self, error: str) -> Dict[str, Any]:
        return {"is_safe": False, "safety_categories": f"Provider error: {error}"}


class OpenAIModerationProvider(BaseGuardrailProvider):
    """OpenAI Moderation API: POST https://api.openai.com/v1/moderations"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "omni-moderation-latest")

    async def check_safety(self, message: str, guardrail_config_id: str) -> Dict[str, Any]:
        url = "https://api.openai.com/v1/moderations"
        headers = self._bearer_headers()
        payload = {
            "model": self.model,
            "input": [{"type": "text", "text": message}],
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    result = data.get("results", [{}])[0]
                    flagged = result.get("flagged", False)
                    categories = result.get("categories", {})

                    violated = [cat.replace("/", " / ").replace("_", " ").title()
                                for cat, val in categories.items() if val]

                    return {
                        "is_safe": not flagged,
                        "safety_categories": ", ".join(violated) if violated else "None",
                    }
        except Exception as e:
            log.error(f"OpenAI Moderation error: {e}")
            return self._fail_closed(str(e))

    async def validate_connection(self) -> bool:
        try:
            result = await self.check_safety("hello", "test")
            return "Provider error" not in result.get("safety_categories", "")
        except Exception:
            return False


class NIMNemoProvider(BaseGuardrailProvider):
    """NVIDIA NIM NeMo: POST {base_url}/v1/chat/completions (safety model)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip("/")
        self.model = config.get("model_name", "llama-3.1-nemoguard-8b-content-safety")

    async def check_safety(self, message: str, guardrail_config_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._bearer_headers()

        system_prompt = (
            "Task: Check if there is unsafe content in the user message. "
            "Provide your safety assessment in JSON format:\n"
            '{"User Safety": "safe" or "unsafe", "Safety Categories": "comma-separated list"}\n'
            "Output only the JSON response."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message},
            ],
            "max_tokens": 300,
            "stream": False,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    text = data["choices"][0]["message"]["content"].strip()
                    try:
                        safety_info = json.loads(text)
                    except json.JSONDecodeError:
                        safety_info = {"User Safety": "unsafe", "Safety Categories": "Parse error"}

                    is_safe = safety_info.get("User Safety") != "unsafe"
                    categories = safety_info.get("Safety Categories", "Unknown")

                    return {
                        "is_safe": is_safe,
                        "safety_categories": categories if not is_safe else "None",
                    }
        except Exception as e:
            log.error(f"NIM NeMo provider error: {e}")
            return self._fail_closed(str(e))

    async def fetch_models(self) -> list:
        """Fetch available models from NIM via GET /v1/models."""
        url = f"{self.base_url}/v1/models"
        headers = self._bearer_headers()
        try:
            timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    models = []
                    for m in data.get("data", []):
                        model_id = m.get("id", "")
                        if model_id:
                            models.append({"name": model_id})
                    models.sort(key=lambda x: x["name"])
                    return models
        except Exception as e:
            log.error(f"Failed to fetch NIM models: {e}")
            return []

    async def validate_connection(self) -> bool:
        url = f"{self.base_url}/v1/models"
        headers = self._bearer_headers()
        try:
            timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    return resp.status < 500
        except Exception:
            return False


class LlamaGuardProvider(BaseGuardrailProvider):
    """LlamaGuard via vLLM/TGI: POST {base_url}/v1/chat/completions (OpenAI compatible)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip("/")
        self.model_name = config.get("model_name", "meta-llama/Llama-Guard-3-8B")
        self.max_tokens = config.get("max_tokens", 512)

    async def check_safety(self, message: str, guardrail_config_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = self._bearer_headers()

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    text = data["choices"][0]["message"]["content"].strip().lower()

                    # LlamaGuard returns "safe" or "unsafe\nS1,S2,..." format
                    is_safe = text.startswith("safe")
                    categories = "None"
                    if not is_safe:
                        lines = text.split("\n")
                        if len(lines) > 1:
                            categories = lines[1].strip()
                        else:
                            categories = "Unknown"

                    return {
                        "is_safe": is_safe,
                        "safety_categories": categories,
                    }
        except Exception as e:
            log.error(f"LlamaGuard provider error: {e}")
            return self._fail_closed(str(e))

    async def fetch_models(self) -> list:
        """Fetch available models from vLLM/TGI via GET /v1/models."""
        url = f"{self.base_url}/v1/models"
        headers = self._bearer_headers()
        try:
            timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    models = []
                    for m in data.get("data", []):
                        model_id = m.get("id", "")
                        if model_id:
                            models.append({"name": model_id})
                    models.sort(key=lambda x: x["name"])
                    return models
        except Exception as e:
            log.error(f"Failed to fetch LlamaGuard models: {e}")
            return []

    async def validate_connection(self) -> bool:
        url = f"{self.base_url}/v1/models"
        headers = self._bearer_headers()
        try:
            timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    return resp.status < 500
        except Exception:
            return False


class OllamaGuardrailProvider(BaseGuardrailProvider):
    """Ollama: POST {base_url}/api/chat with a safety classification model."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")
        self.model_name = config.get("model_name", "llama-guard3")

    async def check_safety(self, message: str, guardrail_config_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                    text = data.get("message", {}).get("content", "").strip().lower()

                    # LlamaGuard format: "safe" or "unsafe\nS1,S2,..."
                    is_safe = text.startswith("safe")
                    categories = "None"
                    if not is_safe:
                        lines = text.split("\n")
                        categories = lines[1].strip() if len(lines) > 1 else "Unknown"

                    return {
                        "is_safe": is_safe,
                        "safety_categories": categories,
                    }
        except Exception as e:
            log.error(f"Ollama guardrail provider error: {e}")
            return self._fail_closed(str(e))

    async def fetch_models(self) -> list:
        """Fetch available models from Ollama via GET /api/tags."""
        url = f"{self.base_url}/api/tags"
        try:
            timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    models = []
                    for m in data.get("models", []):
                        name = m.get("name", "") or ""
                        if name:
                            models.append({"name": name, "size": m.get("size")})
                    models.sort(key=lambda x: x["name"])
                    return models
        except Exception as e:
            log.error(f"Failed to fetch Ollama models from {url}: {e}")
            return []

    async def validate_connection(self) -> bool:
        url = f"{self.base_url}/api/tags"
        try:
            timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    return resp.status < 500
        except Exception:
            return False


class CustomHTTPProvider(BaseGuardrailProvider):
    """Generic custom HTTP guardrail: POST {base_url}{check_endpoint}
    Contract: { "message": str } -> { "is_safe": bool, "safety_categories": str }
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "").rstrip("/")
        self.check_endpoint = config.get("check_endpoint", "/check")

    async def check_safety(self, message: str, guardrail_config_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}{self.check_endpoint}"
        headers = self._bearer_headers()
        payload = {
            "message": message,
            "guardrail_config_id": guardrail_config_id,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return {
                        "is_safe": bool(data.get("is_safe", False)),
                        "safety_categories": data.get("safety_categories", "Unknown"),
                    }
        except Exception as e:
            log.error(f"Custom guardrail provider error ({url}): {e}")
            return self._fail_closed(str(e))

    async def validate_connection(self) -> bool:
        url = f"{self.base_url}{self.check_endpoint}"
        headers = self._bearer_headers()
        payload = {"message": "test connection", "guardrail_config_id": "test"}

        try:
            timeout = aiohttp.ClientTimeout(total=min(self.timeout, 10))
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    return resp.status < 500
        except Exception:
            return False


# Backward compatibility alias
CustomGuardrailService = CustomHTTPProvider
