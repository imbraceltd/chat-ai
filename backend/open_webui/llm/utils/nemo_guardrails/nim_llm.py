import asyncio
import json
import logging
from typing import List, Dict, Any, Optional

import requests
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult, Generation
from nemoguardrails.llm.providers import register_llm_provider
from pydantic import Field

from open_webui.env import NIM_API_BASE, NIM_API_KEY, NIM_MODEL

logger = logging.getLogger(__name__)

# Module-level store for per-org safety configs (populated by guardrails_loader)
ORG_CONFIGS: Dict[str, Dict[str, Any]] = {}


class SelfHostedNIM(BaseLLM):
    api_key: str = Field(...)
    base_url: str = Field(...)
    model: str = Field(default="llama-3.1-nemoguard-8b-content-safety")
    headers: Dict[str, str] = Field(...)

    def __init__(self, **kwargs):
        config = kwargs.get("config", {})
        parameters = config.get("parameters", {})

        init_dict = {
            "api_key": parameters.get("api_key") or kwargs.get("api_key", NIM_API_KEY),
            "base_url": parameters.get("base_url") or kwargs.get("base_url", NIM_API_BASE),
            "model": config.get("model", NIM_MODEL),
            "headers": {
                "Authorization": f"Bearer {parameters.get('api_key') or kwargs.get('api_key', NIM_API_KEY)}",
                "Content-Type": "application/json",
            },
        }

        super().__init__(**init_dict)

    def _generate(self, prompt: str, **kwargs) -> LLMResult:
        try:
            kwargs.pop("temperature", None)

            org_id = "default"
            messages = kwargs.get("messages", [])
            for message in messages:
                if message.get("role") == "context" and isinstance(
                    message.get("content"), dict
                ):
                    org_id = message["content"].get("org_id", org_id)
                    break
            org_id = kwargs.get("org_id", org_id)

            logger.debug(f"NIM safety check for org: {org_id}")
            org_config = ORG_CONFIGS.get(
                org_id, {"safety_categories": [], "competitor_keywords": []}
            )
            safety_categories = org_config["safety_categories"]
            competitor_keywords = org_config["competitor_keywords"]

            system_prompt = (
                "Task: Check if there is unsafe content in the user message according to the safety policy. "
                "Provide a safe or unsafe 'rating' for the user message. If unsafe, include a list of violated categories. "
                "References to competitor products or brands are considered unsafe under 'Competitor Reference'. "
                "Any content related to bomb-making or explosives is explicitly unsafe under 'Violence', 'Illegal Activity', and 'Criminal Planning/Confessions'.\n"
                "<BEGIN UNSAFE CONTENT CATEGORIES>\n"
                + "\n".join(
                    f"S{i + 1}: {cat}" for i, cat in enumerate(safety_categories)
                )
                + (
                    f"\nS{len(safety_categories) + 1}: Competitor Reference"
                    if competitor_keywords
                    else ""
                )
                + "\n<END UNSAFE CONTENT CATEGORIES>\n"
                "<BEGIN COMPETITOR KEYWORDS>\n"
                + (
                    ", ".join(competitor_keywords)
                    if competitor_keywords
                    else "None"
                )
                + "\n<END COMPETITOR KEYWORDS>\n"
                "<BEGIN CONVERSATION>\n"
                f"user: {prompt}\n"
                "<END CONVERSATION>\n"
                'Provide your safety assessment in JSON format:\n'
                '{"User Safety": "safe" or "unsafe", "Safety Categories": "comma-separated list of '
                'applicable safety categories" (omit if all safe)}\n'
                "Output only the JSON response."
            )

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": kwargs.get("max_tokens", 300),
                "top_p": kwargs.get("top_p", 1.0),
                "stream": False,
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"].strip()
            logger.debug(f"Raw NIM response: {text}")

            try:
                json_response = json.loads(text)
                return LLMResult(
                    generations=[[Generation(text=json.dumps(json_response))]]
                )
            except json.JSONDecodeError as e:
                logger.error(f"Malformed JSON response: {text}, Error: {str(e)}")
                return LLMResult(
                    generations=[
                        [
                            Generation(
                                text=json.dumps(
                                    {
                                        "User Safety": "unsafe",
                                        "Reason": "Malformed JSON response from model",
                                    }
                                )
                            )
                        ]
                    ]
                )
        except Exception as e:
            logger.error(f"NIM API call failed: {str(e)}")
            return LLMResult(
                generations=[
                    [
                        Generation(
                            text=json.dumps(
                                {
                                    "User Safety": "unsafe",
                                    "Reason": f"NIM API error: {str(e)}",
                                }
                            )
                        )
                    ]
                ]
            )

    async def _acall(self, prompt: str, **kwargs) -> str:
        result = await asyncio.to_thread(self._generate, prompt, **kwargs)
        return result.generations[0][0].text

    def _generate_multiple(self, prompts: List[str], **kwargs) -> LLMResult:
        generations = []
        for prompt in prompts:
            result = self._generate(prompt, **kwargs)
            generations.append(result.generations[0])
        return LLMResult(generations=generations)

    async def _agenerate(self, prompts: List[str], **kwargs) -> LLMResult:
        return await asyncio.to_thread(self._generate_multiple, prompts, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "self_hosted_nim"


register_llm_provider("self_hosted_nim", SelfHostedNIM)
