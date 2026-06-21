"""Provider-neutral LLM completion clients."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


class LLMClient(Protocol):
    def complete(self, prompt: str, *, model: str, max_tokens: int) -> str:
        """Return text generated for ``prompt``."""


@dataclass
class AnthropicLLMClient:
    api_key: str
    client: object | None = None

    def complete(self, prompt: str, *, model: str, max_tokens: int) -> str:
        client = self.client
        if client is None:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)
            self.client = client
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


@dataclass
class OllamaLLMClient:
    base_url: str = "http://localhost:11434"

    def complete(self, prompt: str, *, model: str, max_tokens: int) -> str:
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0,
                },
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


def make_llm_client(config) -> LLMClient:
    provider = config.llm_provider.strip().lower()
    if provider == "anthropic":
        return AnthropicLLMClient(api_key=config.anthropic_api_key)
    if provider == "ollama":
        return OllamaLLMClient(base_url=config.ollama_url)
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
