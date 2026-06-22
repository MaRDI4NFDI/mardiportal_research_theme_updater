"""Provider-neutral LLM completion clients."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import requests

log = logging.getLogger(__name__)


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
        log.info("LLM request → anthropic/%s (prompt %d chars)", model, len(prompt))
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        log.info("LLM response ← %r", text[:300])
        return text


@dataclass
class OpenAICompatibleLLMClient:
    base_url: str
    api_key: str

    def complete(self, prompt: str, *, model: str, max_tokens: int) -> str:
        log.info("LLM request → openai-compat/%s (prompt %d chars)", model, len(prompt))
        resp = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": False,
                # Ollama-specific: extend context window for reasoning models
                # whose chain-of-thought can exhaust the default 8192-token limit
                "options": {"num_ctx": 32768},
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        text = content if isinstance(content, str) else str(content)
        log.info("LLM response ← %r", text[:300])
        return text


def make_llm_client(config) -> LLMClient:
    provider = config.llm_provider.strip().lower()
    if provider == "anthropic":
        return AnthropicLLMClient(api_key=config.anthropic_api_key)
    if provider == "openai":
        return OpenAICompatibleLLMClient(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
        )
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
