"""Provider-neutral LLM completion clients."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol

import requests


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks that thinking models embed in content.

    Safe no-op for non-thinking models. Handles bleed-through automatically
    without requiring the caller to know whether the model is a thinking model.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(
        self, prompt: str, *, model: str, max_tokens: int, thinking: bool = True
    ) -> str:
        """Return text generated for ``prompt``.

        ``thinking=False`` asks a reasoning model to answer directly, without
        its hidden reasoning step (much faster; fine for simple tasks).
        """


@dataclass
class AnthropicLLMClient:
    api_key: str
    client: object | None = None

    def complete(
        self, prompt: str, *, model: str, max_tokens: int, thinking: bool = True
    ) -> str:
        # Extended thinking is never requested here, so ``thinking`` changes nothing.
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


class LLMUnavailableError(ValueError):
    """The LLM server could not be reached or kept failing after all retries.

    Raised for outages (network errors, timeouts, 429, 5xx), not for requests
    the server rejects outright (bad model name, bad key), so the pipeline can
    tell a temporary outage from a configuration error.
    """


_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_DELAYS = (5, 15, 45)          # pauses between the 4 attempts (~1 min total)
_MAX_RETRY_AFTER = 120               # cap on a server-requested Retry-After wait
_TIMEOUT = (10, 300)                 # (connect, read) seconds; thinking models answer slowly
_EMPTY_ATTEMPTS = 2                  # empty answers are retried once, immediately


def _retry_after(resp) -> float | None:
    """Seconds from a numeric Retry-After header, capped; None if absent or unparsable."""
    try:
        return min(float(resp.headers.get("Retry-After", "")), _MAX_RETRY_AFTER)
    except (TypeError, ValueError):
        return None


@dataclass
class OpenAICompatibleLLMClient:
    base_url: str
    api_key: str

    def _post(
        self, prompt: str, model: str, max_tokens: int, thinking: bool = True
    ) -> requests.Response:
        """POST the chat request, retrying outages with increasing pauses.

        Raises LLMUnavailableError when every attempt hit a network error,
        timeout, 429 or 5xx, and ValueError at once for any other HTTP error.
        """
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
            "options": {"num_ctx": 32768},
        }
        if not thinking:
            # Honoured by vLLM-served Qwen models behind the LiteLLM proxy
            # (llm.science-berlin.de); only sent when thinking is switched off.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        attempts = len(_RETRY_DELAYS) + 1
        for attempt in range(1, attempts + 1):
            wait = _RETRY_DELAYS[attempt - 1] if attempt < attempts else 0
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=_TIMEOUT,
                )
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status not in _RETRY_STATUS:
                    raise ValueError(f"LLM request to {self.base_url} rejected: {exc}") from exc
                error = exc
                if exc.response is not None:
                    wait = _retry_after(exc.response) or wait
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                error = exc
            except requests.exceptions.RequestException as exc:
                raise ValueError(f"LLM request to {self.base_url} failed: {exc}") from exc
            if attempt == attempts:
                raise LLMUnavailableError(
                    f"LLM server unavailable at {self.base_url} after {attempts} attempts: {error}"
                ) from error
            log.warning(
                "LLM request failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt, attempts, error, wait,
            )
            time.sleep(wait)
        raise AssertionError("unreachable")

    def complete(
        self, prompt: str, *, model: str, max_tokens: int, thinking: bool = True
    ) -> str:
        log.info(
            "LLM request → openai-compat/%s (prompt %d chars%s)",
            model, len(prompt), "" if thinking else ", thinking off",
        )
        for attempt in range(1, _EMPTY_ATTEMPTS + 1):
            data = self._post(prompt, model, max_tokens, thinking).json()
            choices = data.get("choices") or []
            if not choices:
                log.warning("Empty choices on attempt %d", attempt)
                continue
            message = choices[0].get("message") or {}
            usage = data.get("usage") or {}
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            content = _strip_thinking(message.get("content") or "")
            if content:
                log.info("LLM response ← %r", content[:300])
                return content
            if reasoning_tokens:
                log.warning(
                    "content empty after stripping thinking blocks (attempt %d); "
                    "model used %d reasoning tokens — thinking model may have "
                    "exhausted budget before writing answer",
                    attempt, reasoning_tokens,
                )
            else:
                log.warning(
                    "Empty response on attempt %d; message keys: %s",
                    attempt, list(message.keys()),
                )
        return ""


def assert_openai_compatible_server_available(
    base_url: str,
    *,
    api_key: str = "",
    retries: int = 2,
    delay: float = 3.0,
) -> None:
    """Raise ValueError if the OpenAI-compatible server is not usable.

    Lists the models via GET {base_url}/models (standard OpenAI API, served by
    Ollama, LiteLLM, vLLM, ...), sending ``api_key`` as bearer token so a bad
    key is caught before any paper is processed. Network and server errors are
    retried up to ``retries`` times, ``delay`` seconds apart; a rejected key
    (401/403) fails immediately.
    """
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code in (401, 403):
                raise ValueError(
                    f"LLM server at {base_url} rejected the API key (HTTP {resp.status_code})"
                )
            resp.raise_for_status()
            log.info("LLM server reachable at %s", base_url)
            return
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt <= retries:
                log.warning(
                    "LLM server not reachable (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, retries + 1, exc, delay,
                )
                time.sleep(delay)
    raise ValueError(f"LLM server unavailable at {base_url}: {last_exc}") from last_exc


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
