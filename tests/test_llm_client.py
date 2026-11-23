from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests

import pytest

from topic_overviews.config import load_config
from topic_overviews.llm.client import (
    AnthropicLLMClient,
    OpenAICompatibleLLMClient,
    _strip_thinking,
    make_llm_client,
)


class FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, *, model, max_tokens, messages):
        self.calls.append((model, max_tokens, messages))
        return SimpleNamespace(content=[SimpleNamespace(text="done")])


def test_anthropic_llm_client_returns_text():
    messages = FakeAnthropicMessages()
    client = AnthropicLLMClient(
        api_key="sk-test",
        client=SimpleNamespace(messages=messages),
    )
    assert client.complete("prompt", model="m", max_tokens=12) == "done"
    assert messages.calls == [("m", 12, [{"role": "user", "content": "prompt"}])]


def test_openai_compatible_llm_client_posts_chat_completion_request():
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "local answer"}}
                ]
            }

    def fake_post(url, *, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResp()

    import topic_overviews.llm.client as client_module

    original_post = client_module.requests.post
    client_module.requests.post = fake_post
    try:
        client = OpenAICompatibleLLMClient(
            base_url="https://ollama.zib.de/api",
            api_key="secret",
        )
        assert client.complete("prompt", model="llama", max_tokens=7) == "local answer"
    finally:
        client_module.requests.post = original_post

    assert captured["url"] == "https://ollama.zib.de/api/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"] == {
        "model": "llama",
        "messages": [{"role": "user", "content": "prompt"}],
        "max_tokens": 7,
        "temperature": 0,
        "stream": False,
        "options": {"num_ctx": 32768},
    }
    assert captured["timeout"] == 300


def test_strip_thinking_removes_think_block():
    raw = "<think>\nsome internal reasoning\n</think>\nActual answer."
    assert _strip_thinking(raw) == "Actual answer."


def test_strip_thinking_noop_for_plain_text():
    assert _strip_thinking("Plain answer.") == "Plain answer."


def test_strip_thinking_handles_multiline_block():
    raw = "<think>\nstep 1\nstep 2\n</think>Final."
    assert _strip_thinking(raw) == "Final."


def test_openai_compatible_strips_thinking_from_content():
    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [{"message": {"content": "<think>internal</think>Clean answer."}}],
                "usage": {"completion_tokens_details": {"reasoning_tokens": 42}},
            }

    import topic_overviews.llm.client as client_module
    original_post = client_module.requests.post
    client_module.requests.post = lambda *a, **kw: FakeResp()
    try:
        client = OpenAICompatibleLLMClient(base_url="http://x", api_key="k")
        assert client.complete("p", model="m", max_tokens=10) == "Clean answer."
    finally:
        client_module.requests.post = original_post


def test_make_llm_client_selects_provider():
    assert isinstance(make_llm_client(load_config({})), AnthropicLLMClient)
    assert isinstance(
        make_llm_client(
            load_config(
                {
                    "TOPIC_OVERVIEWS_LLM_PROVIDER": "openai",
                    "TOPIC_OVERVIEWS_OPENAI_BASE_URL": "https://ollama.zib.de/api",
                    "TOPIC_OVERVIEWS_OPENAI_API_KEY": "secret",
                }
            )
        ),
        OpenAICompatibleLLMClient,
    )


def test_make_llm_client_rejects_unknown_provider():
    cfg = load_config({"TOPIC_OVERVIEWS_LLM_PROVIDER": "nope"})
    with pytest.raises(ValueError):
        make_llm_client(cfg)


def _models_response(status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status} error")
    return resp


def test_server_check_lists_models_with_bearer_token():
    from topic_overviews.llm.client import assert_openai_compatible_server_available

    with patch("topic_overviews.llm.client.requests.get", return_value=_models_response()) as get:
        assert_openai_compatible_server_available("https://llm.example.org/v1/", api_key="sk-1")

    get.assert_called_once()
    assert get.call_args.args[0] == "https://llm.example.org/v1/models"
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-1"}


def test_server_check_fails_fast_on_rejected_key():
    from topic_overviews.llm.client import assert_openai_compatible_server_available

    with patch("topic_overviews.llm.client.requests.get", return_value=_models_response(401)) as get:
        with pytest.raises(ValueError, match="rejected the API key"):
            assert_openai_compatible_server_available("https://llm.example.org/v1", api_key="bad")

    assert get.call_count == 1      # an auth error is not retried


def test_server_check_retries_then_raises_when_unreachable():
    from topic_overviews.llm.client import assert_openai_compatible_server_available

    with patch(
        "topic_overviews.llm.client.requests.get",
        side_effect=requests.exceptions.ConnectionError("down"),
    ) as get:
        with pytest.raises(ValueError, match="unavailable"):
            assert_openai_compatible_server_available("https://llm.example.org/v1", retries=2)

    assert get.call_count == 3
