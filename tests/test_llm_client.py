from types import SimpleNamespace

import pytest

from topic_overviews.config import load_config
from topic_overviews.llm.client import (
    AnthropicLLMClient,
    OpenAICompatibleLLMClient,
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
