from types import SimpleNamespace

import pytest

from topic_overviews.config import load_config
from topic_overviews.llm.client import (
    AnthropicLLMClient,
    OllamaLLMClient,
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


def test_ollama_llm_client_posts_generate_request():
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "local answer"}

    def fake_post(url, *, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResp()

    import topic_overviews.llm.client as client_module

    original_post = client_module.requests.post
    client_module.requests.post = fake_post
    try:
        client = OllamaLLMClient(base_url="http://ollama:11434/")
        assert client.complete("prompt", model="llama", max_tokens=7) == "local answer"
    finally:
        client_module.requests.post = original_post

    assert captured["url"] == "http://ollama:11434/api/generate"
    assert captured["json"] == {
        "model": "llama",
        "prompt": "prompt",
        "stream": False,
        "options": {"num_predict": 7, "temperature": 0},
    }
    assert captured["timeout"] == 300


def test_make_llm_client_selects_provider():
    assert isinstance(make_llm_client(load_config({})), AnthropicLLMClient)
    assert isinstance(
        make_llm_client(
            load_config(
                {
                    "TOPIC_OVERVIEWS_LLM_PROVIDER": "ollama",
                    "TOPIC_OVERVIEWS_OLLAMA_URL": "http://localhost:11434",
                }
            )
        ),
        OllamaLLMClient,
    )


def test_make_llm_client_rejects_unknown_provider():
    cfg = load_config({"TOPIC_OVERVIEWS_LLM_PROVIDER": "nope"})
    with pytest.raises(ValueError):
        make_llm_client(cfg)
