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
    assert captured["timeout"] == (10, 300)   # (connect, read)


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


def _chat_resp(status=200, content="answer", headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            f"{status} error", response=resp
        )
    return resp


def _complete_with(responses):
    """Run complete() against scripted responses; return (result_or_exc, post_mock, sleep_mock)."""
    client = OpenAICompatibleLLMClient(base_url="https://llm.example.org/v1", api_key="k")
    with patch("topic_overviews.llm.client.requests.post", side_effect=responses) as post, \
         patch("topic_overviews.llm.client.time.sleep") as sleep:
        try:
            result = client.complete("p", model="m", max_tokens=10)
        except Exception as exc:  # returned so tests can assert on it
            result = exc
    return result, post, sleep


def test_complete_retries_timeout_and_server_errors():
    result, post, sleep = _complete_with([
        requests.exceptions.ReadTimeout("slow"),
        _chat_resp(503),
        _chat_resp(200, "ok"),
    ])
    assert result == "ok"
    assert post.call_count == 3
    assert [c.args[0] for c in sleep.call_args_list] == [5, 15]


def test_complete_honours_retry_after_on_rate_limit():
    result, post, sleep = _complete_with([
        _chat_resp(429, headers={"Retry-After": "30"}),
        _chat_resp(200, "ok"),
    ])
    assert result == "ok"
    assert [c.args[0] for c in sleep.call_args_list] == [30]


def test_complete_does_not_retry_client_errors():
    from topic_overviews.llm.client import LLMUnavailableError

    for status in (400, 401):
        result, post, sleep = _complete_with([_chat_resp(status)])
        assert isinstance(result, ValueError)
        assert not isinstance(result, LLMUnavailableError)   # a config problem, not an outage
        assert post.call_count == 1
        sleep.assert_not_called()


def test_complete_raises_llm_unavailable_after_all_attempts():
    from topic_overviews.llm.client import LLMUnavailableError

    result, post, sleep = _complete_with(
        [requests.exceptions.ConnectionError("down")] * 4
    )
    assert isinstance(result, LLMUnavailableError)
    assert isinstance(result, ValueError)                     # callers catching ValueError still work
    assert post.call_count == 4
    assert [c.args[0] for c in sleep.call_args_list] == [5, 15, 45]


def test_complete_retries_empty_answer_once_without_backoff():
    result, post, sleep = _complete_with([_chat_resp(200, ""), _chat_resp(200, "")])
    assert result == ""
    assert post.call_count == 2
    sleep.assert_not_called()


def test_complete_disables_thinking_only_when_asked():
    client = OpenAICompatibleLLMClient(base_url="https://llm.example.org/v1", api_key="k")
    with patch("topic_overviews.llm.client.requests.post",
               side_effect=[_chat_resp(200, "a"), _chat_resp(200, "b")]) as post:
        client.complete("p", model="m", max_tokens=10)
        client.complete("p", model="m", max_tokens=10, thinking=False)

    default_body, off_body = (c.kwargs["json"] for c in post.call_args_list)
    assert "chat_template_kwargs" not in default_body
    assert off_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_anthropic_client_accepts_thinking_flag():
    fake = SimpleNamespace(messages=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(content=[SimpleNamespace(text="hi")])))
    assert AnthropicLLMClient(api_key="k", client=fake).complete(
        "p", model="m", max_tokens=5, thinking=False) == "hi"
