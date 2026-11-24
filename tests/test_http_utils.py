import pytest
import requests

from topic_overviews import http_utils
from topic_overviews.http_utils import http_get, http_post


def _response(status, headers=None):
    resp = requests.Response()
    resp.status_code = status
    resp.headers.update(headers or {})
    return resp


class FakeSession:
    """Plays back a list of outcomes (Response or exception) for get and post."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def _next(self, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    get = post = _next


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(http_utils.time, "sleep", waited.append)
    return waited


def test_get_returns_after_transient_failures(sleeps):
    s = FakeSession([
        requests.exceptions.ConnectionError("down"),
        _response(404),
        _response(503),
        _response(200),
    ])
    assert http_get(s, "http://x").status_code == 200
    assert s.calls == 4
    assert sleeps == [5, 10, 20]  # backoff doubles


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_get_does_not_retry_unrecoverable_status(sleeps, status):
    s = FakeSession([_response(status)])
    with pytest.raises(requests.exceptions.HTTPError):
        http_get(s, "http://x")
    assert s.calls == 1 and sleeps == []


def test_get_gives_up_after_configured_attempts(sleeps):
    s = FakeSession([_response(503)] * 5)
    with pytest.raises(requests.exceptions.HTTPError):
        http_get(s, "http://x")
    assert s.calls == 5
    assert sleeps == [5, 10, 20, 40]


def test_backoff_is_capped(sleeps):
    s = FakeSession([_response(503)] * 8 + [_response(200)])
    http_get(s, "http://x", retries=9, delay=30)
    assert sleeps == [30, 60, 60, 60, 60, 60, 60, 60]


def test_retry_after_header_is_honoured_and_capped(sleeps):
    s = FakeSession([
        _response(429, {"Retry-After": "7"}),
        _response(503, {"Retry-After": "9999"}),
        _response(200),
    ])
    http_get(s, "http://x")
    assert sleeps == [7, 120]


def test_get_retries_read_timeout(sleeps):
    s = FakeSession([requests.exceptions.ReadTimeout("slow"), _response(200)])
    assert http_get(s, "http://x").status_code == 200


@pytest.mark.parametrize("status", [404, 429, 502, 503])
def test_post_retries_when_server_did_not_process_request(sleeps, status):
    s = FakeSession([_response(status), _response(200)])
    assert http_post(s, "http://x").status_code == 200
    assert s.calls == 2


def test_post_retries_connection_errors(sleeps):
    s = FakeSession([requests.exceptions.ConnectTimeout("no route"), _response(200)])
    assert http_post(s, "http://x").status_code == 200


@pytest.mark.parametrize("outcome", [
    requests.exceptions.ReadTimeout("write may have happened"),
    _response(500),
    _response(504),
    _response(400),
])
def test_post_does_not_retry_when_write_may_have_happened(sleeps, outcome):
    s = FakeSession([outcome, _response(200)])
    with pytest.raises(requests.exceptions.RequestException):
        http_post(s, "http://x")
    assert s.calls == 1 and sleeps == []


def test_non_requests_errors_are_not_retried(sleeps):
    s = FakeSession([ValueError("bug"), _response(200)])
    with pytest.raises(ValueError):
        http_get(s, "http://x")
    assert s.calls == 1
