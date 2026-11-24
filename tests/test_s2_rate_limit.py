from unittest.mock import MagicMock

import topic_overviews.kg.citation_linker as cl


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def _resp(status, data=None, retry_after=None):
    r = MagicMock()
    r.status_code = status
    r.headers = {"Retry-After": retry_after} if retry_after else {}
    r.json.return_value = {"data": data or []}
    r.raise_for_status.side_effect = None if status < 400 else Exception(f"HTTP {status}")
    return r


def _call(session, clock):
    return cl.fetch_s2_references(arxiv_id="1", session=session, sleep=clock.sleep, clock=clock)


def setup_function():
    cl._s2_last_request = 0.0


def test_consecutive_requests_are_at_least_a_second_apart():
    clock = Clock()
    session = MagicMock()
    session.get.return_value = _resp(200)
    _call(session, clock)
    before = clock.now
    _call(session, clock)
    assert clock.now - before >= 1.0


def test_429_is_retried_and_honors_retry_after():
    clock = Clock()
    session = MagicMock()
    session.get.side_effect = [_resp(429, retry_after="12"), _resp(200, [
        {"citedPaper": {"externalIds": {"DOI": "10.1/x"}, "title": "T"}}])]
    refs = _call(session, clock)
    assert [r["doi"] for r in refs] == ["10.1/x"]
    assert 12 in clock.sleeps
    assert session.get.call_count == 2


def test_gives_up_after_three_429s():
    clock = Clock()
    session = MagicMock()
    session.get.return_value = _resp(429)
    assert _call(session, clock) == []
    assert session.get.call_count == 3
