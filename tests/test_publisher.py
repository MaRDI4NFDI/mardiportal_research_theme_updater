from topic_overviews.wiki.publisher import WikiPublisher


class FakeResp:
    def __init__(self, payload): self._payload = payload
    def raise_for_status(self): pass
    def json(self): return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", params))
        return FakeResp(self._responses.pop(0))

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", data))
        return FakeResp(self._responses.pop(0))


def test_login_then_edit_posts_text_with_token():
    responses = [
        {"query": {"tokens": {"logintoken": "LT"}}},          # GET login token
        {"login": {"result": "Success"}},                      # POST login
        {"query": {"tokens": {"csrftoken": "CT"}}},            # GET csrf token
        {"edit": {"result": "Success"}},                       # POST edit
    ]
    session = FakeSession(responses)
    pub = WikiPublisher("http://api", "bot", "pw", session=session)
    pub.login()
    pub.edit("Topic:Online Algorithms", "= hi =", "update")

    post_calls = [c for c in session.calls if c[0] == "POST"]
    login_data = post_calls[0][1]
    edit_data = post_calls[1][1]
    assert login_data["lgtoken"] == "LT"
    assert edit_data["title"] == "Topic:Online Algorithms"
    assert edit_data["text"] == "= hi ="
    assert edit_data["token"] == "CT"
    assert edit_data["summary"] == "update"


def test_page_exists_true_and_false():
    present = FakeSession([
        {"query": {"pages": {"42": {"pageid": 42, "title": "My Theme"}}}},  # exists
    ])
    assert WikiPublisher("http://api", "bot", "pw", session=present).page_exists("My Theme") is True

    missing = FakeSession([
        {"query": {"pages": {"-1": {"missing": ""}}}},                      # absent
    ])
    assert WikiPublisher("http://api", "bot", "pw", session=missing).page_exists("Nope") is False
