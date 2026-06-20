from types import SimpleNamespace

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.topics import Topic
from topic_overviews.llm.topic_classifier import classify_paper

PAPER = PaperRecord(
    arxiv_id="2401.00001", title="Online Caching with Predictions",
    abstract="We study caching.", authors=["Jane Doe"],
    categories=["cs.DS"], published="2024-01-02", doi=None,
)
TOPICS = [
    Topic(qid="Q10", label="Optimization", description="..."),
    Topic(qid="Q11", label="Online Algorithms", description="..."),
]


class FakeClient:
    def __init__(self, texts):
        self._texts = list(texts)
        self.prompts = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, messages):
        self.prompts.append(messages[0]["content"])
        text = self._texts.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


def test_returns_known_qids_and_drops_unknown():
    client = FakeClient(['{"topics": ["Q11", "Q999"]}'])
    result = classify_paper(PAPER, TOPICS, model="claude-haiku-4-5", api_key="x", client=client)
    assert result == ["Q11"]
    # prompt includes the topics and the paper text
    assert "Q11" in client.prompts[0] and "Online Caching with Predictions" in client.prompts[0]


def test_empty_match_returns_empty_list():
    client = FakeClient(['{"topics": []}'])
    assert classify_paper(PAPER, TOPICS, model="m", api_key="x", client=client) == []


def test_retries_once_on_bad_json_then_succeeds():
    client = FakeClient(["not json", '{"topics": ["Q10"]}'])
    assert classify_paper(PAPER, TOPICS, model="m", api_key="x", client=client) == ["Q10"]


def test_gives_up_after_retry():
    client = FakeClient(["nope", "still nope"])
    assert classify_paper(PAPER, TOPICS, model="m", api_key="x", client=client) == []
