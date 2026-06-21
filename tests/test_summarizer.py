from types import SimpleNamespace

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.llm.summarizer import summarize_paper

PAPER = PaperRecord("2401.00001", "A Title", "An abstract about solvers.",
                    ["Jane Doe"], ["math.NA"], "2024-01-02", None)


class FakeClient:
    def __init__(self, text):
        self._text = text
        self.prompts = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, messages):
        self.prompts.append(messages[0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(text=self._text)])


def test_summarize_returns_stripped_text_and_includes_paper():
    client = FakeClient("  A fast solver for Poisson problems.\n")
    out = summarize_paper(PAPER, model="m", api_key="x", client=client)
    assert out == "A fast solver for Poisson problems."
    assert "An abstract about solvers." in client.prompts[0]


def test_summarize_returns_empty_on_error():
    class Boom:
        messages = SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert summarize_paper(PAPER, model="m", api_key="x", client=Boom()) == ""
