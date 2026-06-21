from types import SimpleNamespace

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.llm.keyworder import keywords_paper

PAPER = PaperRecord("2401.00001", "A Title", "An abstract.", ["Jane Doe"],
                    ["math.NA"], "2024-01-02", None)


def _client(text):
    return SimpleNamespace(
        messages=SimpleNamespace(
            create=lambda **k: SimpleNamespace(content=[SimpleNamespace(text=text)])
        )
    )


def test_keywords_parses_json_array():
    out = keywords_paper(PAPER, model="m", api_key="x",
                         client=_client('Here: ["Solvers", "Multigrid", "Poisson"]'))
    assert out == ["Solvers", "Multigrid", "Poisson"]


def test_keywords_empty_on_bad_output():
    out = keywords_paper(PAPER, model="m", api_key="x", client=_client("no json here"))
    assert out == []
