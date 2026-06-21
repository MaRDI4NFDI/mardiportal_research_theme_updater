import datetime
import json

import pytest

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.harvest.openalex import (
    fetch_openalex_records,
    parse_works_page,
    _reconstruct_abstract,
)

WORK_WITH_ARXIV = {
    "id": "https://openalex.org/W1111111111",
    "doi": "https://doi.org/10.1000/xyz",
    "title": "A Paper on MaRDI",
    "abstract_inverted_index": {"We": [0], "study": [1], "MaRDI": [2]},
    "authorships": [
        {"author": {"display_name": "Jane Doe"}},
        {"author": {"display_name": "John Smith"}},
    ],
    "topics": [{"display_name": "Mathematical Research Data"}],
    "publication_date": "2026-06-15",
    "ids": {"arxiv": "https://arxiv.org/abs/2606.01234", "doi": "https://doi.org/10.1000/xyz"},
}

WORK_WITHOUT_ARXIV = {
    "id": "https://openalex.org/W2222222222",
    "doi": "https://doi.org/10.9999/abc",
    "title": "A Journal-Only Paper",
    "abstract_inverted_index": {"Journal": [0], "paper": [1]},
    "authorships": [{"author": {"display_name": "Alice"}}],
    "topics": [],
    "publication_date": "2026-06-10",
    "ids": {"doi": "https://doi.org/10.9999/abc"},
}

WORK_OLD = {
    "id": "https://openalex.org/W3333333333",
    "doi": None,
    "title": "Old Paper",
    "abstract_inverted_index": {},
    "authorships": [],
    "topics": [],
    "publication_date": "2025-01-01",
    "ids": {},
}


def _page(works, next_cursor=None):
    return json.dumps({
        "meta": {"count": len(works), "per_page": 200, "next_cursor": next_cursor},
        "results": works,
    })


class FakeResp:
    def __init__(self, body):
        self._body = body
    def raise_for_status(self): pass
    def json(self): return json.loads(self._body)


class FakeSession:
    def __init__(self, pages):
        self._pages = list(pages)
        self.calls = []
    def get(self, url, params=None, timeout=None):
        self.calls.append(params or {})
        return FakeResp(self._pages.pop(0) if self._pages else _page([]))


def test_reconstruct_abstract():
    inv = {"We": [0], "study": [1], "MaRDI": [2]}
    assert _reconstruct_abstract(inv) == "We study MaRDI"

def test_reconstruct_abstract_empty():
    assert _reconstruct_abstract(None) == ""
    assert _reconstruct_abstract({}) == ""

def test_parse_works_page_with_arxiv_id():
    records = parse_works_page([WORK_WITH_ARXIV])
    assert len(records) == 1
    r = records[0]
    assert r.arxiv_id == "2606.01234"
    assert r.openalex_id == "W1111111111"
    assert r.title == "A Paper on MaRDI"
    assert r.abstract == "We study MaRDI"
    assert r.authors == ["Jane Doe", "John Smith"]
    assert r.published == "2026-06-15"
    assert r.doi == "10.1000/xyz"
    assert r.record_id == "2606.01234"   # arxiv_id takes precedence

def test_parse_works_page_without_arxiv_id():
    records = parse_works_page([WORK_WITHOUT_ARXIV])
    r = records[0]
    assert r.arxiv_id == ""
    assert r.openalex_id == "W2222222222"
    assert r.record_id == "openalex:W2222222222"
    assert r.doi == "10.9999/abc"

def test_fetch_stops_at_date_boundary():
    session = FakeSession([_page([WORK_WITH_ARXIV, WORK_OLD])])
    records = list(fetch_openalex_records(
        "search=mardi",
        since_days=10,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 21),
    ))
    # cutoff = 2026-06-11; WORK_WITH_ARXIV (06-15) kept, WORK_OLD (2025-01-01) stops iteration
    assert len(records) == 1
    assert records[0].arxiv_id == "2606.01234"

def test_fetch_adds_date_filter_to_api_call():
    session = FakeSession([_page([])])
    list(fetch_openalex_records(
        "search=mardi&filter=funders.id:f4320320879",
        since_days=10,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 21),
    ))
    params = session.calls[0]
    assert params["search"] == "mardi"
    assert "funders.id:f4320320879" in params["filter"]
    assert "from_publication_date:2026-06-11" in params["filter"]

def test_fetch_default_sort_is_publication_date_desc():
    session = FakeSession([_page([])])
    list(fetch_openalex_records(
        "search=mardi",
        since_days=10,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 21),
    ))
    assert session.calls[0]["sort"] == "publication_date:desc"

def test_fetch_follows_cursor_pagination():
    page1 = _page([WORK_WITH_ARXIV], next_cursor="CURSOR_ABC")
    page2 = _page([WORK_WITHOUT_ARXIV])
    session = FakeSession([page1, page2])
    records = list(fetch_openalex_records(
        "search=mardi",
        since_days=30,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 21),
    ))
    assert len(records) == 2
    assert session.calls[0].get("cursor") == "*"
    assert session.calls[1].get("cursor") == "CURSOR_ABC"

def test_fetch_sends_mailto_when_email_set():
    session = FakeSession([_page([])])
    list(fetch_openalex_records(
        "search=mardi",
        since_days=10,
        email="bot@example.com",
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 21),
    ))
    assert session.calls[0].get("mailto") == "bot@example.com"
