import datetime
import json

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.harvest.zbmath import (
    fetch_zbmath_records,
    parse_documents_page,
    _year_range_filter,
)

DOC_WITH_ARXIV = {
    "document_id": 7266523,
    "title": "An Algorithm for Quantum Circuits",
    "abstract": "We describe a new algorithm.",
    "author_biographies": [
        {"name": "Doe, Jane", "zbmath_author_id": "doe.jane.1"},
        {"name": "Smith, John", "zbmath_author_id": "smith.john.1"},
    ],
    "classifications": ["68W25", "81P68"],
    "doi": "10.1234/example",
    "year": 2026,
    "links": {"arxiv": "2606.01234"},
}

DOC_WITHOUT_ARXIV = {
    "document_id": 9999999,
    "title": "A Journal Article",
    "abstract": "Pure math result.",
    "author_biographies": [{"name": "Euler, Leonhard"}],
    "classifications": ["11A41"],
    "doi": "10.9999/prime",
    "year": 2026,
    "links": {},
}

DOC_NO_ABSTRACT = {
    "document_id": 1111111,
    "title": "Short Note",
    "author_biographies": [],
    "classifications": [],
    "year": 2025,
    "links": {},
}


def _response(docs, total=None):
    return json.dumps({
        "result": {
            "total": total if total is not None else len(docs),
            "results": docs,
        }
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
        self.calls.append({"url": url, "params": params or {}})
        return FakeResp(self._pages.pop(0) if self._pages else _response([]))


def test_year_range_filter_same_year():
    today = datetime.date(2026, 6, 22)
    assert _year_range_filter(10, today) == "py:2026"

def test_year_range_filter_spans_year():
    today = datetime.date(2026, 1, 5)
    # cutoff = 2025-12-26 → cutoff year 2025, current year 2026
    assert _year_range_filter(10, today) == "py:2025-2026"


def test_parse_documents_page_with_arxiv():
    records = parse_documents_page([DOC_WITH_ARXIV])
    assert len(records) == 1
    r = records[0]
    assert r.arxiv_id == "2606.01234"
    assert r.zbmath_id == "7266523"
    assert r.title == "An Algorithm for Quantum Circuits"
    assert r.abstract == "We describe a new algorithm."
    assert r.authors == ["Doe, Jane", "Smith, John"]
    assert r.categories == ["68W25", "81P68"]
    assert r.doi == "10.1234/example"
    assert r.published == "2026-01-01"
    assert r.record_id == "2606.01234"   # arxiv_id takes precedence


def test_parse_documents_page_without_arxiv():
    records = parse_documents_page([DOC_WITHOUT_ARXIV])
    r = records[0]
    assert r.arxiv_id == ""
    assert r.zbmath_id == "9999999"
    assert r.record_id == "zbmath:9999999"


def test_parse_documents_page_missing_fields():
    records = parse_documents_page([DOC_NO_ABSTRACT])
    r = records[0]
    assert r.abstract == ""
    assert r.authors == []
    assert r.doi is None
    assert r.zbmath_id == "1111111"


def test_fetch_appends_year_filter_to_query():
    session = FakeSession([_response([])])
    list(fetch_zbmath_records(
        "cc:68W25",
        since_days=10,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    url = session.calls[0]["url"]
    assert "cc:68W25" in url
    assert "py:2026" in url


def test_fetch_does_not_duplicate_py_filter():
    session = FakeSession([_response([])])
    list(fetch_zbmath_records(
        "cc:68W25 py:2025-2026",
        since_days=10,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    url = session.calls[0]["url"]
    assert url.count("py:") == 1


def test_fetch_paginates():
    page1 = _response([DOC_WITH_ARXIV], total=2)
    page2 = _response([DOC_WITHOUT_ARXIV], total=2)
    session = FakeSession([page1, page2])
    records = list(fetch_zbmath_records(
        "cc:68W25",
        since_days=365,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    assert len(records) == 2
    assert session.calls[0]["params"]["start"] == 0
    assert session.calls[1]["params"]["start"] == 1


def test_fetch_stops_when_no_results():
    session = FakeSession([_response([], total=0)])
    records = list(fetch_zbmath_records(
        "cc:11A41",
        since_days=10,
        session=session,
        sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    assert records == []
    assert len(session.calls) == 1
