import datetime
import json

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.harvest.zbmath import (
    fetch_zbmath_records,
    lookup_by_arxiv_id,
    parse_documents_page,
    _year_range_filter,
)

# ---------------------------------------------------------------------------
# Fixtures — new API response format (api.zbmath.org/v1/document/_search)
# ---------------------------------------------------------------------------

DOC_WITH_ARXIV = {
    "id": 7266523,
    "identifier": "2606.12345",
    "title": {"title": "An Algorithm for Quantum Circuits"},
    "contributors": {
        "authors": [
            {"name": "Doe, Jane", "codes": ["doe.jane.1"], "aliases": [], "checked": "0"},
            {"name": "Smith, John", "codes": ["smith.john.1"], "aliases": [], "checked": "0"},
        ],
        "editors": [],
    },
    "msc": [{"code": "68W25", "scheme": "msc2020"}, {"code": "81P68", "scheme": "msc2020"}],
    "links": [
        {"identifier": "10.1234/example", "type": "doi", "url": "https://doi.org/10.1234/example"},
        {"identifier": "2606.01234", "type": "arxiv", "url": "https://arxiv.org/abs/2606.01234"},
    ],
    "year": 2026,
}

DOC_WITHOUT_ARXIV = {
    "id": 9999999,
    "identifier": "2606.99999",
    "title": {"title": "A Journal Article"},
    "contributors": {
        "authors": [
            {"name": "Euler, Leonhard", "codes": ["euler.leonhard"], "aliases": [], "checked": "1"},
        ],
        "editors": [],
    },
    "msc": [{"code": "11A41", "scheme": "msc2020"}],
    "links": [
        {"identifier": "10.9999/prime", "type": "doi", "url": "https://doi.org/10.9999/prime"},
    ],
    "year": 2026,
}

DOC_NO_AUTHORS = {
    "id": 1111111,
    "identifier": "2606.11111",
    "title": {"title": "Short Note"},
    "contributors": {"authors": [], "editors": []},
    "msc": [],
    "links": [],
    "year": 2025,
}

DOC_AUTHOR_NO_CODE = {
    "id": 2222222,
    "identifier": "2606.22222",
    "title": {"title": "Codeless Author"},
    "contributors": {
        "authors": [
            {"name": "Anonymous", "codes": [], "aliases": [], "checked": "0"},
        ],
        "editors": [],
    },
    "msc": [],
    "links": [],
    "year": 2025,
}


def _response(docs, total=None):
    return json.dumps({
        "result": docs,
        "status": {
            "nr_total_results": total if total is not None else len(docs),
            "nr_request_results": len(docs),
            "execution_bool": True,
        },
    })


def _empty_response():
    return json.dumps({
        "result": None,
        "status": {
            "nr_total_results": None,
            "nr_request_results": None,
            "execution_bool": False,
            "internal_code": "successful access. Zero Results for this query.",
        },
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
        return FakeResp(self._pages.pop(0) if self._pages else _empty_response())


# ---------------------------------------------------------------------------
# Year-range filter
# ---------------------------------------------------------------------------

def test_year_range_filter_same_year():
    today = datetime.date(2026, 6, 22)
    assert _year_range_filter(10, today) == "py:2026"

def test_year_range_filter_spans_year():
    today = datetime.date(2026, 1, 5)
    assert _year_range_filter(10, today) == "py:2025-2026"


# ---------------------------------------------------------------------------
# parse_documents_page
# ---------------------------------------------------------------------------

def test_parse_with_arxiv():
    records = parse_documents_page([DOC_WITH_ARXIV])
    assert len(records) == 1
    r = records[0]
    assert r.arxiv_id == "2606.01234"
    assert r.zbmath_id == "7266523"
    assert r.title == "An Algorithm for Quantum Circuits"
    assert r.abstract == ""
    assert r.authors == ["Doe, Jane", "Smith, John"]
    assert r.zbmath_author_ids == [("Doe, Jane", "doe.jane.1"), ("Smith, John", "smith.john.1")]
    assert r.categories == ["68W25", "81P68"]
    assert r.doi == "10.1234/example"
    assert r.published == "2026-01-01"
    assert r.record_id == "2606.01234"   # arxiv_id takes precedence


def test_parse_without_arxiv():
    records = parse_documents_page([DOC_WITHOUT_ARXIV])
    r = records[0]
    assert r.arxiv_id == ""
    assert r.zbmath_id == "9999999"
    assert r.doi == "10.9999/prime"
    assert r.zbmath_author_ids == [("Euler, Leonhard", "euler.leonhard")]
    assert r.record_id == "zbmath:9999999"


def test_parse_no_authors():
    records = parse_documents_page([DOC_NO_AUTHORS])
    r = records[0]
    assert r.authors == []
    assert r.zbmath_author_ids == []
    assert r.doi is None
    assert r.zbmath_id == "1111111"


def test_parse_author_without_code():
    records = parse_documents_page([DOC_AUTHOR_NO_CODE])
    r = records[0]
    assert r.authors == ["Anonymous"]
    assert r.zbmath_author_ids == []   # code list was empty — not included


# ---------------------------------------------------------------------------
# fetch_zbmath_records
# ---------------------------------------------------------------------------

def test_fetch_appends_year_filter():
    session = FakeSession([_empty_response()])
    list(fetch_zbmath_records(
        "cc:68W25", since_days=10, session=session, sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    params = session.calls[0]["params"]
    assert "cc:68W25" in params["search_string"]
    assert "py:2026" in params["search_string"]


def test_fetch_does_not_duplicate_py_filter():
    session = FakeSession([_empty_response()])
    list(fetch_zbmath_records(
        "cc:68W25 py:2025-2026", since_days=10, session=session, sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    assert session.calls[0]["params"]["search_string"].count("py:") == 1


def test_fetch_uses_search_endpoint():
    session = FakeSession([_empty_response()])
    list(fetch_zbmath_records(
        "cc:68W25", since_days=10, session=session, sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    assert session.calls[0]["url"].endswith("/document/_search")


def test_fetch_paginates():
    page1 = _response([DOC_WITH_ARXIV], total=2)
    page2 = _response([DOC_WITHOUT_ARXIV], total=2)
    session = FakeSession([page1, page2])
    records = list(fetch_zbmath_records(
        "cc:68W25", since_days=365, session=session, sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    assert len(records) == 2
    assert session.calls[0]["params"]["page"] == 0
    assert session.calls[1]["params"]["page"] == 1


def test_fetch_stops_when_no_results():
    session = FakeSession([_empty_response()])
    records = list(fetch_zbmath_records(
        "cc:11A41", since_days=10, session=session, sleep=lambda s: None,
        today=datetime.date(2026, 6, 22),
    ))
    assert records == []
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# lookup_by_arxiv_id
# ---------------------------------------------------------------------------

def test_lookup_found():
    session = FakeSession([_response([DOC_WITH_ARXIV])])
    result = lookup_by_arxiv_id("2606.01234", session=session)
    assert result is not None
    assert result.zbmath_id == "7266523"
    assert result.arxiv_id == "2606.01234"
    assert result.zbmath_author_ids == [("Doe, Jane", "doe.jane.1"), ("Smith, John", "smith.john.1")]
    params = session.calls[0]["params"]
    assert params["search_string"] == "arxiv:2606.01234"
    assert params["results_per_page"] == 3


def test_lookup_not_found():
    session = FakeSession([_empty_response()])
    result = lookup_by_arxiv_id("9999.99999", session=session)
    assert result is None


def test_lookup_mismatch_filtered_out():
    # zbMATH returns a record but its arXiv ID does not match the query
    session = FakeSession([_response([DOC_WITH_ARXIV])])
    result = lookup_by_arxiv_id("0000.00000", session=session)
    assert result is None


def test_lookup_empty_arxiv_id():
    session = FakeSession([])
    result = lookup_by_arxiv_id("", session=session)
    assert result is None
    assert session.calls == []
