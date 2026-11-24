"""Batch lookup of already-imported papers (KGClient.prefetch_known, pipeline._prefetched)."""
import pytest

import topic_overviews.kg.client as client_mod
from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.client import KGClient
from topic_overviews.pipeline import _prefetched


def _arxiv(aid):
    return PaperRecord(
        arxiv_id=aid, title=f"Paper {aid}", abstract="", authors=[],
        categories=[], published="2026-11-01", doi="",
    )


def _published(doi="10.1000/xyz", openalex_id="W1"):
    return PaperRecord(
        arxiv_id="", title="Published", abstract="", authors=[],
        categories=[], published="2026-11-01", doi=doi, openalex_id=openalex_id,
    )


class Resp:
    def __init__(self, bindings):
        self._b = bindings

    def json(self):
        return {"results": {"bindings": self._b}}


def row(value, qid, tldr=False):
    r = {
        "id": {"value": value},
        "item": {"value": f"https://portal.mardi4nfdi.de/entity/{qid}"},
    }
    if tldr:
        r["tldr"] = {"value": "summary"}
    return r


class FailingMC:
    """Item loads must not happen for papers answered from the batch."""

    class item:
        @staticmethod
        def get(entity_id):
            raise AssertionError("item load for a paper known from the batch")


def stub_sparql(monkeypatch, rows_by_call):
    queries = []

    def fake_get(session, url, **kwargs):
        queries.append(kwargs["params"]["query"])
        return Resp(rows_by_call(len(queries) - 1, kwargs["params"]["query"]))

    monkeypatch.setattr(client_mod, "http_get", fake_get)
    return queries


def test_batch_answers_later_lookups_without_any_request(monkeypatch):
    kg = KGClient(FailingMC())
    queries = stub_sparql(monkeypatch, lambda i, q: [row("2611.1", "Q10", tldr=True)])
    kg.prefetch_known([_arxiv("2611.1")])

    def no_request(*a, **k):
        raise AssertionError("request after the batch lookup")

    monkeypatch.setattr(client_mod, "http_get", no_request)
    assert kg.find_existing_paper(_arxiv("2611.1")) == "Q10"
    assert kg.paper_has_tldr("Q10") is True
    assert len(queries) >= 1


def test_one_query_for_a_whole_batch(monkeypatch):
    kg = KGClient(FailingMC())
    queries = stub_sparql(monkeypatch, lambda i, q: [])
    kg.prefetch_known([_arxiv(f"2611.{n}") for n in range(50)])
    p21 = [q for q in queries if "wdt:P21 ?id" in q]
    assert len(p21) == 1
    assert '"2611.0"' in p21[0] and '"2611.49"' in p21[0]


def test_batches_are_split_at_batch_size(monkeypatch):
    kg = KGClient(FailingMC())
    queries = stub_sparql(monkeypatch, lambda i, q: [])
    kg.prefetch_known([_arxiv(f"2611.{n}") for n in range(25)], batch_size=10)
    assert len([q for q in queries if "wdt:P21 ?id" in q]) == 3


def test_arxiv_query_has_same_filters_as_the_single_lookup(monkeypatch):
    kg = KGClient(FailingMC())
    queries = stub_sparql(monkeypatch, lambda i, q: [])
    kg.prefetch_known([_arxiv("2611.1")])
    own = next(q for q in queries if "wdt:P21 ?id" in q)
    assert "NOT EXISTS" not in own and "wdt:P1963 ?tldr" in own
    doi = next(q for q in queries if "wdt:P27 ?id" in q)
    assert '"10.48550/ARXIV.2611.1"' in doi
    assert "wdt:P225 []" in doi and "wdt:P1451 []" in doi


def test_published_query_excludes_arxiv_items(monkeypatch):
    kg = KGClient(FailingMC())
    queries = stub_sparql(monkeypatch, lambda i, q: [])
    kg.prefetch_known([_published()])
    shared = [q for q in queries if "wdt:P27 ?id" in q or "wdt:P388 ?id" in q]
    assert len(shared) == 2
    assert all("wdt:P21 []" in q for q in shared)


def test_a_miss_is_still_looked_up_live(monkeypatch):
    kg = KGClient(FailingMC())
    stub_sparql(monkeypatch, lambda i, q: [])  # nothing found in the batch
    kg.prefetch_known([_arxiv("2611.2")])
    live = []
    monkeypatch.setattr(
        kg, "_sparql_find_by_value",
        lambda prop, value, exclude_props=(): live.append((prop, value)) or None,
    )
    assert kg.find_existing_paper(_arxiv("2611.2")) is None
    assert ("P21", "2611.2") in live


def test_item_without_tldr_is_checked_on_the_item(monkeypatch):
    kg = KGClient(FailingMC())
    stub_sparql(monkeypatch, lambda i, q: [row("2611.3", "Q11", tldr=False)])
    kg.prefetch_known([_arxiv("2611.3")])
    assert kg.find_existing_paper(_arxiv("2611.3")) == "Q11"
    with pytest.raises(AssertionError):
        kg.paper_has_tldr("Q11")  # falls through to the live item load


def test_failed_batch_changes_nothing(monkeypatch):
    kg = KGClient(FailingMC())

    def boom(*a, **k):
        raise RuntimeError("qlever down")

    monkeypatch.setattr(client_mod, "http_get", boom)
    assert kg.prefetch_known([_arxiv("2611.4")]) == 0
    assert kg._lookup_cache == {} and kg._tldr_known == set()


def test_identifier_with_quote_is_not_sent(monkeypatch):
    kg = KGClient(FailingMC())
    queries = stub_sparql(monkeypatch, lambda i, q: [])
    kg.prefetch_known([_arxiv('2611.5"x'), _arxiv("2611.6")])
    sent = " ".join(queries)
    assert '2611.5' not in sent and '"2611.6"' in sent


def test_known_identifiers_are_not_asked_again(monkeypatch):
    kg = KGClient(FailingMC())
    queries = stub_sparql(monkeypatch, lambda i, q: [row("2611.7", "Q12")])
    kg.prefetch_known([_arxiv("2611.7")])
    before = len(queries)
    kg.prefetch_known([_arxiv("2611.7")])
    assert not any('"2611.7"' in q and "wdt:P21 ?id" in q for q in queries[before:])


# --- pipeline._prefetched -------------------------------------------------

class Recorder:
    def __init__(self):
        self.batches = []

    def prefetch_known(self, batch):
        self.batches.append([r for r in batch])


def test_prefetched_yields_everything_in_batches():
    kg = Recorder()
    out = list(_prefetched(iter(range(25)), kg, size=10))
    assert out == list(range(25))
    assert [len(b) for b in kg.batches] == [10, 10, 5]


def test_prefetched_exact_multiple_of_size():
    kg = Recorder()
    assert list(_prefetched(iter(range(20)), kg, size=10)) == list(range(20))
    assert [len(b) for b in kg.batches] == [10, 10]


def test_prefetched_works_without_a_client():
    assert list(_prefetched(iter(range(3)), None)) == [0, 1, 2]
    assert list(_prefetched(iter(range(3)), object())) == [0, 1, 2]


def test_prefetched_is_lazy_beyond_one_batch():
    pulled = []

    def source():
        for n in range(1000):
            pulled.append(n)
            yield n

    gen = _prefetched(source(), Recorder(), size=10)
    assert next(gen) == 0
    assert len(pulled) == 10  # only the first batch was read ahead


def test_prefetched_yields_earlier_records_before_a_source_error():
    def source():
        yield from range(5)
        raise RuntimeError("page 2 failed")

    seen = []
    with pytest.raises(RuntimeError, match="page 2 failed"):
        for r in _prefetched(source(), Recorder(), size=10):
            seen.append(r)
    assert seen == [0, 1, 2, 3, 4]


def test_prefetch_failure_does_not_stop_the_harvest():
    class Broken:
        def prefetch_known(self, batch):
            raise RuntimeError("boom")

    assert list(_prefetched(iter(range(4)), Broken(), size=2)) == [0, 1, 2, 3]
