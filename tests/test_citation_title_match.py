"""Title-based citation matching: only an unambiguous publication counts."""
from unittest.mock import MagicMock

from topic_overviews.kg.citation_linker import _search_kg_by_title, resolve_s2_references

TITLE = "A Priori Error Estimates for Finite Element Methods"
API = "https://example.org/w/api.php"
PUBLICATION = {"claims": {"P1460": [{"mainsnak": {"datavalue": {"value": {"id": "Q5976449"}}}}]}}
OTHER = {"claims": {"P1460": [{"mainsnak": {"datavalue": {"value": {"id": "Q5984635"}}}}]}}


def _session(hits, entities):
    def get(url, params=None, **kwargs):
        resp = MagicMock()
        if params["action"] == "wbsearchentities":
            resp.json.return_value = {"search": hits}
        else:
            resp.json.return_value = {"entities": entities}
        return resp
    session = MagicMock()
    session.get.side_effect = get
    return session


def _hit(qid, label=TITLE):
    return {"id": qid, "label": label}


def test_unique_publication_with_exact_title_matches():
    s = _session([_hit("Q1")], {"Q1": PUBLICATION})
    assert _search_kg_by_title(TITLE, API, s) == "Q1"


def test_label_comparison_ignores_case():
    s = _session([_hit("Q1", TITLE.upper())], {"Q1": PUBLICATION})
    assert _search_kg_by_title(TITLE, API, s) == "Q1"


def test_item_that_is_not_a_publication_is_rejected():
    s = _session([_hit("Q1")], {"Q1": OTHER})
    assert _search_kg_by_title(TITLE, API, s) == ""


def test_item_without_profile_is_rejected():
    s = _session([_hit("Q1")], {"Q1": {"claims": {}}})
    assert _search_kg_by_title(TITLE, API, s) == ""


def test_two_publications_with_the_same_title_are_ambiguous():
    s = _session([_hit("Q1"), _hit("Q2")], {"Q1": PUBLICATION, "Q2": PUBLICATION})
    assert _search_kg_by_title(TITLE, API, s) == ""


def test_non_publication_namesake_does_not_make_it_ambiguous():
    s = _session([_hit("Q1"), _hit("Q2")], {"Q1": PUBLICATION, "Q2": OTHER})
    assert _search_kg_by_title(TITLE, API, s) == "Q1"


def test_prefix_hit_with_different_label_is_ignored():
    s = _session([_hit("Q1", TITLE + " II")], {"Q1": PUBLICATION})
    assert _search_kg_by_title(TITLE, API, s) == ""


def test_short_generic_title_is_not_searched():
    s = _session([_hit("Q1", "Introduction")], {"Q1": PUBLICATION})
    assert _search_kg_by_title("Introduction", API, s) == ""
    s.get.assert_not_called()


def test_no_exact_hit_skips_the_claims_request():
    s = _session([_hit("Q1", "Something else entirely different")], {})
    assert _search_kg_by_title(TITLE, API, s) == ""
    assert s.get.call_count == 1


def test_api_failure_returns_empty():
    s = MagicMock()
    s.get.side_effect = Exception("503")
    assert _search_kg_by_title(TITLE, API, s) == ""


def test_resolve_s2_references_uses_the_checked_title_match(monkeypatch):
    import topic_overviews.kg.citation_linker as cl
    monkeypatch.setattr(cl, "_batch_sparql_lookup", lambda *a, **k: {})
    s = _session([_hit("Q1")], {"Q1": OTHER})
    refs = [{"arxiv_id": "", "doi": "", "openalex_id": "", "title": TITLE}]
    assert resolve_s2_references(refs, "http://sparql", API, session=s) == []
    s = _session([_hit("Q1")], {"Q1": PUBLICATION})
    assert resolve_s2_references(refs, "http://sparql", API, session=s) == ["Q1"]
