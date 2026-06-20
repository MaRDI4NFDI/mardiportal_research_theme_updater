from topic_overviews.kg.model import qid_from_uri, P_MAIN_SUBJECT, Q_SCHOLARLY_ARTICLE
from topic_overviews.kg.sparql import run_sparql


def test_constants():
    assert P_MAIN_SUBJECT == "P30"
    assert Q_SCHOLARLY_ARTICLE == "Q56887"


def test_qid_from_uri():
    assert qid_from_uri("https://portal.mardi4nfdi.de/entity/Q42") == "Q42"
    assert qid_from_uri("Q7") == "Q7"


def test_run_sparql_flattens_bindings():
    payload = {
        "results": {"bindings": [
            {"topic": {"value": "https://x/entity/Q1"}, "label": {"value": "Optimization"}},
        ]}
    }

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return payload

    class FakeSession:
        def get(self, url, params=None, headers=None, timeout=None):
            return FakeResp()

    rows = run_sparql("http://endpoint", "SELECT ...", session=FakeSession())
    assert rows == [{"topic": "https://x/entity/Q1", "label": "Optimization"}]
