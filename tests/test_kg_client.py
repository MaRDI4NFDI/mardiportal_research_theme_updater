from unittest.mock import MagicMock

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.client import KGClient, to_wbi_time

PAPER = PaperRecord(
    arxiv_id="2401.00001", title="A New Bound for Online Caching",
    abstract="...", authors=["Jane Doe", "John Smith"],
    categories=["math.OC", "cs.DS"], published="2024-01-02", doi="10.1000/xyz",
)


class _Sitelink:
    def __init__(self, title): self.title = title


class FakeSitelinks:
    def __init__(self, data=None):
        self._d = dict(data or {})
        self.set_calls = []

    def get(self, site):
        return _Sitelink(self._d[site]) if site in self._d else None

    def set(self, site=None, title=None):
        self.set_calls.append((site, title))
        self._d[site] = title


class FakeItem:
    def __init__(self, values=None, item_id="Q500", sitelinks=None):
        self.claims = []
        self.label = None
        self.id = item_id
        self._values = values or {}
        self.written = False
        self.sitelinks = FakeSitelinks(sitelinks)

    class _Labels:
        def __init__(self, outer): self.outer = outer
        def set(self, language, value): self.outer.label = (language, value)

    @property
    def labels(self): return FakeItem._Labels(self)

    def add_claim(self, prop, value=None, action="append_or_replace"):
        self.claims.append((prop, value))

    def get_value(self, prop):
        return self._values.get(prop, [])

    def write(self):
        self.written = True
        return self


class FakeItemNS:
    def __init__(self, item): self._item = item
    def new(self): return self._item
    def get(self, entity_id=None): return self._item


class FakeMC:
    def __init__(self, existing=None, item=None):
        self._existing = existing or []
        self.item = FakeItemNS(item or FakeItem())
        self.searched = []

    def search_entity_by_value(self, prop, value):
        self.searched.append((prop, value))
        return self._existing


def test_to_wbi_time():
    assert to_wbi_time("2024-01-02") == "+2024-01-02T00:00:00Z"


def test_import_new_paper_writes_only_paper_statements():
    item = FakeItem()
    mc = FakeMC(existing=[], item=item)
    kg = KGClient(mc)
    kg.find_existing_paper = lambda record: None  # bypass HTTP lookups in tests
    qid = kg.import_paper(PAPER)

    assert qid == "Q500"
    assert item.label == ("en", "A New Bound for Online Caching")
    assert ("P31", "Q56887") in item.claims          # instance of scholarly article
    assert ("P1460", "Q5976449") in item.claims      # MaRDI publication profile type
    assert ("P21", "2401.00001") in item.claims         # arXiv id
    assert ("P27", "10.1000/xyz") in item.claims        # DOI
    assert ("P159", "A New Bound for Online Caching") in item.claims
    assert ("P28", "+2024-01-02T00:00:00Z") in item.claims
    assert ("P22", "math.OC") in item.claims
    assert ("P43", "Jane Doe") in item.claims
    # The paper carries NO topic/membership statement — papers stay topic-agnostic.
    assert not any(prop == "P265" for prop, _ in item.claims)
    assert not any(prop == "P30" for prop, _ in item.claims)


def test_import_paper_sets_tldr_when_given():
    item = FakeItem()
    mc = FakeMC(existing=[], item=item)
    KGClient(mc).import_paper(PAPER, tldr="A one-sentence summary.")
    assert ("P1963", "A one-sentence summary.") in item.claims


def test_import_paper_without_tldr_sets_no_tldr_claim():
    item = FakeItem()
    mc = FakeMC(existing=[], item=item)
    KGClient(mc).import_paper(PAPER)
    assert not any(prop == "P1963" for prop, _ in item.claims)


def test_import_paper_sets_generated_by_when_given():
    item = FakeItem()
    mc = FakeMC(existing=[], item=item)
    KGClient(mc).import_paper(PAPER, generated_by="Q7266558")
    assert ("P1642", "Q7266558") in item.claims


def test_import_paper_sets_one_keyword_claim_per_keyword():
    item = FakeItem()
    mc = FakeMC(existing=[], item=item)
    KGClient(mc).import_paper(PAPER, keywords=["Caching", "Online", "Predictions"])
    assert ("P1964", "Caching") in item.claims
    assert ("P1964", "Online") in item.claims
    assert ("P1964", "Predictions") in item.claims
    assert sum(1 for p, _ in item.claims if p == "P1964") == 3


def test_import_existing_paper_reuses_item():
    item = FakeItem()
    mc = FakeMC(existing=["Q500"], item=item)
    kg = KGClient(mc)
    kg.find_existing_paper = lambda record: "Q500"  # bypass HTTP lookups in tests
    qid = kg.import_paper(PAPER)
    assert qid == "Q500"
    # existing item fetched, not newly labelled
    assert item.label is None
    assert ("P31", "Q56887") in item.claims      # claims still written on existing item


def test_get_paper_qid_returns_existing_qid():
    item = FakeItem()
    mc = FakeMC(existing=["Q500"], item=item)
    assert KGClient(mc).get_paper_qid(PAPER.arxiv_id) == "Q500"


def test_paper_has_tldr_checks_existing_paper_claims():
    item = FakeItem(values={"P1963": ["Already there"]})
    mc = FakeMC(existing=["Q500"], item=item)
    assert KGClient(mc).paper_has_tldr("Q500") is True


def test_link_topic_adds_paper_to_topic_when_absent():
    topic = FakeItem(item_id="Q11")
    mc = FakeMC(item=topic)
    KGClient(mc).link_topic("Q11", "Q500")
    assert ("P265", "Q500") in topic.claims      # has part(s) -> paper, on the TOPIC item
    assert topic.written is True


def test_link_topic_is_idempotent_when_paper_already_listed():
    topic = FakeItem(item_id="Q11", values={"P265": ["Q500", "Q777"]})
    mc = FakeMC(item=topic)
    KGClient(mc).link_topic("Q11", "Q500")
    assert topic.claims == []        # nothing added
    assert topic.written is False     # no write attempted


def test_get_theme_sitelink_returns_title_when_connected():
    theme = FakeItem(item_id="Q11", sitelinks={"mardi": "My Theme Page"})
    mc = FakeMC(item=theme)
    assert KGClient(mc).get_theme_sitelink("Q11") == "My Theme Page"


def test_get_theme_sitelink_none_when_unconnected():
    theme = FakeItem(item_id="Q11")
    mc = FakeMC(item=theme)
    assert KGClient(mc).get_theme_sitelink("Q11") is None


def test_set_theme_sitelink_sets_mardi_site_and_writes():
    theme = FakeItem(item_id="Q11")
    mc = FakeMC(item=theme)
    KGClient(mc).set_theme_sitelink("Q11", "My Theme Page")
    assert theme.sitelinks.set_calls == [("mardi", "My Theme Page")]
    assert theme.written is True


def _make_p265_entities(paper_date_pairs: list[tuple[str, str]]) -> dict:
    """Build a fake wbgetentities response for a theme with P265 links."""
    p265 = [
        {
            "id": f"Q9999${qid}",
            "mainsnak": {"datavalue": {"value": {"id": qid}}},
        }
        for qid, _ in paper_date_pairs
    ]
    return {"entities": {"Q9999": {"claims": {"P265": p265}}}}


def _make_paper_entities(paper_date_pairs: list[tuple[str, str]]) -> dict:
    """Build a fake wbgetentities response mapping paper QIDs to their P28 dates."""
    entities = {}
    for qid, date in paper_date_pairs:
        entities[qid] = {
            "claims": {
                "P28": [{"mainsnak": {"datavalue": {"value": {"time": date}}}}]
            } if date else {}
        }
    return {"entities": entities}


def _mock_session(theme_response, papers_response, remove_success=True) -> MagicMock:
    """Session mock where _session is pre-set (no login calls needed)."""
    s = MagicMock()
    csrf_resp = MagicMock()
    csrf_resp.json.return_value = {"query": {"tokens": {"csrftoken": "csrf"}}}
    theme_resp = MagicMock()
    theme_resp.raise_for_status = MagicMock()
    theme_resp.json.return_value = theme_response
    papers_resp = MagicMock()
    papers_resp.raise_for_status = MagicMock()
    papers_resp.json.return_value = papers_response
    remove_resp = MagicMock()
    remove_resp.json.return_value = {"success": 1} if remove_success else {"error": "fail"}

    # get calls: theme entities, paper entities, csrf token
    s.get.side_effect = [theme_resp, papers_resp, csrf_resp]
    s.post.side_effect = [remove_resp]
    return s


class TestEnforceThemeLimit:
    def _client(self, session) -> KGClient:
        mc = FakeMC()
        client = KGClient(mc, api_url="https://example.org/api", bot_user="bot", bot_password="pw")
        client._session = session
        return client

    def test_no_op_when_under_limit(self):
        pairs = [("Q1", "2024-01-01"), ("Q2", "2024-02-01")]
        session = MagicMock()
        theme_resp = MagicMock()
        theme_resp.raise_for_status = MagicMock()
        theme_resp.json.return_value = _make_p265_entities(pairs)
        session.get.return_value = theme_resp
        client = self._client(session)
        assert client.enforce_theme_limit("Q9999", max_papers=5) == 0
        session.post.assert_not_called()

    def test_removes_oldest_when_over_limit(self):
        pairs = [("Q1", "+2024-01-01T00:00:00Z"), ("Q2", "+2024-06-01T00:00:00Z"), ("Q3", "+2024-03-01T00:00:00Z")]
        session = _mock_session(_make_p265_entities(pairs), _make_paper_entities(pairs))
        client = self._client(session)
        removed = client.enforce_theme_limit("Q9999", max_papers=2)
        assert removed == 1
        # wbremoveclaims should have been called with Q1's GUID (oldest)
        post_call = session.post.call_args
        assert "Q9999$Q1" in post_call.kwargs.get("data", {}).get("claim", "")

    def test_no_op_when_max_papers_zero(self):
        client = KGClient(FakeMC(), api_url="https://example.org/api")
        assert client.enforce_theme_limit("Q9999", max_papers=0) == 0

    def test_no_op_when_api_url_not_set(self):
        client = KGClient(FakeMC())
        assert client.enforce_theme_limit("Q9999", max_papers=10) == 0


def test_import_paper_without_arxiv_id_writes_no_p21():
    """A paper harvested from OpenAlex with no arXiv ID should not get a P21 claim."""
    claims_written = []

    class FakeItem:
        def add_claim(self, prop, value=None, qualifiers=None):
            claims_written.append(prop)
        def get_value(self, prop):
            return []
        def write(self):
            class R:
                id = "Q999"
            return R()

    class FakeMC:
        def search_entity_by_value(self, prop, val):
            return []
        class item:
            @staticmethod
            def new():
                i = FakeItem()
                i.labels = type("L", (), {"set": lambda s, lang, v: None})()
                return i

    from topic_overviews.kg import model as M

    kg = KGClient(FakeMC())
    record = PaperRecord(
        arxiv_id="",
        title="OpenAlex Only Paper",
        abstract="",
        authors=[],
        categories=[],
        published="2026-06-15",
        doi="10.1000/oa",
        openalex_id="W9876543210",
    )
    qid = kg.import_paper(record)
    assert qid == "Q999"
    assert M.P_ARXIV_ID not in claims_written
