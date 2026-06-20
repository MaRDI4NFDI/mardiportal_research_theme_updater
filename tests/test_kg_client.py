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
    qid = KGClient(mc).import_paper(PAPER)

    assert qid == "Q500"
    assert mc.searched == [("wdt:P21", "2401.00001")]
    assert item.label == ("en", "A New Bound for Online Caching")
    assert ("wdt:P31", "wd:Q56887") in item.claims          # instance of scholarly article
    assert ("wdt:P21", "2401.00001") in item.claims         # arXiv id
    assert ("wdt:P27", "10.1000/xyz") in item.claims        # DOI
    assert ("wdt:P159", "A New Bound for Online Caching") in item.claims
    assert ("wdt:P28", "+2024-01-02T00:00:00Z") in item.claims
    assert ("wdt:P22", "math.OC") in item.claims
    assert ("wdt:P43", "Jane Doe") in item.claims
    # The paper carries NO topic/membership statement — papers stay topic-agnostic.
    assert not any(prop == "wdt:P265" for prop, _ in item.claims)
    assert not any(prop == "wdt:P30" for prop, _ in item.claims)


def test_import_existing_paper_reuses_item():
    item = FakeItem()
    mc = FakeMC(existing=["Q500"], item=item)
    qid = KGClient(mc).import_paper(PAPER)
    assert qid == "Q500"
    # existing item fetched, not newly labelled
    assert item.label is None
    assert ("wdt:P31", "wd:Q56887") in item.claims      # claims still written on existing item


def test_link_topic_adds_paper_to_topic_when_absent():
    topic = FakeItem(item_id="Q11")
    mc = FakeMC(item=topic)
    KGClient(mc).link_topic("Q11", "Q500")
    assert ("wdt:P265", "wd:Q500") in topic.claims      # has part(s) -> paper, on the TOPIC item
    assert topic.written is True


def test_link_topic_is_idempotent_when_paper_already_listed():
    topic = FakeItem(item_id="Q11", values={"wdt:P265": ["Q500", "Q777"]})
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
