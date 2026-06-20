from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.client import KGClient, to_wbi_time

PAPER = PaperRecord(
    arxiv_id="2401.00001", title="A New Bound for Online Caching",
    abstract="...", authors=["Jane Doe", "John Smith"],
    categories=["math.OC", "cs.DS"], published="2024-01-02", doi="10.1000/xyz",
)


class FakeItem:
    def __init__(self):
        self.claims = []
        self.label = None
        self.id = "Q500"

    class _Labels:
        def __init__(self, outer): self.outer = outer
        def set(self, language, value): self.outer.label = (language, value)

    @property
    def labels(self): return FakeItem._Labels(self)

    def add_claim(self, prop, value=None, action="append_or_replace"):
        self.claims.append((prop, value))

    def write(self): return self


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


def test_import_new_paper_writes_all_statements_and_links():
    item = FakeItem()
    mc = FakeMC(existing=[], item=item)
    qid = KGClient(mc).import_paper(PAPER, ["Q11", "Q12"])

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
    assert ("wdt:P30", "wd:Q11") in item.claims             # main subject -> topic
    assert ("wdt:P30", "wd:Q12") in item.claims


def test_import_existing_paper_reuses_item():
    item = FakeItem()
    mc = FakeMC(existing=["Q500"], item=item)
    qid = KGClient(mc).import_paper(PAPER, ["Q11"])
    assert qid == "Q500"
    # existing item fetched, not newly labelled
    assert item.label is None
    assert ("wdt:P31", "wd:Q56887") in item.claims      # claims still written on existing item
    assert ("wdt:P30", "wd:Q11") in item.claims          # topic link added to existing item
