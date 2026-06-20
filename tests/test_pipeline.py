import datetime

from topic_overviews.config import load_config
from topic_overviews.state import State
from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.topics import Topic
from topic_overviews import pipeline

TOPICS = [Topic(qid="Q11", label="Online Algorithms", description="...")]
P1 = PaperRecord("2401.00001", "Caching", "abs", ["Jane Doe"], ["cs.DS"], "2024-01-02", None)
P2 = PaperRecord("2401.00002", "Unrelated", "abs", ["X"], ["math.AG"], "2024-01-03", None)


class FakeKG:
    def __init__(self):
        self.imported = []
        self.links = []
    def import_paper(self, record):
        self.imported.append(record.arxiv_id); return "Q999"
    def link_topic(self, topic_qid, paper_qid):
        self.links.append((topic_qid, paper_qid))


def test_harvest_step_imports_only_matched_and_updates_state():
    cfg = load_config({"TOPIC_OVERVIEWS_DRY_RUN": "false"})
    state = State()
    kg = FakeKG()

    def fake_fetch(from_date, set_spec): return iter([P1, P2])
    def fake_classify(paper, topics, *, model, api_key):
        return ["Q11"] if paper.arxiv_id == "2401.00001" else []

    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg,
                                  fetch=fake_fetch, classify=fake_classify)
    assert count == 1
    assert kg.imported == ["2401.00001"]
    assert kg.links == [("Q11", "Q999")]            # paper Q999 added to topic Q11 via P265
    assert state.seen_ids == {"2401.00001", "2401.00002"}
    assert state.last_harvest == datetime.date.today().isoformat()


def test_harvest_step_skips_seen_ids():
    cfg = load_config({})
    state = State(seen_ids={"2401.00001"})
    kg = FakeKG()
    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg,
                                  fetch=lambda f, set_spec: iter([P1]),
                                  classify=lambda *a, **k: ["Q11"])
    assert count == 0
    assert kg.imported == []


def test_harvest_step_dry_run_does_not_import():
    cfg = load_config({"TOPIC_OVERVIEWS_DRY_RUN": "true"})
    state = State()
    kg = FakeKG()
    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg,
                                  fetch=lambda f, set_spec: iter([P1]),
                                  classify=lambda *a, **k: ["Q11"])
    assert count == 1 and kg.imported == []


def test_harvest_step_respects_harvest_limit():
    cfg = load_config({"TOPIC_OVERVIEWS_HARVEST_LIMIT": "2"})
    state = State()
    kg = FakeKG()
    P3 = PaperRecord("2401.00003", "Third", "abs", ["Z"], ["math.CO"], "2024-01-04", None)
    calls = []

    def fake_fetch(from_date, set_spec):
        return iter([P1, P2, P3])

    def fake_classify(paper, topics, *, model, api_key):
        calls.append(paper.arxiv_id)
        return ["Q11"]

    pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg,
                          fetch=fake_fetch, classify=fake_classify)
    # only the first 2 new papers are considered/classified; the 3rd is never reached
    assert calls == ["2401.00001", "2401.00002"]
    assert state.seen_ids == {"2401.00001", "2401.00002"}


def test_harvest_step_isolates_failing_paper():
    cfg = load_config({"TOPIC_OVERVIEWS_DRY_RUN": "false"})
    state = State()
    kg = FakeKG()

    def fake_fetch(from_date, set_spec): return iter([P1, P2])
    def fake_classify(paper, topics, *, model, api_key):
        if paper.arxiv_id == "2401.00001":
            return ["Q11"]
        raise RuntimeError("classify exploded")

    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg,
                                  fetch=fake_fetch, classify=fake_classify)
    assert count == 1
    assert kg.imported == ["2401.00001"]
    assert kg.links == [("Q11", "Q999")]
    assert "2401.00001" in state.seen_ids
    assert "2401.00002" in state.seen_ids


class FakePublisher:
    def __init__(self, existing_pages=None):
        self._existing = set(existing_pages or [])
        self.edited = []

    def page_exists(self, title):
        return title in self._existing

    def edit(self, title, text, summary):
        self.edited.append((title, text))


class FakeSitelinkKG:
    def __init__(self, connected=None):
        self._connected = dict(connected or {})   # qid -> page title
        self.sitelinks_set = []

    def get_theme_sitelink(self, qid):
        return self._connected.get(qid)

    def set_theme_sitelink(self, qid, title):
        self.sitelinks_set.append((qid, title))
        self._connected[qid] = title


def test_ensure_theme_pages_creates_page_and_sitelink_when_unconnected():
    cfg = load_config({})
    pub = FakePublisher()
    kg = FakeSitelinkKG()
    result = pipeline.ensure_theme_pages_step(cfg, topics=TOPICS, publisher=pub, kg=kg)

    assert result == ["Online Algorithms"]
    # page created with the stub, then sitelink wired to it
    assert ("Online Algorithms", "{{ResearchTheme}}\n") in pub.edited
    assert kg.sitelinks_set == [("Q11", "Online Algorithms")]
    # index page also written
    assert ("Research themes", "= Research themes =\n\n* [[Online Algorithms]]\n") in pub.edited


def test_ensure_theme_pages_skips_already_connected_theme():
    cfg = load_config({})
    pub = FakePublisher()
    kg = FakeSitelinkKG(connected={"Q11": "Existing Page"})
    pipeline.ensure_theme_pages_step(cfg, topics=TOPICS, publisher=pub, kg=kg)
    assert kg.sitelinks_set == []                       # no sitelink rewired
    assert all(t != "Online Algorithms" for t, _ in pub.edited)  # no theme page created
    assert any(t == "Research themes" for t, _ in pub.edited)    # index still refreshed


def test_ensure_theme_pages_does_not_hijack_existing_page():
    cfg = load_config({})
    pub = FakePublisher(existing_pages={"Online Algorithms"})   # unrelated page already there
    kg = FakeSitelinkKG()
    pipeline.ensure_theme_pages_step(cfg, topics=TOPICS, publisher=pub, kg=kg)
    assert kg.sitelinks_set == []                       # did not hijack
    assert all(t != "Online Algorithms" for t, _ in pub.edited)


def test_ensure_theme_pages_dry_run_writes_nothing():
    cfg = load_config({"TOPIC_OVERVIEWS_DRY_RUN": "true"})
    pub = FakePublisher()
    kg = FakeSitelinkKG()
    result = pipeline.ensure_theme_pages_step(cfg, topics=TOPICS, publisher=pub, kg=kg)
    assert result == ["Online Algorithms"]
    assert pub.edited == []
    assert kg.sitelinks_set == []
