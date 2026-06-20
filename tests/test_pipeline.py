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
    def __init__(self):
        self.ensured = []
        self.edited = []

    def ensure_page(self, title, text, summary):
        self.ensured.append((title, text))
        return True

    def edit(self, title, text, summary):
        self.edited.append(title)


def test_generate_pages_step_ensures_theme_stub_pages_and_index():
    cfg = load_config({})
    pub = FakePublisher()
    result = pipeline.generate_pages_step(cfg, topics=TOPICS, publisher=pub)

    assert result == ["Online Algorithms"]
    # the theme page is created with just the template stub (not a built table)
    assert pub.ensured == [("Online Algorithms", "{{ResearchTheme}}\n")]
    # the master index is (over)written
    assert pub.edited == ["Research themes"]


def test_generate_pages_step_dry_run_writes_nothing():
    cfg = load_config({"TOPIC_OVERVIEWS_DRY_RUN": "true"})
    pub = FakePublisher()
    result = pipeline.generate_pages_step(cfg, topics=TOPICS, publisher=pub)
    assert result == ["Online Algorithms"]
    assert pub.ensured == []
    assert pub.edited == []
