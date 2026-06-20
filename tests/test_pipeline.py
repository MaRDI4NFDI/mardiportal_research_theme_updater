import datetime

from topic_overviews.config import load_config
from topic_overviews.state import State
from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.topics import Topic
from topic_overviews.kg.pagedata import TopicPageData, PaperEntry
from topic_overviews import pipeline

TOPICS = [Topic(qid="Q11", label="Online Algorithms", description="...")]
P1 = PaperRecord("2401.00001", "Caching", "abs", ["Jane Doe"], ["cs.DS"], "2024-01-02", None)
P2 = PaperRecord("2401.00002", "Unrelated", "abs", ["X"], ["math.AG"], "2024-01-03", None)


class FakeKG:
    def __init__(self): self.imported = []
    def import_paper(self, record, topic_qids):
        self.imported.append((record.arxiv_id, topic_qids)); return "Q999"


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
    assert kg.imported == [("2401.00001", ["Q11"])]
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


def test_generate_pages_step_publishes_topic_and_index():
    cfg = load_config({})
    published = []

    class FakePublisher:
        def edit(self, title, text, summary): published.append(title)

    def fake_page_data(endpoint, topic):
        return TopicPageData(topic.qid, topic.label, topic.description,
                             [PaperEntry("Caching", ["Jane Doe"], "2024", "2401.00001")])

    result = pipeline.generate_pages_step(cfg, topics=TOPICS, publisher=FakePublisher(),
                                          fetch_page_data=fake_page_data)
    assert [d.label for d in result] == ["Online Algorithms"]
    assert published == ["Topic:Online Algorithms", "Topic overview"]


def test_generate_pages_step_dry_run_does_not_publish():
    cfg = load_config({"TOPIC_OVERVIEWS_DRY_RUN": "true"})
    published = []

    class FakePublisher:
        def edit(self, title, text, summary): published.append(title)

    def fake_page_data(endpoint, topic):
        return TopicPageData(topic.qid, topic.label, topic.description,
                             [PaperEntry("Caching", ["Jane Doe"], "2024", "2401.00001")])

    result = pipeline.generate_pages_step(cfg, topics=TOPICS, publisher=FakePublisher(),
                                          fetch_page_data=fake_page_data)
    assert [d.label for d in result] == ["Online Algorithms"]
    assert published == []
