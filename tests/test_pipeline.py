import datetime
import logging

from topic_overviews.config import load_config
from topic_overviews.state import State
from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.topics import Topic
from topic_overviews import pipeline

OA_TOPICS = [
    Topic(qid="Q20", label="MaRDI", description="...", openalex_query="search=mardi&filter=funders.id:f123"),
]
OA_PAPER = PaperRecord("", "MaRDI Portal Paper", "abs", ["Dr. X"], [], "2024-06-01", doi="10.1/xyz", openalex_id="W9999999999")

TOPICS = [Topic(qid="Q11", label="Online Algorithms", description="...")]
QUERY_TOPICS = [
    Topic(qid="Q11", label="Online Algorithms", description="...", arxiv_query="cat:cs.DS"),
    Topic(qid="Q12", label="Numerical Analysis", description="...", arxiv_query="cat:math.NA"),
]
P1 = PaperRecord("2401.00001", "Caching", "abs", ["Jane Doe"], ["cs.DS"], "2024-01-02", None)
P2 = PaperRecord("2401.00002", "Unrelated", "abs", ["X"], ["math.AG"], "2024-01-03", None)


def _cfg(env=None):
    values = {"TOPIC_OVERVIEWS_ARXIV_QUERY": "cat:cs.DS"}
    values.update(env or {})
    return load_config(values)


class FakeKG:
    def __init__(self):
        self.imported = []        # (arxiv_id, tldr, keywords)
        self.links = []
        self.paper_qids = {}
        self.paper_tldrs = set()
    def import_paper(self, record, tldr=None, keywords=None, generated_by=None):
        self.imported.append((record.arxiv_id, tldr, keywords)); return "Q999"
    def link_topic(self, topic_qid, paper_qid):
        self.links.append((topic_qid, paper_qid))
    def get_paper_qid(self, arxiv_id):
        return self.paper_qids.get(arxiv_id)
    def paper_has_tldr(self, paper_qid):
        return paper_qid in self.paper_tldrs
    def enforce_theme_limit(self, topic_qid, max_papers):
        return 0


def _summ(text="tl;dr"):
    return lambda paper, *a, **k: text


def _kw(kws=("kw",)):
    return lambda paper, *a, **k: list(kws)


def test_harvest_step_imports_only_matched_and_updates_state():
    cfg = _cfg({"TOPIC_OVERVIEWS_DRY_RUN": "false"})
    state = State()
    kg = FakeKG()

    def fake_fetch(config): return iter([P1, P2])
    def fake_classify(paper, topics, *, model, api_key, **kwargs):
        return ["Q11"] if paper.arxiv_id == "2401.00001" else []

    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg, model="test-model",
                                  fetch=fake_fetch, classify=fake_classify,
                                  summarize=_summ("a short summary"), keyworder=_kw(["A", "B"]))
    assert count == 1
    assert kg.imported == [("2401.00001", "a short summary", ["A", "B"])]   # tldr + keywords passed through
    assert kg.links == [("Q11", "Q999")]            # paper Q999 added to topic Q11 via P265
    assert state.seen_ids == {"2401.00001", "2401.00002"}
    assert state.last_harvest == datetime.date.today().isoformat()


def test_harvest_step_logs_inserted_qids(caplog):
    cfg = _cfg({"TOPIC_OVERVIEWS_DRY_RUN": "false"})
    state = State()
    kg = FakeKG()

    def fake_fetch(config):
        return iter([P1])

    def fake_classify(paper, topics, *, model, api_key, **kwargs):
        return ["Q11"]

    kg.import_paper = lambda record, tldr=None, keywords=None, generated_by=None: "Q123"

    with caplog.at_level(logging.INFO, logger="topic_overviews.pipeline"):
        count = pipeline.harvest_step(
            cfg,
            state,
            topics=TOPICS,
            kg=kg,
            model="test-model",
            fetch=fake_fetch,
            classify=fake_classify,
            summarize=_summ("a short summary"),
            keyworder=_kw(["A", "B"]),
        )

    assert count == 1
    assert "Classifying arXiv paper 2401.00001 (Caching) with model test-model" in caplog.text
    assert "Generating TL;DR and keywords for 2401.00001 (Caching)" in caplog.text
    assert "Inserted arXiv paper 2401.00001 as KG item Q123" in caplog.text
    assert "Harvest inserted 1 paper(s): Q123" in caplog.text


def test_harvest_step_skips_papers_that_already_have_tldr(caplog):
    cfg = _cfg({"TOPIC_OVERVIEWS_DRY_RUN": "false"})
    state = State()
    kg = FakeKG()
    kg.paper_qids["2401.00001"] = "Q555"
    kg.paper_tldrs.add("Q555")

    with caplog.at_level(logging.INFO, logger="topic_overviews.pipeline"):
        count = pipeline.harvest_step(
            cfg,
            state,
            topics=TOPICS,
            kg=kg,
            model="test-model",
            fetch=lambda config: iter([P1]),
            classify=lambda *a, **k: ["Q11"],
            summarize=_summ("a short summary"),
            keyworder=_kw(["A", "B"]),
        )

    assert count == 0
    assert kg.imported == []
    assert kg.links == []
    assert "Skipping arXiv paper 2401.00001 (Caching): KG item Q555 already has P1963" in caplog.text


def test_harvest_step_skips_seen_ids():
    cfg = _cfg()
    state = State(seen_ids={"2401.00001"})
    kg = FakeKG()
    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg, model="test-model",
                                  fetch=lambda config: iter([P1]),
                                  classify=lambda *a, **k: ["Q11"])
    assert count == 0
    assert kg.imported == []


def test_harvest_step_dry_run_does_not_import():
    cfg = _cfg({"TOPIC_OVERVIEWS_DRY_RUN": "true"})
    state = State()
    kg = FakeKG()
    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg, model="test-model",
                                  fetch=lambda config: iter([P1]),
                                  classify=lambda *a, **k: ["Q11"])
    assert count == 1 and kg.imported == []


def test_harvest_step_respects_harvest_limit():
    cfg = _cfg({"TOPIC_OVERVIEWS_HARVEST_LIMIT": "2"})
    state = State()
    kg = FakeKG()
    P3 = PaperRecord("2401.00003", "Third", "abs", ["Z"], ["math.CO"], "2024-01-04", None)
    calls = []

    def fake_fetch(config):
        return iter([P1, P2, P3])

    def fake_classify(paper, topics, *, model, api_key, **kwargs):
        calls.append(paper.arxiv_id)
        return ["Q11"]

    pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg, model="test-model",
                          fetch=fake_fetch, classify=fake_classify, summarize=_summ(), keyworder=_kw())
    # only the first 2 new papers are considered/classified; the 3rd is never reached
    assert calls == ["2401.00001", "2401.00002"]
    assert state.seen_ids == {"2401.00001", "2401.00002"}


def test_harvest_step_uses_theme_arxiv_queries():
    cfg = _cfg({"TOPIC_OVERVIEWS_ARXIV_QUERY": "fallback"})
    state = State()
    kg = FakeKG()
    queries = []

    def fake_fetch(config):
        queries.append(config.arxiv_query)
        if config.arxiv_query == "cat:cs.DS":
            return iter([P1])
        return iter([P2])

    def fake_classify(paper, topics, *, model, api_key, **kwargs):
        return ["Q11"] if paper.arxiv_id == "2401.00001" else ["Q12"]

    count = pipeline.harvest_step(
        cfg,
        state,
        topics=QUERY_TOPICS,
        kg=kg,
        model="test-model",
        fetch=fake_fetch,
        classify=fake_classify,
        summarize=_summ(),
        keyworder=_kw(),
    )
    assert count == 2
    assert queries == ["cat:cs.DS", "cat:math.NA"]
    assert kg.links == [("Q11", "Q999"), ("Q12", "Q999")]


def test_harvest_step_deduplicates_theme_arxiv_queries():
    cfg = _cfg()
    topics = [
        Topic(qid="Q11", label="A", description="", arxiv_query="cat:math.NA"),
        Topic(qid="Q12", label="B", description="", arxiv_query="cat:math.NA"),
    ]
    queries = []

    pipeline.harvest_step(
        cfg,
        State(),
        topics=topics,
        kg=FakeKG(),
        model="test-model",
        fetch=lambda config: queries.append(config.arxiv_query) or iter([]),
        classify=lambda *a, **k: [],
    )
    assert queries == ["cat:math.NA"]


def test_harvest_step_purges_imported_paper_pages():
    cfg = _cfg()
    state = State()
    kg = FakeKG()

    class PurgePub:
        def __init__(self): self.purged = []
        def purge(self, titles): self.purged.extend(titles)

    pub = PurgePub()
    pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg, model="test-model",
                          fetch=lambda config: iter([P1, P2]),
                          classify=lambda paper, *a, **k: ["Q11"] if paper.arxiv_id == "2401.00001" else [],
                          summarize=_summ(), keyworder=_kw(), publisher=pub)
    # only the matched/imported paper's page title is purged
    assert pub.purged == ["Caching"]


def test_harvest_step_isolates_failing_paper():
    cfg = _cfg({"TOPIC_OVERVIEWS_DRY_RUN": "false"})
    state = State()
    kg = FakeKG()

    def fake_fetch(config): return iter([P1, P2])
    def fake_classify(paper, topics, *, model, api_key, **kwargs):
        if paper.arxiv_id == "2401.00001":
            return ["Q11"]
        raise RuntimeError("classify exploded")

    count = pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg, model="test-model",
                                  fetch=fake_fetch, classify=fake_classify, summarize=_summ(), keyworder=_kw())
    assert count == 1
    assert kg.imported == [("2401.00001", "tl;dr", ["kw"])]
    assert kg.links == [("Q11", "Q999")]
    assert "2401.00001" in state.seen_ids
    assert "2401.00002" in state.seen_ids


def test_harvest_step_runs_openalex_pass():
    cfg = _cfg()
    state = State()
    kg = FakeKG()
    oa_records = []

    def fake_fetch_oa(query_str, since_days, **kwargs):
        oa_records.append(query_str)
        return iter([OA_PAPER])

    def fake_classify(paper, topics, *, model, api_key, **kwargs):
        return ["Q20"]

    count = pipeline.harvest_step(
        cfg,
        state,
        topics=OA_TOPICS,
        kg=kg,
        model="test-model",
        fetch=lambda config: iter([]),         # arXiv yields nothing
        fetch_oa=fake_fetch_oa,
        classify=fake_classify,
        summarize=_summ("tldr"),
        keyworder=_kw(["kw"]),
    )
    assert count == 1
    assert oa_records == ["search=mardi&filter=funders.id:f123"]
    assert kg.imported[0][0] == ""             # arxiv_id empty (OpenAlex-only paper)
    assert "openalex:W9999999999" in state.seen_ids


def test_harvest_step_deduplicates_openalex_queries():
    cfg = _cfg()
    topics = [
        Topic(qid="Q20", label="A", description="", openalex_query="search=mardi"),
        Topic(qid="Q21", label="B", description="", openalex_query="search=mardi"),
    ]
    oa_calls = []

    pipeline.harvest_step(
        cfg,
        State(),
        topics=topics,
        kg=FakeKG(),
        model="test-model",
        fetch=lambda config: iter([]),
        fetch_oa=lambda qs, sd, **kw: oa_calls.append(qs) or iter([]),
        classify=lambda *a, **k: [],
    )
    assert oa_calls == ["search=mardi"]


def test_harvest_step_skips_seen_openalex_ids():
    cfg = _cfg()
    state = State(seen_ids={"openalex:W9999999999"})
    kg = FakeKG()

    count = pipeline.harvest_step(
        cfg,
        state,
        topics=OA_TOPICS,
        kg=kg,
        model="test-model",
        fetch=lambda config: iter([]),
        fetch_oa=lambda qs, sd, **kw: iter([OA_PAPER]),
        classify=lambda *a, **k: ["Q20"],
    )
    assert count == 0
    assert kg.imported == []


def test_harvest_step_no_openalex_when_no_query():
    cfg = _cfg()
    topics = [Topic(qid="Q11", label="Online Algorithms", description="...")]
    oa_calls = []

    pipeline.harvest_step(
        cfg,
        State(),
        topics=topics,
        kg=FakeKG(),
        model="test-model",
        fetch=lambda config: iter([]),
        fetch_oa=lambda qs, sd, **kw: oa_calls.append(qs) or iter([]),
        classify=lambda *a, **k: [],
    )
    assert oa_calls == []


class FakePublisher:
    def __init__(self, existing_pages=None):
        self._existing = set(existing_pages or [])
        self.edited = []
        self.purged = []

    def page_exists(self, title):
        return title in self._existing

    def edit(self, title, text, summary):
        self.edited.append((title, text))

    def purge(self, titles):
        self.purged.extend(titles)


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
    assert all(t != "Research themes" for t, _ in pub.edited)
    # the new theme page is purged so it renders fresh
    assert pub.purged == ["Online Algorithms"]


def test_ensure_theme_pages_skips_already_connected_theme():
    cfg = load_config({})
    pub = FakePublisher()
    kg = FakeSitelinkKG(connected={"Q11": "Existing Page"})
    pipeline.ensure_theme_pages_step(cfg, topics=TOPICS, publisher=pub, kg=kg)
    assert kg.sitelinks_set == []                       # no sitelink rewired
    assert all(t != "Online Algorithms" for t, _ in pub.edited)  # no theme page created
    assert pub.edited == []
    assert pub.purged == ["Existing Page"]


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
