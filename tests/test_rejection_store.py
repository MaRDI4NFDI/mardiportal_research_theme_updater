"""Rejection store: remembered "no theme" decisions skip the LLM on later runs."""
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from lakefs.exceptions import NotFoundException

from topic_overviews import pipeline
from topic_overviews.config import load_config
from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.topics import Topic
from topic_overviews.rejection_store import (
    RejectionStore, fingerprint, load_store, save_store,
)
from topic_overviews.state import State

TOPICS = [Topic(qid="Q11", label="Online Algorithms", description="caching etc.")]
P1 = PaperRecord("2401.00001", "Caching", "abs", ["Jane Doe"], ["cs.DS"], "2024-01-02", None)
P2 = PaperRecord("2401.00002", "Unrelated", "abs", ["X"], ["math.AG"], "2024-01-03", None)
LAKEFS = dict(url="https://lake.example", user="u", password="p", repo="mardi-workflows-files")


# --- fingerprint -----------------------------------------------------------

def test_fingerprint_stable_and_order_independent():
    a = Topic("Q1", "A", "da")
    b = Topic("Q2", "B", "db")
    assert fingerprint([a, b], "m") == fingerprint([b, a], "m")


@pytest.mark.parametrize("change", [
    lambda: ([Topic("Q11", "Online Algorithms", "edited")], "m"),          # description
    lambda: ([Topic("Q11", "Renamed", "caching etc.")], "m"),              # label
    lambda: (TOPICS + [Topic("Q12", "New theme", "x")], "m"),              # theme added
    lambda: (TOPICS, "other-model"),                                       # model
])
def test_fingerprint_changes_with_question(change):
    assert fingerprint(*change()) != fingerprint(TOPICS, "m")


# --- store ----------------------------------------------------------------

def test_add_then_is_rejected_counts_hits():
    store = RejectionStore(today=lambda: "2026-11-23")
    assert not store.is_rejected("fp", "r1")
    store.add("fp", "r1")
    assert store.dirty and len(store) == 1
    assert store.is_rejected("fp", "r1")
    assert not store.is_rejected("other-fp", "r1")
    assert store.hits == 1


def test_reclassify_never_skips_but_still_records():
    store = RejectionStore({"version": 1, "rejected": {"fp": {"r1": "2026-11-20"}}},
                           skip_known=False, today=lambda: "2026-11-23")
    assert not store.is_rejected("fp", "r1")
    store.add("fp", "r2")
    assert len(store) == 2


def test_discard_removes_entry_and_empty_fingerprint():
    store = RejectionStore({"version": 1, "rejected": {"fp": {"r1": "2026-11-20"}}})
    store.discard("fp", "r1")
    assert store.dirty and store.to_dict()["rejected"] == {}
    clean = RejectionStore()
    clean.discard("fp", "missing")
    assert not clean.dirty


def test_prune_drops_old_entries_only():
    store = RejectionStore(
        {"version": 1, "rejected": {
            "a": {"old": "2026-11-01", "new": "2026-11-20"},
            "b": {"old2": "2026-10-01"},
        }},
        today=lambda: "2026-11-23",
    )
    assert store.prune(17) == 2          # cutoff 2026-11-06
    assert store.to_dict()["rejected"] == {"a": {"new": "2026-11-20"}}
    assert store.dirty


# --- lakeFS load/save -----------------------------------------------------

def _patch_lakefs(obj):
    repo = MagicMock()
    repo.ref.return_value.object.return_value = obj
    repo.branch.return_value.object.return_value = obj
    return patch("lakefs.Client"), patch("lakefs.repository", return_value=repo), repo


def test_load_reads_existing_file():
    obj = MagicMock()
    obj.reader.return_value.__enter__.return_value = io.BytesIO(
        json.dumps({"version": 1, "rejected": {"fp": {"r1": "2026-11-20"}}}).encode()
    )
    c, r, repo = _patch_lakefs(obj)
    with c, r as mock_repository:
        store = load_store(**LAKEFS)
    mock_repository.assert_called_once()
    assert mock_repository.call_args.args[0] == "mardi-workflows-files"
    repo.ref.assert_called_once_with("main")
    assert store.is_rejected("fp", "r1")


def test_load_missing_file_gives_empty_store():
    obj = MagicMock()
    obj.reader.return_value.__enter__.side_effect = NotFoundException(404, "not found")
    c, r, _ = _patch_lakefs(obj)
    with c, r:
        store = load_store(**LAKEFS)
    assert len(store) == 0


def test_load_failure_never_raises():
    with patch("lakefs.Client", side_effect=RuntimeError("boom")):
        store = load_store(**LAKEFS, skip_known=False)
    assert len(store) == 0 and store.skip_known is False


def test_load_without_lakefs_config_gives_empty_store():
    assert len(load_store(url="", user="", password="", repo="r")) == 0


def test_save_uploads_and_commits_when_dirty():
    obj = MagicMock()
    c, r, repo = _patch_lakefs(obj)
    store = RejectionStore(today=lambda: "2026-11-23")
    store.add("fp", "r1")
    with c, r:
        assert save_store(store, **LAKEFS, metadata={"model": "m"}) is True
    body = obj.upload.call_args.args[0]
    assert json.loads(body) == {"version": 1, "rejected": {"fp": {"r1": "2026-11-23"}}}
    assert obj.upload.call_args.kwargs["pre_sign"] is False
    repo.branch.return_value.commit.assert_called_once()
    assert repo.branch.return_value.commit.call_args.kwargs["metadata"] == {"model": "m"}
    assert not store.dirty


def test_save_skips_unchanged_store():
    with patch("lakefs.Client") as client:
        assert save_store(RejectionStore(), **LAKEFS) is False
    client.assert_not_called()


def test_save_failure_never_raises():
    store = RejectionStore()
    store.add("fp", "r1")
    with patch("lakefs.Client", side_effect=RuntimeError("boom")):
        assert save_store(store, **LAKEFS) is False
    assert store.dirty


# --- pipeline integration -------------------------------------------------

class _KG:
    def __init__(self):
        self.imported = []
    def import_paper(self, record, keywords=None):
        self.imported.append(record.arxiv_id); return "Q999"
    def link_topic(self, *a): pass
    def add_tldr(self, *a): pass
    def find_existing_paper(self, record): return None
    def paper_has_tldr(self, qid): return False
    def enforce_theme_limit(self, *a): return 0


def _run(store, classify, dry_run=False):
    cfg = load_config({
        "TOPIC_OVERVIEWS_ARXIV_QUERY": "cat:cs.DS",
        "TOPIC_OVERVIEWS_DRY_RUN": "true" if dry_run else "false",
    })
    return pipeline.harvest_step(
        cfg, State(), topics=TOPICS, kg=_KG(), model="m",
        fetch=lambda c: iter([P1, P2]), classify=classify,
        summarize=lambda *a, **k: "tldr", keyworder=lambda *a, **k: ["kw"],
        lookup_zb=lambda arxiv_id: None, rejections=store,
    )


def test_pipeline_records_rejection_and_skips_it_next_run():
    calls = []

    def classify(paper, topics, **kw):
        calls.append(paper.arxiv_id)
        return ["Q11"] if paper.arxiv_id == P1.arxiv_id else []

    store = RejectionStore()
    _run(store, classify)
    assert calls == [P1.arxiv_id, P2.arxiv_id]
    assert store.added == 1

    calls.clear()
    second = RejectionStore(store.to_dict())
    _run(second, classify)
    assert calls == [P1.arxiv_id]          # P2 skipped: already rejected
    assert second.hits == 1


def test_pipeline_reclassify_asks_again():
    calls = []
    fp = fingerprint(TOPICS, "m")
    store = RejectionStore({"version": 1, "rejected": {fp: {P2.arxiv_id: "2026-11-22"}}},
                           skip_known=False)

    def classify(paper, topics, **kw):
        calls.append(paper.arxiv_id)
        return ["Q11"]                     # now P2 matches after all

    _run(store, classify)
    assert calls == [P1.arxiv_id, P2.arxiv_id]
    assert len(store) == 0                 # stale rejection removed


def test_pipeline_does_not_record_classifier_errors():
    def classify(paper, topics, **kw):
        raise ValueError("classifier returned no parsable JSON after 2 attempts")

    store = RejectionStore()
    _run(store, classify)
    assert len(store) == 0


def test_keyword_match_bypasses_store():
    kw_topics = [Topic(qid="Q11", label="Online Algorithms", description="caching etc.",
                       auto_classify_keywords=["caching"])]
    fp = fingerprint(kw_topics, "m")
    store = RejectionStore({"version": 1, "rejected": {fp: {P1.arxiv_id: "2026-11-22"}}})
    cfg = load_config({"TOPIC_OVERVIEWS_ARXIV_QUERY": "cat:cs.DS"})
    kg = _KG()
    pipeline.harvest_step(
        cfg, State(), topics=kw_topics, kg=kg, model="m",
        fetch=lambda c: iter([P1]), classify=lambda *a, **k: [],
        summarize=lambda *a, **k: "tldr", keyworder=lambda *a, **k: ["kw"],
        lookup_zb=lambda arxiv_id: None, rejections=store,
    )
    assert kg.imported == [P1.arxiv_id]   # keyword rule wins over a stored rejection


def test_unconfigured_lakefs_warns_on_load_and_save(caplog):
    import logging
    from topic_overviews.rejection_store import load_store, save_store

    with caplog.at_level(logging.WARNING):
        store = load_store(url="", user="", password="", repo="r")
        store.add("fp", "rid")
        assert save_store(store, url="", user="", password="", repo="r") is False
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("DISABLED" in m and "url, user" in m for m in messages)
    assert any("NOT saved" in m and "1 new" in m for m in messages)
