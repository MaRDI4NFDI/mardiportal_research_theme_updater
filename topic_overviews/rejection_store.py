"""Remember papers the LLM classifier rejected, so later runs skip them.

With since_days=10 and a daily schedule, every paper a query returns is seen
about ten times. A paper the LLM placed in no theme would be classified again
on each of those runs. This store records such "no theme" decisions in a JSON
file in lakeFS and lets the pipeline skip the LLM call next time.

A decision is only valid for the exact question that was asked. Each entry is
therefore keyed by a fingerprint of the covering themes (QID, label,
description), the model and the classifier prompt. When any of those change
(new theme on the same query, edited description, new model), the fingerprint
changes and the paper is classified again.

File format::

    {"version": 1, "rejected": {"<fingerprint>": {"<record_id>": "YYYY-MM-DD"}}}

Entries older than the largest harvest window (plus a margin) are pruned,
because those papers no longer come back from the queries.

lakeFS problems never stop a run: loading falls back to an empty store,
saving only logs a warning.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging

from .kg.topics import Topic

log = logging.getLogger(__name__)

STATE_PATH = "topic-overviews/rejected.json"
FORMAT_VERSION = 1
PRUNE_MARGIN_DAYS = 7


def _today() -> str:
    return _dt.date.today().isoformat()


def fingerprint(covering: list[Topic], model: str | None) -> str:
    """Identify the classification question asked for one covering theme set."""
    from .llm.topic_classifier import _SYSTEM

    themes = sorted((t.qid, t.label, t.description or "") for t in covering)
    payload = json.dumps([model or "", _SYSTEM, themes], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RejectionStore:
    """In-memory set of rejected (fingerprint, record_id) pairs.

    ``skip_known=False`` (the ``reclassify`` flow parameter) keeps recording
    new decisions but never skips a paper, so every paper is classified again.
    """

    def __init__(self, data: dict | None = None, *, skip_known: bool = True, today=_today):
        self._today = today
        rejected = (data or {}).get("rejected", {})
        self._rejected: dict[str, dict[str, str]] = {
            fp: dict(entries) for fp, entries in rejected.items() if isinstance(entries, dict)
        }
        self.skip_known = skip_known
        self.dirty = False
        self.hits = 0
        self.added = 0

    def __len__(self) -> int:
        return sum(len(v) for v in self._rejected.values())

    def is_rejected(self, fp: str, record_id: str) -> bool:
        """True if the paper should be skipped (and counts it as a hit)."""
        if not self.skip_known or record_id not in self._rejected.get(fp, {}):
            return False
        self.hits += 1
        return True

    def add(self, fp: str, record_id: str) -> None:
        self._rejected.setdefault(fp, {})[record_id] = self._today()
        self.dirty = True
        self.added += 1

    def discard(self, fp: str, record_id: str) -> None:
        """Forget a rejection (a reclassification matched a theme after all)."""
        entries = self._rejected.get(fp, {})
        if entries.pop(record_id, None) is not None:
            self.dirty = True
            if not entries:
                del self._rejected[fp]

    def prune(self, max_age_days: int) -> int:
        """Drop entries recorded more than *max_age_days* ago; return how many."""
        cutoff = (_dt.date.fromisoformat(self._today()) - _dt.timedelta(days=max_age_days)).isoformat()
        removed = 0
        for fp in list(self._rejected):
            entries = self._rejected[fp]
            for rid in [r for r, day in entries.items() if day < cutoff]:
                del entries[rid]
                removed += 1
            if not entries:
                del self._rejected[fp]
        if removed:
            self.dirty = True
        return removed

    def to_dict(self) -> dict:
        return {"version": FORMAT_VERSION, "rejected": self._rejected}


def load_store(
    *, url: str, user: str, password: str, repo: str, branch: str = "main",
    skip_known: bool = True, path: str = STATE_PATH,
) -> RejectionStore:
    """Read the store from lakeFS; an empty store if missing or unreachable."""
    if not (url and user and repo):
        log.info("Rejection store: lakeFS not configured — classifying every paper")
        return RejectionStore(skip_known=skip_known)
    try:
        import lakefs
        from lakefs.exceptions import NotFoundException

        client = lakefs.Client(host=url, username=user, password=password)
        obj = lakefs.repository(repo, client=client).ref(branch).object(path)
        try:
            with obj.reader(mode="rb", pre_sign=False) as fh:
                data = json.loads(fh.read().decode("utf-8"))
        except NotFoundException:
            log.info("Rejection store: %s/%s/%s not found — starting empty", repo, branch, path)
            return RejectionStore(skip_known=skip_known)
        if data.get("version") != FORMAT_VERSION:
            log.warning("Rejection store: unknown version %r — starting empty", data.get("version"))
            return RejectionStore(skip_known=skip_known)
        store = RejectionStore(data, skip_known=skip_known)
        log.info("Rejection store: loaded %d rejected paper(s) from %s/%s/%s", len(store), repo, branch, path)
        return store
    except Exception as exc:
        log.warning("Rejection store: could not load from lakeFS (%s) — classifying every paper", exc)
        return RejectionStore(skip_known=skip_known)


def save_store(
    store: RejectionStore, *, url: str, user: str, password: str, repo: str,
    branch: str = "main", path: str = STATE_PATH, metadata: dict | None = None,
) -> bool:
    """Upload and commit the store if it changed. Returns True if committed."""
    if not store.dirty:
        log.info("Rejection store: unchanged — nothing to save")
        return False
    if not (url and user and repo):
        return False
    try:
        import lakefs

        client = lakefs.Client(host=url, username=user, password=password)
        br = lakefs.repository(repo, client=client).branch(branch)
        body = json.dumps(store.to_dict(), ensure_ascii=False, sort_keys=True, indent=1)
        br.object(path).upload(
            body.encode("utf-8"), mode="wb",
            content_type="application/json; charset=utf-8",
            pre_sign=False,  # see lakefs_upload.upload_markdown
        )
        br.commit(
            message=f"topic-overviews: update rejected papers ({len(store)} entries)",
            metadata={k: str(v) for k, v in (metadata or {}).items()},
        )
        log.info(
            "Rejection store: saved %d entries (%d new) to %s/%s/%s",
            len(store), store.added, repo, branch, path,
        )
        store.dirty = False
        return True
    except Exception as exc:
        log.warning("Rejection store: could not save to lakeFS (%s)", exc)
        return False
