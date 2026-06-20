# Thematic Research Overviews Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python pipeline that harvests new arXiv `math.*` papers, classifies each against the research topics registered in the MaRDI Knowledge Graph (LLM), imports relevant papers into the KG with topic links, and generates one MediaWiki overview page per topic.

**Architecture:** Plain-Python package `topic_overviews` with two sequential pipeline steps (`harvest_step`, `generate_pages_step`) wired by a thin `__main__` orchestrator. Each step and unit is a plain typed function with injected dependencies, so a later port to Prefect (`@task`/`@flow`) is mechanical. The KG is the source of truth for both topics (read) and papers (write).

**Tech Stack:** Python 3.13, `requests`, `anthropic` SDK, `mardiclient` (MaRDI Wikibase client, installed from `../docker-importer/src/mardiclient`), standard-library `xml.etree` + `json`, `pytest`.

## Global Constraints

- **Python 3.13.** Match the docker-importer toolchain.
- **Env-var-first config.** All secrets/operational settings come from environment variables (so Prefect Secret blocks work later). An optional local `.env` is dev-only.
- **No topic config file.** The topic registry lives entirely in the KG (items that are `instance of` the `overview topic` class). The pipeline reads it; it never creates or edits topics.
- **Prefect-readiness.** Pipeline steps are plain typed functions with injected dependencies. Do not introduce orchestration state that resists a `@task`/`@flow` port.
- **Idempotent KG writes.** Search-before-create by arXiv ID (P21). Re-running a day must not duplicate paper items.
- **Default model `claude-haiku-4-5`** (overridable via `TOPIC_OVERVIEWS_MODEL`).
- **MaRDI Wikibase IDs (verified 2026-06-20):** instance of `P31`; arXiv ID `P21`; arXiv classification `P22`; author (item) `P16`; author name string `P43`; publication date `P28`; title `P159`; DOI `P27`; main subject `P30`; subclass of `P36`; affiliation `P55`. Classes: scholarly article `Q56887`; preprint `Q159099`. The `overview topic` class QID must be created once in the KG and supplied via `TOPIC_OVERVIEWS_OVERVIEW_TOPIC_QID`.
- **Commit messages:** plain conventional-commit style. No co-author trailers, no "Claude"/"AI" mentions.
- **TDD.** Every task: failing test → run (fail) → minimal impl → run (pass) → commit.

**Spec:** `docs/superpowers/specs/2026-06-20-thematic-research-overviews-design.md`

---

## File Structure

```
topic-overviews/
  pyproject.toml
  .env.example
  README.md
  topic_overviews/
    __init__.py
    config.py                 # Config dataclass + load_config(env)
    state.py                  # State dataclass + load_state/save_state
    harvest/
      __init__.py
      arxiv_oai.py            # PaperRecord, parse_oai_response, fetch_records
    llm/
      __init__.py
      topic_classifier.py     # classify_paper(record, topics, ...) -> list[str]
    kg/
      __init__.py
      model.py                # PID/QID constants
      sparql.py               # run_sparql(endpoint, query) -> list[dict]
      topics.py               # Topic, load_registered_topics(...)
      client.py               # KGClient.import_paper(record, topic_qids) -> str
      pagedata.py             # PaperEntry, TopicPageData, fetch_topic_page_data(...)
    wiki/
      __init__.py
      page_builder.py         # build_topic_page(data), build_index_page(list)
      publisher.py            # WikiPublisher.login(), .edit(title, text, summary)
    pipeline.py               # harvest_step(config, state, ...), generate_pages_step(config, ...)
    __main__.py               # CLI orchestrator (--dry-run)
  tests/
    conftest.py
    test_config.py
    test_state.py
    test_arxiv_oai.py
    test_topics.py
    test_topic_classifier.py
    test_kg_client.py
    test_pagedata.py
    test_page_builder.py
    test_publisher.py
    test_pipeline.py
    fixtures/oai_listrecords.xml
```

---

## Task 1: Project scaffold + config

**Files:**
- Create: `topic-overviews/pyproject.toml`
- Create: `topic-overviews/.env.example`
- Create: `topic-overviews/README.md`
- Create: `topic-overviews/topic_overviews/__init__.py`
- Create: `topic-overviews/topic_overviews/config.py`
- Create: `topic-overviews/tests/__init__.py` (empty), `topic-overviews/tests/test_config.py`

**Interfaces:**
- Produces: `Config` (frozen dataclass) and `load_config(env: Mapping[str, str] = os.environ) -> Config`.
  Fields: `arxiv_set: str`, `model: str`, `relevance_threshold: float`, `state_path: str`, `overview_topic_qid: str`, `anthropic_api_key: str`, `mediawiki_api_url: str`, `mediawiki_bot_user: str`, `mediawiki_bot_password: str`, `wikibase_url: str`, `sparql_endpoint_url: str`, `dry_run: bool`.

- [ ] **Step 1: Initialize the repo and package dirs**

```bash
cd /home/tim/mardi/topic-overviews
git init
mkdir -p topic_overviews/harvest topic_overviews/llm topic_overviews/kg topic_overviews/wiki tests/fixtures
touch topic_overviews/__init__.py topic_overviews/harvest/__init__.py topic_overviews/llm/__init__.py topic_overviews/kg/__init__.py topic_overviews/wiki/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "topic-overviews"
version = "0.1.0"
description = "Automated thematic research overviews for the MaRDI Knowledge Graph"
requires-python = ">=3.13"
dependencies = [
    "requests>=2.31",
    "anthropic>=0.40",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["topic_overviews*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `.env.example` and `README.md`**

`.env.example`:
```bash
# Operational
TOPIC_OVERVIEWS_ARXIV_SET=math
TOPIC_OVERVIEWS_MODEL=claude-haiku-4-5
TOPIC_OVERVIEWS_RELEVANCE_THRESHOLD=0.0
TOPIC_OVERVIEWS_STATE_PATH=state.json
TOPIC_OVERVIEWS_OVERVIEW_TOPIC_QID=Q0   # QID of the 'overview topic' class (create once in the KG)
TOPIC_OVERVIEWS_DRY_RUN=false
# Secrets
ANTHROPIC_API_KEY=
MEDIAWIKI_API_URL=https://portal.mardi4nfdi.de/w/api.php
MEDIAWIKI_BOT_USER=
MEDIAWIKI_BOT_PASSWORD=
WIKIBASE_URL=https://portal.mardi4nfdi.de
SPARQL_ENDPOINT_URL=https://query.portal.mardi4nfdi.de/proxy/wdqs/bigdata/namespace/wdq/sparql
```

`README.md` (minimal):
```markdown
# topic-overviews

Automated thematic research overviews with MaRDI Knowledge-Graph integration.
Harvest arXiv math.* -> LLM-classify against KG-registered topics -> import to
the KG -> generate per-topic MediaWiki overview pages.

See `docs/superpowers/specs/` and `docs/superpowers/plans/`.

Run:
    python -m topic_overviews --dry-run
```

(Note: the `SPARQL_ENDPOINT_URL` and `MEDIAWIKI_API_URL` shown are best-guess defaults — confirm the live endpoints against the MaRDI portal during Task 5 / Task 9.)

- [ ] **Step 4: Write the failing test**

`tests/test_config.py`:
```python
from topic_overviews.config import load_config


def test_load_config_reads_env_and_defaults():
    env = {
        "ANTHROPIC_API_KEY": "sk-test",
        "TOPIC_OVERVIEWS_OVERVIEW_TOPIC_QID": "Q123",
        "TOPIC_OVERVIEWS_DRY_RUN": "true",
    }
    cfg = load_config(env)
    assert cfg.anthropic_api_key == "sk-test"
    assert cfg.overview_topic_qid == "Q123"
    assert cfg.dry_run is True
    assert cfg.model == "claude-haiku-4-5"        # default
    assert cfg.arxiv_set == "math"                # default
    assert cfg.relevance_threshold == 0.0         # default, coerced to float


def test_dry_run_defaults_false():
    cfg = load_config({})
    assert cfg.dry_run is False
```

- [ ] **Step 5: Run test to verify it fails**

Run: `cd /home/tim/mardi/topic-overviews && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.config'`

- [ ] **Step 6: Implement `config.py`**

```python
"""Environment-variable-first configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Config:
    arxiv_set: str
    model: str
    relevance_threshold: float
    state_path: str
    overview_topic_qid: str
    anthropic_api_key: str
    mediawiki_api_url: str
    mediawiki_bot_user: str
    mediawiki_bot_password: str
    wikibase_url: str
    sparql_endpoint_url: str
    dry_run: bool


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_config(env: Mapping[str, str] = os.environ) -> Config:
    return Config(
        arxiv_set=env.get("TOPIC_OVERVIEWS_ARXIV_SET", "math"),
        model=env.get("TOPIC_OVERVIEWS_MODEL", "claude-haiku-4-5"),
        relevance_threshold=float(env.get("TOPIC_OVERVIEWS_RELEVANCE_THRESHOLD", "0.0")),
        state_path=env.get("TOPIC_OVERVIEWS_STATE_PATH", "state.json"),
        overview_topic_qid=env.get("TOPIC_OVERVIEWS_OVERVIEW_TOPIC_QID", "Q0"),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        mediawiki_api_url=env.get("MEDIAWIKI_API_URL", ""),
        mediawiki_bot_user=env.get("MEDIAWIKI_BOT_USER", ""),
        mediawiki_bot_password=env.get("MEDIAWIKI_BOT_PASSWORD", ""),
        wikibase_url=env.get("WIKIBASE_URL", ""),
        sparql_endpoint_url=env.get("SPARQL_ENDPOINT_URL", ""),
        dry_run=_flag(env.get("TOPIC_OVERVIEWS_DRY_RUN", "false")),
    )
```

- [ ] **Step 7: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold topic-overviews package and env-first config"
```

---

## Task 2: Harvest state persistence

**Files:**
- Create: `topic_overviews/state.py`
- Create: `tests/test_state.py`

**Interfaces:**
- Produces: `State` dataclass (`last_harvest: str | None`, `seen_ids: set[str]`),
  `load_state(path: str) -> State`, `save_state(path: str, state: State) -> None`.
  `load_state` on a missing file returns a fresh empty `State`. `save_state` writes atomically.

- [ ] **Step 1: Write the failing test**

`tests/test_state.py`:
```python
from topic_overviews.state import State, load_state, save_state


def test_load_missing_returns_empty(tmp_path):
    st = load_state(str(tmp_path / "nope.json"))
    assert st.last_harvest is None
    assert st.seen_ids == set()


def test_save_then_load_roundtrip(tmp_path):
    path = str(tmp_path / "state.json")
    save_state(path, State(last_harvest="2026-06-19", seen_ids={"2401.00001", "2401.00002"}))
    st = load_state(path)
    assert st.last_harvest == "2026-06-19"
    assert st.seen_ids == {"2401.00001", "2401.00002"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.state'`

- [ ] **Step 3: Implement `state.py`**

```python
"""Persistent harvest cursor + de-duplication of seen arXiv IDs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class State:
    last_harvest: str | None = None      # ISO date "YYYY-MM-DD"
    seen_ids: set[str] = field(default_factory=set)


def load_state(path: str) -> State:
    if not os.path.exists(path):
        return State()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return State(
        last_harvest=data.get("last_harvest"),
        seen_ids=set(data.get("seen_ids", [])),
    )


def save_state(path: str, state: State) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"last_harvest": state.last_harvest, "seen_ids": sorted(state.seen_ids)},
            f,
        )
    os.replace(tmp, path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_state.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add harvest state persistence"
```

---

## Task 3: arXiv OAI-PMH harvester

**Files:**
- Create: `topic_overviews/harvest/arxiv_oai.py`
- Create: `tests/fixtures/oai_listrecords.xml`
- Create: `tests/test_arxiv_oai.py`

**Interfaces:**
- Produces:
  - `PaperRecord` dataclass: `arxiv_id: str`, `title: str`, `abstract: str`, `authors: list[str]`, `categories: list[str]`, `published: str` (YYYY-MM-DD), `doi: str | None`.
  - `parse_oai_response(xml: str) -> tuple[list[PaperRecord], str | None]` — returns records and the next resumption token (or None). Skips deleted records (no `arXiv` metadata block).
  - `fetch_records(from_date: str | None, set_spec: str = "math", session=None, sleep=time.sleep) -> Iterator[PaperRecord]` — follows resumption tokens, polite delay between pages.

- [ ] **Step 1: Create the fixture `tests/fixtures/oai_listrecords.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    <record>
      <header><identifier>oai:arXiv.org:2401.00001</identifier></header>
      <metadata>
        <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
          <id>2401.00001</id>
          <created>2024-01-02</created>
          <title>A New Bound for Online Caching</title>
          <abstract>We prove a tighter competitive ratio for caching.</abstract>
          <categories>math.OC cs.DS</categories>
          <doi>10.1000/xyz</doi>
          <authors>
            <author><keyname>Doe</keyname><forenames>Jane</forenames></author>
            <author><keyname>Smith</keyname><forenames>John</forenames></author>
          </authors>
        </arXiv>
      </metadata>
    </record>
    <record>
      <header status="deleted"><identifier>oai:arXiv.org:2401.00002</identifier></header>
    </record>
    <resumptionToken>TOKEN123</resumptionToken>
  </ListRecords>
</OAI-PMH>
```

- [ ] **Step 2: Write the failing test**

`tests/test_arxiv_oai.py`:
```python
from pathlib import Path

from topic_overviews.harvest.arxiv_oai import PaperRecord, parse_oai_response, fetch_records

FIXTURE = (Path(__file__).parent / "fixtures" / "oai_listrecords.xml").read_text()


def test_parse_extracts_record_and_token():
    records, token = parse_oai_response(FIXTURE)
    assert token == "TOKEN123"
    assert len(records) == 1                      # deleted record skipped
    r = records[0]
    assert r == PaperRecord(
        arxiv_id="2401.00001",
        title="A New Bound for Online Caching",
        abstract="We prove a tighter competitive ratio for caching.",
        authors=["Jane Doe", "John Smith"],
        categories=["math.OC", "cs.DS"],
        published="2024-01-02",
        doi="10.1000/xyz",
    )


def test_fetch_records_follows_resumption_token():
    page1 = FIXTURE
    page2 = FIXTURE.replace("TOKEN123", "").replace("2401.00001", "2402.00009")

    class FakeResp:
        def __init__(self, text): self.text = text; self.status_code = 200
        def raise_for_status(self): pass

    calls = []

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            calls.append(params)
            return FakeResp(page1 if len(calls) == 1 else page2)

    ids = [r.arxiv_id for r in fetch_records(None, session=FakeSession(), sleep=lambda s: None)]
    assert ids == ["2401.00001", "2402.00009"]
    assert calls[0]["from"] is None or "from" not in calls[0]
    assert calls[1]["resumptionToken"] == "TOKEN123"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_arxiv_oai.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.harvest.arxiv_oai'`

- [ ] **Step 4: Implement `harvest/arxiv_oai.py`**

```python
"""Incremental arXiv harvest via OAI-PMH (ListRecords, arXiv metadata format)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator
from xml.etree import ElementTree as ET

import requests

OAI_URL = "http://export.arxiv.org/oai2"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    doi: str | None = None


def _text(el, path: str) -> str | None:
    found = el.find(path, NS)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split())


def parse_oai_response(xml: str) -> tuple[list[PaperRecord], str | None]:
    root = ET.fromstring(xml)
    records: list[PaperRecord] = []
    for rec in root.findall(".//oai:record", NS):
        meta = rec.find(".//arxiv:arXiv", NS)
        if meta is None:
            continue  # deleted / no metadata
        authors = []
        for a in meta.findall("arxiv:authors/arxiv:author", NS):
            name = " ".join(
                p for p in [_text(a, "arxiv:forenames"), _text(a, "arxiv:keyname")] if p
            )
            if name:
                authors.append(name)
        records.append(
            PaperRecord(
                arxiv_id=_text(meta, "arxiv:id") or "",
                title=_text(meta, "arxiv:title") or "",
                abstract=_text(meta, "arxiv:abstract") or "",
                authors=authors,
                categories=(_text(meta, "arxiv:categories") or "").split(),
                published=_text(meta, "arxiv:created") or "",
                doi=_text(meta, "arxiv:doi"),
            )
        )
    token_el = root.find(".//oai:resumptionToken", NS)
    token = token_el.text.strip() if token_el is not None and token_el.text and token_el.text.strip() else None
    return records, token


def fetch_records(
    from_date: str | None,
    set_spec: str = "math",
    session=None,
    sleep=time.sleep,
) -> Iterator[PaperRecord]:
    session = session or requests.Session()
    params: dict = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_spec}
    if from_date:
        params["from"] = from_date
    while True:
        resp = session.get(OAI_URL, params=params, timeout=60)
        resp.raise_for_status()
        records, token = parse_oai_response(resp.text)
        yield from records
        if not token:
            return
        params = {"verb": "ListRecords", "resumptionToken": token}
        sleep(3)  # arXiv OAI politeness
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_arxiv_oai.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add arXiv OAI-PMH harvester"
```

---

## Task 4: KG constants + SPARQL helper

**Files:**
- Create: `topic_overviews/kg/model.py`
- Create: `topic_overviews/kg/sparql.py`
- Create: `tests/test_sparql.py`

**Interfaces:**
- Produces (`model.py`): module constants `P_INSTANCE_OF="P31"`, `P_ARXIV_ID="P21"`, `P_ARXIV_CLASSIFICATION="P22"`, `P_AUTHOR="P16"`, `P_AUTHOR_NAME_STRING="P43"`, `P_PUBLICATION_DATE="P28"`, `P_TITLE="P159"`, `P_DOI="P27"`, `P_MAIN_SUBJECT="P30"`, `P_SUBCLASS_OF="P36"`, `P_AFFILIATION="P55"`, `Q_SCHOLARLY_ARTICLE="Q56887"`, `Q_PREPRINT="Q159099"`; helper `qid_from_uri(uri: str) -> str`.
- Produces (`sparql.py`): `run_sparql(endpoint: str, query: str, session=None) -> list[dict[str, str]]` — returns each binding row as a flat `{var: value}` dict.

- [ ] **Step 1: Write the failing test**

`tests/test_sparql.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sparql.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.kg.model'`

- [ ] **Step 3: Implement `kg/model.py`**

```python
"""Resolved MaRDI Wikibase property/class IDs. Verified 2026-06-20."""
from __future__ import annotations

# Properties
P_INSTANCE_OF = "P31"
P_ARXIV_ID = "P21"
P_ARXIV_CLASSIFICATION = "P22"
P_AUTHOR = "P16"
P_AUTHOR_NAME_STRING = "P43"
P_PUBLICATION_DATE = "P28"
P_TITLE = "P159"
P_DOI = "P27"
P_MAIN_SUBJECT = "P30"
P_SUBCLASS_OF = "P36"
P_AFFILIATION = "P55"

# Classes
Q_SCHOLARLY_ARTICLE = "Q56887"
Q_PREPRINT = "Q159099"


def qid_from_uri(uri: str) -> str:
    """Extract the trailing Q-id from an entity URI (or pass through a bare QID)."""
    return uri.rstrip("/").rsplit("/", 1)[-1]
```

- [ ] **Step 4: Implement `kg/sparql.py`**

```python
"""Thin SPARQL JSON-results helper."""
from __future__ import annotations

import requests


def run_sparql(endpoint: str, query: str, session=None) -> list[dict[str, str]]:
    session = session or requests.Session()
    resp = session.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return [
        {var: cell["value"] for var, cell in row.items()}
        for row in data["results"]["bindings"]
    ]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_sparql.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add KG id constants and SPARQL helper"
```

---

## Task 5: Read registered topics from the KG

**Files:**
- Create: `topic_overviews/kg/topics.py`
- Create: `tests/test_topics.py`

**Interfaces:**
- Consumes: `kg.sparql.run_sparql`, `kg.model.qid_from_uri`.
- Produces:
  - `Topic` dataclass: `qid: str`, `label: str`, `description: str`.
  - `load_registered_topics(sparql_endpoint: str, overview_topic_qid: str, run=run_sparql) -> list[Topic]` — queries items that are `instance of` (P31) the overview-topic class; missing descriptions become `""`.

- [ ] **Step 1: Write the failing test**

`tests/test_topics.py`:
```python
from topic_overviews.kg.topics import Topic, load_registered_topics


def test_load_registered_topics_parses_rows():
    captured = {}

    def fake_run(endpoint, query):
        captured["endpoint"] = endpoint
        captured["query"] = query
        return [
            {"topic": "https://x/entity/Q10", "label": "Optimization",
             "desc": "Mathematical optimization."},
            {"topic": "https://x/entity/Q11", "label": "Numerical Analysis"},
        ]

    topics = load_registered_topics("http://ep", "Q5", run=fake_run)
    assert topics == [
        Topic(qid="Q10", label="Optimization", description="Mathematical optimization."),
        Topic(qid="Q11", label="Numerical Analysis", description=""),
    ]
    assert "Q5" in captured["query"]            # filters on the overview-topic class
    assert captured["endpoint"] == "http://ep"


def test_load_registered_topics_empty():
    assert load_registered_topics("http://ep", "Q5", run=lambda e, q: []) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.kg.topics'`

- [ ] **Step 3: Implement `kg/topics.py`**

```python
"""Read the topic registry from the KG (items that are instance-of overview-topic)."""
from __future__ import annotations

from dataclasses import dataclass

from .model import P_INSTANCE_OF, qid_from_uri
from .sparql import run_sparql


@dataclass
class Topic:
    qid: str
    label: str
    description: str


_QUERY = """SELECT ?topic ?label ?desc WHERE {{
  ?topic wdt:{p_inst} wd:{cls} .
  ?topic rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL {{ ?topic schema:description ?desc . FILTER(LANG(?desc) = "en") }}
}}"""


def load_registered_topics(sparql_endpoint: str, overview_topic_qid: str, run=run_sparql) -> list[Topic]:
    query = _QUERY.format(p_inst=P_INSTANCE_OF, cls=overview_topic_qid)
    rows = run(sparql_endpoint, query)
    return [
        Topic(
            qid=qid_from_uri(row["topic"]),
            label=row["label"],
            description=row.get("desc", ""),
        )
        for row in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topics.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: read registered overview topics from the KG"
```

> **Live check (not a test):** the `wdt:`/`wd:`/`rdfs:`/`schema:` prefixes must be recognized by the MaRDI SPARQL endpoint. If the endpoint does not predefine them, prepend explicit `PREFIX` lines built from `config.wikibase_url`. Verify `_QUERY` against the live endpoint before first production run.

---

## Task 6: LLM topic classifier

**Files:**
- Create: `topic_overviews/llm/topic_classifier.py`
- Create: `tests/test_topic_classifier.py`

**Interfaces:**
- Consumes: `harvest.arxiv_oai.PaperRecord`, `kg.topics.Topic`.
- Produces: `classify_paper(paper: PaperRecord, topics: list[Topic], *, model: str, api_key: str, client=None) -> list[str]` — returns the matched topic QIDs (subset of the given topics' QIDs); `[]` when none match. Validates JSON, drops unknown QIDs, retries once on malformed output.

- [ ] **Step 1: Write the failing test**

`tests/test_topic_classifier.py`:
```python
from types import SimpleNamespace

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.kg.topics import Topic
from topic_overviews.llm.topic_classifier import classify_paper

PAPER = PaperRecord(
    arxiv_id="2401.00001", title="Online Caching with Predictions",
    abstract="We study caching.", authors=["Jane Doe"],
    categories=["cs.DS"], published="2024-01-02", doi=None,
)
TOPICS = [
    Topic(qid="Q10", label="Optimization", description="..."),
    Topic(qid="Q11", label="Online Algorithms", description="..."),
]


class FakeClient:
    def __init__(self, texts):
        self._texts = list(texts)
        self.prompts = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, messages):
        self.prompts.append(messages[0]["content"])
        text = self._texts.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


def test_returns_known_qids_and_drops_unknown():
    client = FakeClient(['{"topics": ["Q11", "Q999"]}'])
    result = classify_paper(PAPER, TOPICS, model="claude-haiku-4-5", api_key="x", client=client)
    assert result == ["Q11"]
    # prompt includes the topics and the paper text
    assert "Q11" in client.prompts[0] and "Online Caching with Predictions" in client.prompts[0]


def test_empty_match_returns_empty_list():
    client = FakeClient(['{"topics": []}'])
    assert classify_paper(PAPER, TOPICS, model="m", api_key="x", client=client) == []


def test_retries_once_on_bad_json_then_succeeds():
    client = FakeClient(["not json", '{"topics": ["Q10"]}'])
    assert classify_paper(PAPER, TOPICS, model="m", api_key="x", client=client) == ["Q10"]


def test_gives_up_after_retry():
    client = FakeClient(["nope", "still nope"])
    assert classify_paper(PAPER, TOPICS, model="m", api_key="x", client=client) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_topic_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.llm.topic_classifier'`

- [ ] **Step 3: Implement `llm/topic_classifier.py`**

```python
"""Classify a paper against the KG-registered topics using Claude."""
from __future__ import annotations

import json
import re

from ..harvest.arxiv_oai import PaperRecord
from ..kg.topics import Topic

_SYSTEM = (
    "You classify a mathematics paper into a fixed list of research topics. "
    "Return ONLY a JSON object of the form {\"topics\": [\"Q123\", ...]} listing the "
    "QIDs of the topics the paper clearly belongs to. Use an empty list if none fit. "
    "Never invent QIDs that are not in the provided list."
)


def _build_prompt(paper: PaperRecord, topics: list[Topic]) -> str:
    topic_lines = "\n".join(f"- {t.qid}: {t.label} — {t.description}" for t in topics)
    return (
        f"{_SYSTEM}\n\nTOPICS:\n{topic_lines}\n\n"
        f"PAPER TITLE: {paper.title}\n"
        f"ABSTRACT: {paper.abstract}\n\n"
        'Respond with JSON only, e.g. {"topics": ["Q11"]}.'
    )


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def classify_paper(
    paper: PaperRecord,
    topics: list[Topic],
    *,
    model: str,
    api_key: str,
    client=None,
) -> list[str]:
    if client is None:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

    valid = {t.qid for t in topics}
    prompt = _build_prompt(paper, topics)

    for _ in range(2):
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        try:
            data = _extract_json(text)
            return [q for q in data.get("topics", []) if q in valid]
        except (ValueError, json.JSONDecodeError):
            continue
    return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_topic_classifier.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add LLM topic classifier"
```

---

## Task 7: KG writer (import paper + topic links)

**Files:**
- Create: `topic_overviews/kg/client.py`
- Create: `tests/test_kg_client.py`

**Interfaces:**
- Consumes: `harvest.arxiv_oai.PaperRecord`, constants from `kg.model`.
- Produces:
  - `KGClient(mc)` — wraps a `mardiclient.MardiClient`-like object (`mc.search_entity_by_value(prop, value) -> list[str]`, `mc.item.new() -> item`, `mc.item.get(entity_id=qid) -> item`; item has `.labels.set(lang, value)`, `.add_claim(prop, value=..., action=...)`, `.write() -> item` where the written item has `.id`).
  - `KGClient.import_paper(record: PaperRecord, topic_qids: list[str]) -> str` — idempotent upsert by arXiv ID; sets paper statements; adds a `main subject` (P30) claim per matched topic; returns the paper QID.
  - `make_kg_client(config) -> KGClient` — builds a real `MardiClient` from config (imported lazily).
  - helper `to_wbi_time(date: str) -> str` — `"YYYY-MM-DD"` → `"+YYYY-MM-DDT00:00:00Z"`.

- [ ] **Step 1: Write the failing test**

`tests/test_kg_client.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_kg_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.kg.client'`

- [ ] **Step 3: Implement `kg/client.py`**

```python
"""Write papers and topic links into the MaRDI Wikibase via mardiclient."""
from __future__ import annotations

from ..harvest.arxiv_oai import PaperRecord
from . import model as M


def to_wbi_time(date: str) -> str:
    return f"+{date}T00:00:00Z"


class KGClient:
    def __init__(self, mc):
        self.mc = mc

    def import_paper(self, record: PaperRecord, topic_qids: list[str]) -> str:
        existing = self.mc.search_entity_by_value(f"wdt:{M.P_ARXIV_ID}", record.arxiv_id)
        if existing:
            item = self.mc.item.get(entity_id=existing[0])
        else:
            item = self.mc.item.new()
            item.labels.set("en", record.title[:250])

        item.add_claim(f"wdt:{M.P_INSTANCE_OF}", value=f"wd:{M.Q_SCHOLARLY_ARTICLE}")
        item.add_claim(f"wdt:{M.P_ARXIV_ID}", value=record.arxiv_id)
        if record.doi:
            item.add_claim(f"wdt:{M.P_DOI}", value=record.doi)
        item.add_claim(f"wdt:{M.P_TITLE}", value=record.title)
        if record.published:
            item.add_claim(f"wdt:{M.P_PUBLICATION_DATE}", value=to_wbi_time(record.published))
        for cat in record.categories:
            item.add_claim(f"wdt:{M.P_ARXIV_CLASSIFICATION}", value=cat)
        for name in record.authors:
            item.add_claim(f"wdt:{M.P_AUTHOR_NAME_STRING}", value=name)
        for tq in topic_qids:
            item.add_claim(f"wdt:{M.P_MAIN_SUBJECT}", value=f"wd:{tq}")

        return item.write().id


def make_kg_client(config) -> KGClient:
    from mardiclient import MardiClient

    mc = MardiClient(
        user=config.mediawiki_bot_user,
        password=config.mediawiki_bot_password,
        login_with_bot=True,
        mediawiki_api_url=config.mediawiki_api_url,
        sparql_endpoint_url=config.sparql_endpoint_url,
        wikibase_url=config.wikibase_url,
    )
    return KGClient(mc)
```

Both branches (new item vs. fetched existing item) build claims on the same
`item`, then write once and return its QID. The existing-item branch deliberately
skips `labels.set` so an already-titled item is not relabelled.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_kg_client.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add KG writer for papers and topic links"
```

> **Live check (not a test):** confirm `add_claim` accepts the date value format for P28 and `wd:`-prefixed item values for P31/P30 against the live Wikibase (the mardiclient `get_claim` resolves datatypes from the property). Author modeling is intentionally minimal (name string P43); richer P16 author-item resolution is a future extension.

---

## Task 8: Topic page data (SPARQL read for pages)

**Files:**
- Create: `topic_overviews/kg/pagedata.py`
- Create: `tests/test_pagedata.py`

**Interfaces:**
- Consumes: `kg.sparql.run_sparql`, `kg.model` (`P_MAIN_SUBJECT`, `P_TITLE`, `P_PUBLICATION_DATE`, `P_ARXIV_ID`, `P_AUTHOR_NAME_STRING`), `kg.topics.Topic`.
- Produces:
  - `PaperEntry` dataclass: `title: str`, `authors: list[str]`, `year: str`, `arxiv_id: str`.
  - `TopicPageData` dataclass: `qid: str`, `label: str`, `description: str`, `papers: list[PaperEntry]`.
  - `fetch_topic_page_data(sparql_endpoint: str, topic: Topic, run=run_sparql) -> TopicPageData` — one row per paper; authors arrive as a `"; "`-joined string (SPARQL GROUP_CONCAT) and are split back into a list; `year` is the first 4 chars of the publication date.

- [ ] **Step 1: Write the failing test**

`tests/test_pagedata.py`:
```python
from topic_overviews.kg.topics import Topic
from topic_overviews.kg.pagedata import PaperEntry, TopicPageData, fetch_topic_page_data

TOPIC = Topic(qid="Q11", label="Online Algorithms", description="Algorithms online.")


def test_fetch_topic_page_data_builds_entries():
    captured = {}

    def fake_run(endpoint, query):
        captured["query"] = query
        return [
            {"title": "Online Caching", "year": "2024-01-02",
             "arxiv": "2401.00001", "authors": "Jane Doe; John Smith"},
            {"title": "Ski Rental Revisited", "year": "2023-11-09",
             "arxiv": "2311.00050", "authors": "Ada Lovelace"},
        ]

    data = fetch_topic_page_data("http://ep", TOPIC, run=fake_run)
    assert data == TopicPageData(
        qid="Q11", label="Online Algorithms", description="Algorithms online.",
        papers=[
            PaperEntry(title="Online Caching", authors=["Jane Doe", "John Smith"],
                       year="2024", arxiv_id="2401.00001"),
            PaperEntry(title="Ski Rental Revisited", authors=["Ada Lovelace"],
                       year="2023", arxiv_id="2311.00050"),
        ],
    )
    assert "Q11" in captured["query"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pagedata.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.kg.pagedata'`

- [ ] **Step 3: Implement `kg/pagedata.py`**

```python
"""Read per-topic paper lists from the KG for page generation."""
from __future__ import annotations

from dataclasses import dataclass

from . import model as M
from .sparql import run_sparql
from .topics import Topic


@dataclass
class PaperEntry:
    title: str
    authors: list[str]
    year: str
    arxiv_id: str


@dataclass
class TopicPageData:
    qid: str
    label: str
    description: str
    papers: list[PaperEntry]


_QUERY = """SELECT ?title ?year ?arxiv (GROUP_CONCAT(?author; SEPARATOR="; ") AS ?authors) WHERE {{
  ?paper wdt:{p_subject} wd:{topic} .
  ?paper wdt:{p_title} ?title .
  OPTIONAL {{ ?paper wdt:{p_date} ?year }}
  OPTIONAL {{ ?paper wdt:{p_arxiv} ?arxiv }}
  OPTIONAL {{ ?paper wdt:{p_author} ?author }}
}} GROUP BY ?title ?year ?arxiv ORDER BY DESC(?year)"""


def fetch_topic_page_data(sparql_endpoint: str, topic: Topic, run=run_sparql) -> TopicPageData:
    query = _QUERY.format(
        p_subject=M.P_MAIN_SUBJECT, topic=topic.qid, p_title=M.P_TITLE,
        p_date=M.P_PUBLICATION_DATE, p_arxiv=M.P_ARXIV_ID, p_author=M.P_AUTHOR_NAME_STRING,
    )
    rows = run(sparql_endpoint, query)
    papers = [
        PaperEntry(
            title=row.get("title", ""),
            authors=[a for a in row.get("authors", "").split("; ") if a],
            year=(row.get("year", "") or "")[:4],
            arxiv_id=row.get("arxiv", ""),
        )
        for row in rows
    ]
    return TopicPageData(qid=topic.qid, label=topic.label, description=topic.description, papers=papers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pagedata.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: read per-topic paper lists from the KG"
```

---

## Task 9: Wikitext page builder

**Files:**
- Create: `topic_overviews/wiki/page_builder.py`
- Create: `tests/test_page_builder.py`

**Interfaces:**
- Consumes: `kg.pagedata.PaperEntry`, `kg.pagedata.TopicPageData`.
- Produces:
  - `build_topic_page(data: TopicPageData) -> str` — wikitext: heading, description, sortable wikitable of papers (Title / Authors / Year / arXiv link).
  - `build_index_page(topics: list[TopicPageData]) -> str` — wikitext index linking each topic page and showing its paper count.
  - module constant `TOPIC_PAGE_PREFIX = "Topic:"` and `INDEX_PAGE_TITLE = "Topic overview"`.

- [ ] **Step 1: Write the failing test**

`tests/test_page_builder.py`:
```python
from topic_overviews.kg.pagedata import PaperEntry, TopicPageData
from topic_overviews.wiki.page_builder import (
    build_topic_page, build_index_page, TOPIC_PAGE_PREFIX, INDEX_PAGE_TITLE,
)

DATA = TopicPageData(
    qid="Q11", label="Online Algorithms", description="Algorithms that act online.",
    papers=[
        PaperEntry(title="Online Caching", authors=["Jane Doe", "John Smith"],
                   year="2024", arxiv_id="2401.00001"),
    ],
)


def test_build_topic_page_exact():
    assert build_topic_page(DATA) == (
        "= Online Algorithms =\n"
        "\n"
        "Algorithms that act online.\n"
        "\n"
        '{| class="wikitable sortable"\n'
        "! Title !! Authors !! Year !! arXiv\n"
        "|-\n"
        "| Online Caching || Jane Doe; John Smith || 2024 || "
        "[https://arxiv.org/abs/2401.00001 2401.00001]\n"
        "|}\n"
    )


def test_build_index_page_exact():
    assert build_index_page([DATA]) == (
        "= Topic overview =\n"
        "\n"
        "* [[Topic:Online Algorithms|Online Algorithms]] (1 papers)\n"
    )


def test_constants():
    assert TOPIC_PAGE_PREFIX == "Topic:"
    assert INDEX_PAGE_TITLE == "Topic overview"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_page_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.wiki.page_builder'`

- [ ] **Step 3: Implement `wiki/page_builder.py`**

```python
"""Render topic overview pages and the index as MediaWiki wikitext (pure)."""
from __future__ import annotations

from ..kg.pagedata import TopicPageData

TOPIC_PAGE_PREFIX = "Topic:"
INDEX_PAGE_TITLE = "Topic overview"


def build_topic_page(data: TopicPageData) -> str:
    lines = [
        f"= {data.label} =",
        "",
        data.description,
        "",
        '{| class="wikitable sortable"',
        "! Title !! Authors !! Year !! arXiv",
    ]
    for p in data.papers:
        link = f"[https://arxiv.org/abs/{p.arxiv_id} {p.arxiv_id}]" if p.arxiv_id else ""
        lines.append("|-")
        lines.append(f"| {p.title} || {'; '.join(p.authors)} || {p.year} || {link}")
    lines.append("|}")
    return "\n".join(lines) + "\n"


def build_index_page(topics: list[TopicPageData]) -> str:
    lines = [f"= {INDEX_PAGE_TITLE} =", ""]
    for t in topics:
        lines.append(f"* [[{TOPIC_PAGE_PREFIX}{t.label}|{t.label}]] ({len(t.papers)} papers)")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_page_builder.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: render topic overview wikitext pages"
```

---

## Task 10: MediaWiki publisher

**Files:**
- Create: `topic_overviews/wiki/publisher.py`
- Create: `tests/test_publisher.py`

**Interfaces:**
- Produces:
  - `WikiPublisher(api_url, user, password, session=None)` with:
    - `login() -> None` — fetches a login token then logs in via the MediaWiki `clientlogin`/`login` action.
    - `edit(title: str, text: str, summary: str) -> None` — fetches a CSRF token then POSTs `action=edit`.
  - Uses a `requests.Session`-like object (`.get(url, params=...)`, `.post(url, data=...)` returning objects with `.json()` and `.raise_for_status()`).

- [ ] **Step 1: Write the failing test**

`tests/test_publisher.py`:
```python
from topic_overviews.wiki.publisher import WikiPublisher


class FakeResp:
    def __init__(self, payload): self._payload = payload
    def raise_for_status(self): pass
    def json(self): return self._payload


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", params))
        return FakeResp(self._responses.pop(0))

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", data))
        return FakeResp(self._responses.pop(0))


def test_login_then_edit_posts_text_with_token():
    responses = [
        {"query": {"tokens": {"logintoken": "LT"}}},          # GET login token
        {"login": {"result": "Success"}},                      # POST login
        {"query": {"tokens": {"csrftoken": "CT"}}},            # GET csrf token
        {"edit": {"result": "Success"}},                       # POST edit
    ]
    session = FakeSession(responses)
    pub = WikiPublisher("http://api", "bot", "pw", session=session)
    pub.login()
    pub.edit("Topic:Online Algorithms", "= hi =", "update")

    post_calls = [c for c in session.calls if c[0] == "POST"]
    login_data = post_calls[0][1]
    edit_data = post_calls[1][1]
    assert login_data["lgtoken"] == "LT"
    assert edit_data["title"] == "Topic:Online Algorithms"
    assert edit_data["text"] == "= hi ="
    assert edit_data["token"] == "CT"
    assert edit_data["summary"] == "update"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_publisher.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.wiki.publisher'`

- [ ] **Step 3: Implement `wiki/publisher.py`**

```python
"""Publish wikitext pages to MediaWiki via the action API (bot login)."""
from __future__ import annotations

import requests


class WikiPublisher:
    def __init__(self, api_url: str, user: str, password: str, session=None):
        self.api_url = api_url
        self.user = user
        self.password = password
        self.session = session or requests.Session()

    def _get_token(self, kind: str) -> str:
        resp = self.session.get(
            self.api_url,
            params={"action": "query", "meta": "tokens", "type": kind, "format": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        key = "logintoken" if kind == "login" else "csrftoken"
        return resp.json()["query"]["tokens"][key]

    def login(self) -> None:
        token = self._get_token("login")
        resp = self.session.post(
            self.api_url,
            data={
                "action": "login",
                "lgname": self.user,
                "lgpassword": self.password,
                "lgtoken": token,
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        if resp.json().get("login", {}).get("result") != "Success":
            raise RuntimeError(f"MediaWiki login failed: {resp.json()}")

    def edit(self, title: str, text: str, summary: str) -> None:
        token = self._get_token("csrf")
        resp = self.session.post(
            self.api_url,
            data={
                "action": "edit",
                "title": title,
                "text": text,
                "summary": summary,
                "bot": "1",
                "token": token,
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        if resp.json().get("edit", {}).get("result") != "Success":
            raise RuntimeError(f"MediaWiki edit failed for {title}: {resp.json()}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_publisher.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add MediaWiki publisher"
```

> **Live check (not a test):** the portal may require `assert=bot` and a BotPassword (the `login` action with a bot password, or `clientlogin`). Confirm the exact login flow against `MEDIAWIKI_API_URL` and adjust `login()` if needed.

---

## Task 11: Pipeline steps + CLI orchestrator

**Files:**
- Create: `topic_overviews/pipeline.py`
- Create: `topic_overviews/__main__.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `config.Config`, `state.State`, `harvest.arxiv_oai.fetch_records`, `kg.topics.load_registered_topics`, `llm.topic_classifier.classify_paper`, `kg.client.KGClient`, `kg.pagedata.fetch_topic_page_data`, `wiki.page_builder.build_topic_page/build_index_page`, `wiki.publisher.WikiPublisher`.
- Produces:
  - `harvest_step(config, state, *, topics, kg, fetch=fetch_records, classify=classify_paper) -> int` — iterates new papers (skipping `state.seen_ids`), classifies, imports matched papers via `kg`, updates `state.seen_ids` and `state.last_harvest`; returns count imported. Honors `config.dry_run` (skips `kg.import_paper`).
  - `generate_pages_step(config, *, topics, publisher, fetch_page_data=fetch_topic_page_data) -> list[TopicPageData]` — builds + publishes one page per topic and the index. Honors `config.dry_run` (skips `publisher.edit`).
  - `__main__`: parse `--dry-run`, load config+state, build deps, run both steps, save state.

- [ ] **Step 1: Write the failing test**

`tests/test_pipeline.py`:
```python
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
    pipeline.harvest_step(cfg, state, topics=TOPICS, kg=kg,
                          fetch=lambda f, set_spec: iter([P1]),
                          classify=lambda *a, **k: ["Q11"])
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'topic_overviews.pipeline'`

- [ ] **Step 3: Implement `pipeline.py`**

```python
"""Sequential pipeline steps. Plain functions with injected deps (Prefect-ready)."""
from __future__ import annotations

import datetime

from .config import Config
from .state import State
from .harvest.arxiv_oai import fetch_records
from .kg.topics import Topic
from .kg.pagedata import TopicPageData, fetch_topic_page_data
from .llm.topic_classifier import classify_paper
from .wiki.page_builder import (
    build_topic_page, build_index_page, TOPIC_PAGE_PREFIX, INDEX_PAGE_TITLE,
)


def harvest_step(
    config: Config,
    state: State,
    *,
    topics: list[Topic],
    kg,
    fetch=fetch_records,
    classify=classify_paper,
) -> int:
    imported = 0
    for record in fetch(state.last_harvest, config.arxiv_set):
        if record.arxiv_id in state.seen_ids:
            continue
        state.seen_ids.add(record.arxiv_id)
        matched = classify(
            record, topics, model=config.model, api_key=config.anthropic_api_key
        )
        if not matched:
            continue
        if not config.dry_run:
            kg.import_paper(record, matched)
        imported += 1
    state.last_harvest = datetime.date.today().isoformat()
    return imported


def generate_pages_step(
    config: Config,
    *,
    topics: list[Topic],
    publisher,
    fetch_page_data=fetch_topic_page_data,
) -> list[TopicPageData]:
    page_data: list[TopicPageData] = []
    for topic in topics:
        data = fetch_page_data(config.sparql_endpoint_url, topic)
        page_data.append(data)
        if not config.dry_run:
            publisher.edit(
                f"{TOPIC_PAGE_PREFIX}{topic.label}", build_topic_page(data),
                "Update topic overview",
            )
    if not config.dry_run:
        publisher.edit(INDEX_PAGE_TITLE, build_index_page(page_data), "Update topic index")
    return page_data
```

- [ ] **Step 4: Run pipeline tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Implement `__main__.py`**

```python
"""CLI entry point: run the harvest + page-generation pipeline once."""
from __future__ import annotations

import argparse
import logging

from .config import load_config
from .state import load_state, save_state
from .kg.topics import load_registered_topics
from .kg.client import make_kg_client
from .wiki.publisher import WikiPublisher
from . import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="topic_overviews")
    parser.add_argument("--dry-run", action="store_true", help="harvest + classify but skip all writes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("topic_overviews")

    config = load_config()
    if args.dry_run:
        object.__setattr__(config, "dry_run", True)  # frozen dataclass override

    state = load_state(config.state_path)
    topics = load_registered_topics(config.sparql_endpoint_url, config.overview_topic_qid)
    log.info("Loaded %d registered topics", len(topics))

    kg = None if config.dry_run else make_kg_client(config)
    imported = pipeline.harvest_step(config, state, topics=topics, kg=kg)
    log.info("Imported %d papers", imported)
    save_state(config.state_path, state)

    publisher = None if config.dry_run else _make_publisher(config)
    pages = pipeline.generate_pages_step(config, topics=topics, publisher=publisher)
    log.info("Generated %d topic pages", len(pages))


def _make_publisher(config) -> WikiPublisher:
    pub = WikiPublisher(config.mediawiki_api_url, config.mediawiki_bot_user, config.mediawiki_bot_password)
    pub.login()
    return pub


if __name__ == "__main__":
    main()
```

> Note: in `--dry-run`, `kg` and `publisher` are `None`; the pipeline steps never call them because `config.dry_run` short-circuits the write branches. The `object.__setattr__` line is needed because `Config` is a frozen dataclass.

- [ ] **Step 6: Run the full test suite**

Run: `python -m pytest -v`
Expected: PASS (all tests across all modules)

- [ ] **Step 7: Smoke-test the dry run wiring (no network writes)**

Run:
```bash
TOPIC_OVERVIEWS_DRY_RUN=true python -m topic_overviews --dry-run
```
Expected: it attempts to load topics from the SPARQL endpoint and logs progress. If `SPARQL_ENDPOINT_URL` is unset/unreachable it will error on the topic query — that is acceptable for this wiring smoke test; the unit suite already covers logic. (Set a real endpoint to exercise end-to-end.)

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat: wire harvest + page-generation pipeline and CLI"
```

---

## Post-implementation: live integration checklist (not unit-tested)

These require live MaRDI credentials/endpoints and are validated manually after the unit suite is green:

- [ ] Create the **`overview topic` class** item once in the KG; set `TOPIC_OVERVIEWS_OVERVIEW_TOPIC_QID`.
- [ ] Register 1–2 **initial topic items** in the KG (instance-of the class, with label + description). Confirm `load_registered_topics` returns them.
- [ ] Confirm the **SPARQL prefix** assumptions (`wdt:`/`wd:`/`rdfs:`/`schema:`) and both queries against the live endpoint.
- [ ] Confirm **mardiclient** writes for P31/P30 (item values) and P28 (date) against a staging Wikibase; run one paper end-to-end.
- [ ] Confirm the **MediaWiki bot login + edit** flow against the portal; publish one topic page.
- [ ] Decide and document the topic **page namespace** (`Topic:` assumed) — adjust `TOPIC_PAGE_PREFIX` if the portal uses a different namespace.
- [ ] Run once for real, review a generated page, then schedule (daily) — later as a Prefect deployment.
