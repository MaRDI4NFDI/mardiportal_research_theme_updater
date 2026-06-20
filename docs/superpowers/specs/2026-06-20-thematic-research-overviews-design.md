# Automated Thematic Research Overviews with Knowledge-Graph Integration

**Date:** 2026-06-20
**Status:** Design approved, pending implementation plan

## Summary

A standalone Python pipeline that continuously scans new mathematics papers on
arXiv, uses an LLM to classify each paper against the set of **research topics
registered in the MaRDI Knowledge Graph**, imports the relevant papers into the
KG (Wikibase) with topic links, and generates one **MediaWiki overview page per
topic** on the portal.

Topics are **KG-managed**: each topic is a dedicated Wikibase item of a new
`overview topic` class. They are created and edited directly in the KG (wiki/
API), not in a config file. At startup the pipeline queries the KG for all
registered overview topics and works from that set, so new topics added in the
KG are picked up automatically on the next run.

The inspiration is [Algorithms with Predictions (ALPS)](https://algorithms-with-predictions.github.io/) —
a curated, label-organized overview of a research field. We take over the *idea*
(per-topic overview pages with browsable paper lists) but replace ALPS's manual
curation with an automated, LLM-driven pipeline wired into the MaRDI KG. We do
**not** copy the ALPS site design.

## Goals

- Continuously monitor new arXiv `math.*` publications.
- Use an LLM to **classify each paper against the topics registered in the KG**
  (items of the `overview topic` class). A paper may match one, several, or none.
- Import relevant publications into the MaRDI KG (Wikibase).
- Build KG links between topics, publications, authors, and institutions.
- Generate dynamic per-topic overview pages as MediaWiki pages on the portal.

## Non-goals

- No human review/approval gate — the LLM's decisions are written directly to
  the KG (fully automatic). A correction workflow is out of scope for v1.
- No Crossref/OpenAlex harvesting in v1 (arXiv `math.*` only). The harvester is
  structured so additional sources can be added later.
- No separate React/static frontend. Overview pages are MediaWiki wiki pages.
- Not part of `docker-importer`. This is a standalone project.

## Key decisions

| Decision | Choice |
|---|---|
| Project home | Standalone directory `topic-overviews/` in the workspace (push to a MaRDI4NFDI repo later) |
| Source | arXiv `math.*` via OAI-PMH, incremental harvest |
| Topic model | **KG-managed.** Dedicated `overview topic` class; each topic is a KG item created/edited directly in the KG. Pipeline reads the registered set at startup. LLM only classifies papers into this set (or "none") |
| Review gate | None — fully automatic writes to the KG |
| KG writes | Self-contained via `mardiclient` with our own slim arXiv→Wikibase mapping |
| LLM | Anthropic `claude-haiku-4-5` (configurable via env) |
| Overview pages | MediaWiki wiki pages generated from KG data |
| Orchestration | Plain Python now, structured for a trivial later port to a Prefect flow |
| Secrets/config | Environment-variable-first (Prefect surfaces Secret blocks as env vars) |

## Architecture

One workflow, two sequential steps, run by a thin plain-Python orchestrator:

```
arXiv math.* (OAI-PMH, new records since last run)
      │  title + abstract + categories + authors + dates
      ▼
[Step 1] Harvest + metadata generation
   • read registered topics from the KG (instance of 'overview topic')
   • state file -> only new arXiv IDs
   • LLM classifies paper against the registered topic set (0..n matches)
   • KG writer (mardiclient): paper item + "main subject" link to each matched topic
      ▼
[Step 2] MediaWiki page generation
   • SPARQL per topic -> papers, authors, institutions
   • render wikitext -> publish per-topic page + master index via MediaWiki API
```

### Step 1 — Harvest + metadata generation

- Harvest arXiv `math.*` via **OAI-PMH** (`http://export.arxiv.org/oai2`,
  `metadataPrefix=arXiv`, `set=math`), incremental by `from`/`until` date window
  and resumption tokens.
- A **state file** (JSON or SQLite at a configurable path) records the
  last-harvest cursor and seen arXiv IDs, so each run only processes new papers
  and re-runs are safe.
- At startup, **query the KG** for all registered topics — items that are
  `instance of` (P31) the `overview topic` class — collecting each topic's QID,
  label, and description. This set drives classification for the run.
- For each new paper, call the LLM with:
  - the **registered topic set from the KG** (each topic's QID, label,
    description), and
  - the paper's **title + abstract**.
- The LLM returns JSON listing which topics the paper belongs to:
  `{"topics": ["Q1234", "Q5678"]}` — possibly empty (paper irrelevant to all
  registered topics). Response is schema-validated against the known topic QIDs
  with one retry; unknown QIDs are dropped. A per-topic confidence/threshold may
  be applied.
- A paper matching **no** topic is skipped (not imported) — only relevant papers
  enter the KG.
- KG writes via `mardiclient` (all idempotent upserts):
  - Ensure the **paper item** (search by arXiv ID P21 first): title (P159),
    authors (P16 / author-name-string P43), publication date (P28), arXiv
    classification (P22), DOI (P27) when present, `instance of` (P31) scholarly
    article (Q56887).
  - Add statement: paper —`main subject` (P30)→ each matched topic item.

### Step 2 — MediaWiki page generation

- For each topic, query the KG via **SPARQL** for its papers (title, year,
  authors, arXiv link) and aggregated authors/institutions (traversed through
  the papers' author statements — no extra KG writes needed).
- Render **wikitext** per topic: description, a sortable paper table
  (ALPS-style: title / authors / year / link), top authors and institutions, and
  links to related topics. Also render a **master index page** listing all
  topics.
- Publish via the **MediaWiki API** (edit with bot credentials). Each page is
  regenerated in full every run (idempotent).

### KG data model (Wikibase)

- **`overview topic` class** — a dedicated Wikibase item created **once** as a
  prerequisite; its QID is recorded in `kg/model.py` (and overridable via env).
  Every topic item is `instance of` (P31) → this class.
- **Topic items** — created and maintained **directly in the KG** (wiki/API),
  not by the pipeline. Each carries: `instance of` (P31) → `overview topic`, a
  label (topic name), and a description (used as LLM classification guidance and
  as the page intro). The pipeline **reads** them; it does not create or edit
  them.
  - Optional bridge: a curator may add `subclass of` (P36) → a canonical
    existing concept (e.g. *numerical analysis* `Q6481500`) so the topic stays
    linked into the wider graph. The pipeline ignores this for classification but
    pages may surface it.
- **Publication item** — slim arXiv→Wikibase mapping we own: arXiv ID (P21),
  title (P159), author (P16) / author-name-string (P43), publication date (P28),
  arXiv classification (P22), DOI (P27) when present, `instance of` (P31) →
  scholarly article (Q56887).
- **Links** — publication —`main subject` (P30)→ topic item. Topic↔author and
  topic↔institution are derived at page-build time via SPARQL over the papers'
  author and affiliation (P55) statements; no extra writes.
- All writes are idempotent upserts (search-before-create), so no rollback
  machinery is required.

### Resolved MaRDI Wikibase IDs

Verified against `https://portal.mardi4nfdi.de` on 2026-06-20. Centralized in
`kg/model.py`.

| Concept | ID | Notes |
|---|---|---|
| instance of | P31 | typing statement |
| arXiv ID | P21 | dedupe key for papers |
| arXiv classification | P22 | math.* category strings |
| arXiv author ID | P172 | when resolvable |
| author (item) | P16 | use when author item exists |
| author name string | P43 | fallback when no author item |
| publication date | P28 | |
| title | P159 | |
| DOI | P27 | when present |
| main subject | P30 | paper → topic link |
| subclass of | P36 | optional bridge: topic → canonical concept |
| affiliation | P55 | (affiliation string P49) — for institution traversal |
| published in | P200 | optional |
| scholarly article (class) | Q56887 | paper `instance of` value |
| preprint (class) | Q159099 | alternative paper type for arXiv-only items |
| **overview topic (class)** | **Q?? — create once** | topic items' `instance of` value; QID recorded in `kg/model.py` |

> Author-item creation/disambiguation is intentionally minimal in v1: prefer an
> existing author item (P16) when `mardiclient` resolves one, else record the
> author name string (P43). Richer disambiguation is a future extension.

## Project layout

```
topic-overviews/
  config.example.toml          # dev-only overrides; real config comes from env vars
  topic_overviews/
    config.py                  # env-var-first config resolution (+ optional TOML/.env)
    state.py                   # last-harvest cursor + seen-IDs dedupe
    harvest/arxiv_oai.py       # OAI-PMH fetch + parse
    llm/topic_classifier.py    # Anthropic call + JSON contract + validation/retry
    kg/model.py                # resolved property/QID constants (incl. overview-topic class), schema helpers
    kg/topics.py               # query registered overview topics from the KG
    kg/client.py               # mardiclient wrapper: paper item + main-subject links
    wiki/page_builder.py       # wikitext templates (golden-testable, pure)
    wiki/publisher.py          # MediaWiki API edits
    pipeline.py                # harvest_step(), generate_pages_step()
    __main__.py                # orchestrator: runs the steps in order
  tests/
  docs/superpowers/specs/      # this spec
```

### Prefect-readiness

Each step in `pipeline.py` is a plain, typed function that takes its config/
inputs and returns its result; `__main__` calls them in sequence. Porting to
Prefect later is mechanical: decorate each step with `@task`, wrap the sequence
in a `@flow`. No restructuring required. This is an explicit design constraint —
do not introduce orchestration state that would resist that port.

### Configuration (env-var-first)

**Secrets and operational settings** resolve from **environment variables**
first, so the same code runs locally, in a container, and in a future Prefect
flow (Prefect Secret blocks surface as env vars). An optional local `.env` is
supported for developer convenience only:

- `TOPIC_OVERVIEWS_ARXIV_SET` (default `math`)
- `TOPIC_OVERVIEWS_MODEL` (default `claude-haiku-4-5`)
- `TOPIC_OVERVIEWS_RELEVANCE_THRESHOLD`
- `TOPIC_OVERVIEWS_STATE_PATH`
- `TOPIC_OVERVIEWS_OVERVIEW_TOPIC_QID` (QID of the `overview topic` class; default in `kg/model.py`)
- `ANTHROPIC_API_KEY`
- `MEDIAWIKI_API_URL`, `MEDIAWIKI_BOT_USER`, `MEDIAWIKI_BOT_PASSWORD` (page edits)
- `WIKIBASE_URL` + Wikibase write credentials as used by `mardiclient`
- `SPARQL_ENDPOINT_URL`
- `TOPIC_OVERVIEWS_DRY_RUN`

There is **no topic-list config file** — the topic registry lives entirely in the
KG (items of the `overview topic` class). `kg/topics.py` reads it at startup.

## Error handling

- Per-paper failures are logged and skipped — one bad paper never fails the run.
- LLM responses are schema-validated with a single retry; on repeated failure the
  paper is skipped and logged.
- `--dry-run` / `TOPIC_OVERVIEWS_DRY_RUN` performs harvesting and LLM assignment
  but skips all KG and MediaWiki writes.
- Idempotent upserts mean a crashed run is recovered by simply re-running.

## Testing (TDD)

- `kg/topics.py` — mocked KG query; assert it returns the registered topics
  (QID/label/description) for `instance of overview topic`, and handles an empty
  registry.
- `harvest/arxiv_oai.py` — parse fixture OAI-PMH XML into paper records.
- `llm/topic_classifier.py` — mocked Anthropic client; assert prompt includes the
  registered topic set + paper text; valid/empty/invalid/retry JSON paths behave;
  unknown topic QIDs are dropped.
- `kg/client.py` — mocked `mardiclient`; assert the exact statements written for
  new vs. existing paper and the `main subject` link to each matched topic.
- `wiki/page_builder.py` — golden wikitext for a sample topic + index page.
- `pipeline.py` — step ordering and that Step 2 consumes Step 1's results.
- Optional: integration test against a dev Wikibase / MediaWiki.

## Prerequisites / open items for implementation

- A MediaWiki **bot account** + Wikibase write credentials on the MaRDI portal.
- An **Anthropic API key**.
- Property/class IDs are resolved (see table above). **Create the `overview
  topic` class item once** in the KG and record its QID in `kg/model.py`.
- **Register the initial topics** in the KG as `overview topic` items (label +
  description, optional `subclass of` bridge) — done in the KG, not in code.
- Author/affiliation modeling depends on what `mardiclient` exposes — confirm the
  minimal P16/P43 path during implementation.
- Decide the MediaWiki **namespace / page-naming convention** for topic pages and
  the index (e.g. `Topic:<name>` and a `Topic overview` index page).
- Confirm arXiv OAI-PMH usage limits and set a polite request cadence.

## Future extensions (out of scope for v1)

- Port to a Prefect flow (the structure already supports it).
- Additional sources: Crossref, OpenAlex.
- Richer author disambiguation / institution resolution.
- A correction/curation workflow over auto-generated assignments.
