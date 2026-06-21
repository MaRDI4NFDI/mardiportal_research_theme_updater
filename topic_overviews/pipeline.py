"""Sequential pipeline steps. Plain functions with injected deps (Prefect-ready)."""
from __future__ import annotations

import datetime
import dataclasses
import logging

from .config import Config
from .state import State
from .harvest.arxiv_search import search_records
from .harvest.openalex import fetch_openalex_records
from .kg.topics import Topic
from .llm.topic_classifier import classify_paper
from .llm.summarizer import summarize_paper
from .llm.keyworder import keywords_paper
from .llm.client import make_llm_client
from .wiki.page_builder import RESEARCH_THEME_STUB

log = logging.getLogger(__name__)


class PipelineError(Exception):
    """Fatal pipeline error — should terminate the run."""


def default_harvest(config: Config):
    """Default paper source: arXiv keyword search over a recent date window."""
    return search_records(config.arxiv_query, config.since_days)


def _harvest_configs(config: Config, topics: list[Topic]) -> list[Config]:
    """Build one arXiv-search config per distinct theme query.

    Theme-owned queries are preferred. The global query remains a compatibility
    fallback for themes that do not yet have the KG property set.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        query = (topic.arxiv_query or config.arxiv_query).strip()
        if query and query not in seen:
            seen.add(query)
            queries.append(query)
    if not queries and config.arxiv_query.strip():
        queries.append(config.arxiv_query.strip())
    return [dataclasses.replace(config, arxiv_query=query) for query in queries]


def _openalex_queries(topics: list[Topic]) -> list[str]:
    """Collect unique non-empty openalex_query values from topics, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for topic in topics:
        q = topic.openalex_query.strip()
        if q and q not in seen:
            seen.add(q)
            result.append(q)
    return result


def _process_record(
    record,
    source_label: str,
    covering: list[Topic],
    config: Config,
    state,
    kg,
    classify,
    summarize,
    keyworder,
    llm,
    model: str | None,
    topic_label: dict,
    imported_titles: list,
    imported_qids: list,
) -> bool:
    """Classify, optionally import, and link one record. Returns True if imported."""
    rid = record.record_id
    if rid in state.seen_ids:
        return False
    state.seen_ids.add(rid)

    existing_qid = None
    get_paper_qid = getattr(kg, "get_paper_qid", None)
    paper_has_tldr = getattr(kg, "paper_has_tldr", None)
    if callable(get_paper_qid) and callable(paper_has_tldr) and record.arxiv_id:
        existing_qid = get_paper_qid(record.arxiv_id)
        if existing_qid and paper_has_tldr(existing_qid):
            log.info(
                "Skipping %s paper %s (%s): KG item %s already has P1963",
                source_label, rid, record.title, existing_qid,
            )
            return False

    log.info("Classifying %s paper %s (%s) with model %s", source_label, rid, record.title, model)
    matched = classify(
        record,
        covering,
        model=model,
        api_key=config.anthropic_api_key,
        llm=llm,
    )
    matched_labels = [f"{topic_label.get(q, q)} ({q})" for q in matched] if matched else []
    log.info(
        "Classified %s paper %s (%s) into: %s",
        source_label, rid, record.title,
        ", ".join(matched_labels) if matched_labels else "no matching theme",
    )
    if not matched:
        return False

    if not config.dry_run:
        log.info("Generating TL;DR and keywords for %s (%s)", rid, record.title)
        tldr = summarize(record, model=model, api_key=config.anthropic_api_key, llm=llm)
        if not tldr:
            raise PipelineError(f"No TL;DR generated for {rid} ({record.title!r})")
        keywords = keyworder(record, model=model, api_key=config.anthropic_api_key, llm=llm)
        paper_qid = kg.import_paper(
            record, tldr=tldr, keywords=keywords, generated_by=config.model_qid or None,
        )
        imported_qids.append(paper_qid)
        log.info("Inserted %s paper %s as KG item %s", source_label, rid, paper_qid)
        for topic_qid in matched:
            kg.link_topic(topic_qid, paper_qid)
            log.info(
                "Linked %s to theme %s (%s)", paper_qid,
                topic_label.get(topic_qid, topic_qid), topic_qid,
            )
            if config.theme_max_papers:
                kg.enforce_theme_limit(topic_qid, config.theme_max_papers)
    imported_titles.append(record.title)
    return True


def harvest_step(
    config: Config,
    state: State,
    *,
    topics: list[Topic],
    kg,
    fetch=default_harvest,
    fetch_oa=None,
    classify=classify_paper,
    summarize=summarize_paper,
    keyworder=keywords_paper,
    llm=None,
    model: str | None = None,
    publisher=None,
) -> int:
    imported = 0
    imported_titles: list[str] = []
    imported_qids: list[str] = []
    topic_label = {t.qid: t.label for t in topics}
    llm = llm or make_llm_client(config)
    model = model or config.model_qid

    # --- arXiv pass ---
    for harvest_config in _harvest_configs(config, topics):
        covering = [
            t for t in topics
            if (t.arxiv_query or config.arxiv_query).strip() == harvest_config.arxiv_query
        ]
        log.info(
            "Harvesting arXiv query %r for %d theme(s): %s",
            harvest_config.arxiv_query, len(covering),
            ", ".join(f"{t.label} ({t.qid})" for t in covering),
        )
        query_imported = 0
        for record in fetch(harvest_config):
            try:
                did_import = _process_record(
                    record, "arXiv", covering, config, state, kg,
                    classify, summarize, keyworder, llm, model,
                    topic_label, imported_titles, imported_qids,
                )
            except PipelineError:
                raise
            except Exception as exc:
                log.warning("Skipping paper %s due to error: %s", record.arxiv_id, exc)
                continue
            if did_import:
                imported += 1
                query_imported += 1
            if config.harvest_limit and query_imported >= config.harvest_limit:
                break

    # --- OpenAlex pass ---
    _fetch_oa = fetch_oa or (
        lambda qs, sd, **kw: fetch_openalex_records(
            qs, sd, email=config.openalex_email
        )
    )
    for oa_query in _openalex_queries(topics):
        covering = [t for t in topics if t.openalex_query.strip() == oa_query]
        log.info(
            "Harvesting OpenAlex query %r for %d theme(s): %s",
            oa_query, len(covering),
            ", ".join(f"{t.label} ({t.qid})" for t in covering),
        )
        query_imported = 0
        for record in _fetch_oa(oa_query, config.since_days):
            try:
                did_import = _process_record(
                    record, "OpenAlex", covering, config, state, kg,
                    classify, summarize, keyworder, llm, model,
                    topic_label, imported_titles, imported_qids,
                )
            except PipelineError:
                raise
            except Exception as exc:
                log.warning(
                    "Skipping OpenAlex paper %s due to error: %s",
                    getattr(record, "openalex_id", "?"), exc,
                )
                continue
            if did_import:
                imported += 1
                query_imported += 1
            if config.harvest_limit and query_imported >= config.harvest_limit:
                break

    if publisher is not None and not config.dry_run and imported_titles:
        publisher.purge(imported_titles)
    state.last_harvest = datetime.date.today().isoformat()
    if imported_qids:
        log.info("Harvest inserted %d paper(s): %s", imported, ", ".join(imported_qids))
    else:
        log.info("Harvest inserted 0 papers")
    return imported


def ensure_theme_pages_step(
    config: Config,
    *,
    topics: list[Topic],
    publisher,
    kg,
) -> list[str]:
    """Make every research theme renderable: ensure each has a ``{{ResearchTheme}}``
    wiki page connected to its item via the ``mardi`` sitelink.

    Per theme:
      - already has a sitelink -> leave it (page is wired; curator owns content);
      - else if a page with the target title already exists -> log and skip (do
        not hijack an unrelated page);
      - else -> create the stub page and set the item's sitelink to it.

    The page title is the theme label (main namespace, Person convention).
    Returns the theme labels processed.
    """
    titles = [topic.label for topic in topics]
    if config.dry_run:
        return titles

    page_titles: list[str] = []
    for topic in topics:
        connected = kg.get_theme_sitelink(topic.qid)
        if connected:
            page_titles.append(connected)  # already wired; refresh its (cached) table
            continue
        title = topic.label
        if publisher.page_exists(title):
            log.warning(
                "Theme %s (%s): page %r already exists and is not linked — "
                "skipping to avoid hijacking it; connect it manually.",
                topic.qid, topic.label, title,
            )
            continue
        publisher.edit(title, RESEARCH_THEME_STUB, "Create research theme page")
        kg.set_theme_sitelink(topic.qid, title)
        page_titles.append(title)

    # Purge theme pages so their live tables refresh.
    publisher.purge(page_titles)
    return titles
