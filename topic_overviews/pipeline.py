"""Sequential pipeline steps. Plain functions with injected deps (Prefect-ready)."""
from __future__ import annotations

import datetime
import dataclasses
import logging

from .config import Config
from .state import State
from .harvest.arxiv_search import search_records
from .harvest.openalex import fetch_openalex_records
from .harvest.zbmath import fetch_zbmath_records
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


def _effective_since_days(topic: Topic, config: Config) -> int:
    return topic.since_days if topic.since_days is not None else config.since_days


def _harvest_configs(config: Config, topics: list[Topic]) -> list[Config]:
    """Build one arXiv-search config per distinct (query, since_days) pair.

    Theme-owned queries and since_days are preferred. The global values remain
    fallbacks for themes that do not yet have the KG properties set.
    """
    seen: set[tuple[str, int]] = set()
    result: list[Config] = []
    for topic in topics:
        query = (topic.arxiv_query or config.arxiv_query).strip()
        days = _effective_since_days(topic, config)
        if query and (query, days) not in seen:
            seen.add((query, days))
            result.append(dataclasses.replace(config, arxiv_query=query, since_days=days))
    if not result and config.arxiv_query.strip():
        result.append(config)
    return result


def _openalex_query_configs(config: Config, topics: list[Topic]) -> list[tuple[str, int]]:
    """Collect unique (openalex_query, since_days) pairs from topics, preserving order."""
    seen: set[tuple[str, int]] = set()
    result: list[tuple[str, int]] = []
    for topic in topics:
        q = topic.openalex_query.strip()
        if not q:
            continue
        days = _effective_since_days(topic, config)
        if (q, days) not in seen:
            seen.add((q, days))
            result.append((q, days))
    return result


def _zbmath_query_configs(config: Config, topics: list[Topic]) -> list[tuple[str, int]]:
    """Collect unique (zbmath_query, since_days) pairs from topics, preserving order."""
    seen: set[tuple[str, int]] = set()
    result: list[tuple[str, int]] = []
    for topic in topics:
        q = (topic.zbmath_query or config.zbmath_query).strip()
        if not q:
            continue
        days = _effective_since_days(topic, config)
        if (q, days) not in seen:
            seen.add((q, days))
            result.append((q, days))
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
            log.warning("Skipping %s (%s): LLM returned empty TL;DR", rid, record.title)
            return False
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
    fetch_zb=None,
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
            and _effective_since_days(t, config) == harvest_config.since_days
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
                log.warning("Skipping paper %s due to error: %s", record.record_id, exc)
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
    for oa_query, oa_since_days in _openalex_query_configs(config, topics):
        covering = [
            t for t in topics
            if t.openalex_query.strip() == oa_query
            and _effective_since_days(t, config) == oa_since_days
        ]
        log.info(
            "Harvesting OpenAlex query %r (since_days=%d) for %d theme(s): %s",
            oa_query, oa_since_days, len(covering),
            ", ".join(f"{t.label} ({t.qid})" for t in covering),
        )
        query_imported = 0
        for record in _fetch_oa(oa_query, oa_since_days):
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

    # --- zbMATH pass ---
    _fetch_zb = fetch_zb or (lambda qs, sd, **kw: fetch_zbmath_records(qs, sd))
    for zb_query, zb_since_days in _zbmath_query_configs(config, topics):
        covering = [
            t for t in topics
            if (t.zbmath_query or config.zbmath_query).strip() == zb_query
            and _effective_since_days(t, config) == zb_since_days
        ]
        log.info(
            "Harvesting zbMATH query %r (since_days=%d) for %d theme(s): %s",
            zb_query, zb_since_days, len(covering),
            ", ".join(f"{t.label} ({t.qid})" for t in covering),
        )
        query_imported = 0
        for record in _fetch_zb(zb_query, zb_since_days):
            try:
                did_import = _process_record(
                    record, "zbMATH", covering, config, state, kg,
                    classify, summarize, keyworder, llm, model,
                    topic_label, imported_titles, imported_qids,
                )
            except PipelineError:
                raise
            except Exception as exc:
                log.warning(
                    "Skipping zbMATH paper %s due to error: %s",
                    getattr(record, "zbmath_id", "?"), exc,
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
