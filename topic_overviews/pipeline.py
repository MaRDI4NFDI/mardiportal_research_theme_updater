"""Sequential pipeline steps. Plain functions with injected deps (Prefect-ready)."""
from __future__ import annotations

import datetime
import dataclasses
import logging

from .config import Config
from .state import State
from .harvest.arxiv_search import search_records
from .kg.topics import Topic
from .llm.topic_classifier import classify_paper
from .llm.summarizer import summarize_paper
from .llm.keyworder import keywords_paper
from .wiki.page_builder import build_index_page, RESEARCH_THEME_STUB, INDEX_PAGE_TITLE

log = logging.getLogger(__name__)


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


def harvest_step(
    config: Config,
    state: State,
    *,
    topics: list[Topic],
    kg,
    fetch=default_harvest,
    classify=classify_paper,
    summarize=summarize_paper,
    keyworder=keywords_paper,
    publisher=None,
) -> int:
    imported = 0
    considered = 0
    imported_titles: list[str] = []
    for harvest_config in _harvest_configs(config, topics):
        log.info("Harvesting arXiv query: %s", harvest_config.arxiv_query)
        for record in fetch(harvest_config):
            if record.arxiv_id in state.seen_ids:
                continue
            state.seen_ids.add(record.arxiv_id)
            considered += 1
            try:
                matched = classify(
                    record, topics, model=config.model, api_key=config.anthropic_api_key
                )
                if matched:
                    if not config.dry_run:
                        tldr = summarize(
                            record, model=config.model, api_key=config.anthropic_api_key
                        )
                        keywords = keyworder(
                            record, model=config.model, api_key=config.anthropic_api_key
                        )
                        paper_qid = kg.import_paper(
                            record, tldr=tldr, keywords=keywords,
                            generated_by=config.model_qid or None,
                        )
                        for topic_qid in matched:
                            kg.link_topic(topic_qid, paper_qid)
                    imported += 1
                    imported_titles.append(record.title)
            except Exception as exc:
                log.warning("Skipping paper %s due to error: %s", record.arxiv_id, exc)
            if config.harvest_limit and considered >= config.harvest_limit:
                break
        if config.harvest_limit and considered >= config.harvest_limit:
            break
    # Purge each new paper's page so it renders fresh on the portal.
    if publisher is not None and not config.dry_run and imported_titles:
        publisher.purge(imported_titles)
    state.last_harvest = datetime.date.today().isoformat()
    return imported


def ensure_theme_pages_step(
    config: Config,
    *,
    topics: list[Topic],
    publisher,
    kg,
) -> list[str]:
    """Make every research theme renderable: ensure each has a ``{{ResearchTheme}}``
    wiki page connected to its item via the ``mardi`` sitelink. Refresh the index.

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

    publisher.edit(
        INDEX_PAGE_TITLE, build_index_page(topics), "Update research theme index"
    )
    # Purge theme pages (so their live tables refresh) and the index.
    publisher.purge(page_titles + [INDEX_PAGE_TITLE])
    return titles
