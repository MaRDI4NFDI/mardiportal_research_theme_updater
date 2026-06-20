"""Sequential pipeline steps. Plain functions with injected deps (Prefect-ready)."""
from __future__ import annotations

import datetime
import logging

from .config import Config
from .state import State
from .harvest.arxiv_oai import fetch_records
from .kg.topics import Topic
from .llm.topic_classifier import classify_paper
from .wiki.page_builder import build_index_page, RESEARCH_THEME_STUB, INDEX_PAGE_TITLE

log = logging.getLogger(__name__)


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
        try:
            matched = classify(
                record, topics, model=config.model, api_key=config.anthropic_api_key
            )
            if not matched:
                continue
            if not config.dry_run:
                paper_qid = kg.import_paper(record)
                for topic_qid in matched:
                    kg.link_topic(topic_qid, paper_qid)
            imported += 1
        except Exception as exc:
            log.warning("Skipping paper %s due to error: %s", record.arxiv_id, exc)
            continue
    state.last_harvest = datetime.date.today().isoformat()
    return imported


def generate_pages_step(
    config: Config,
    *,
    topics: list[Topic],
    publisher,
) -> list[str]:
    """Ensure each research theme has a ``{{ResearchTheme}}`` page, and refresh
    the master index. Theme pages render live via the template, so nothing is
    built here — the page is created only if missing (curator edits are kept).

    Returns the theme page titles processed.
    """
    titles = [topic.label for topic in topics]
    if not config.dry_run:
        for topic in topics:
            publisher.ensure_page(
                topic.label, RESEARCH_THEME_STUB, "Create research theme page"
            )
        publisher.edit(
            INDEX_PAGE_TITLE, build_index_page(topics), "Update research theme index"
        )
    return titles
