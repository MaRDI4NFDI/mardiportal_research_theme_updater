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
