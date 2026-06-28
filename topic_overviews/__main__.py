"""CLI entry point: run the harvest + page-generation pipeline once."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from .config import load_config
from .state import State
from .kg.topics import count_all_topics, load_registered_topics
from .kg.client import make_kg_client
from .kg.model_items import get_llm_model_identifier
from .wiki.publisher import make_publisher
from . import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(prog="topic_overviews")
    parser.add_argument("--dry-run", action="store_true", help="harvest + classify but skip all writes")
    parser.add_argument(
        "--themes-only", action="store_true",
        help="skip arXiv harvest/classify; only ensure theme pages + sitelinks "
             "(no Anthropic key needed) — handy for testing a new theme",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("topic_overviews")

    config = load_config()
    if args.dry_run:
        config = dataclasses.replace(config, dry_run=True)

    model = ""
    if not args.themes_only:
        model = get_llm_model_identifier(config.mediawiki_api_url, config.model_qid)

    topics = load_registered_topics(
        config.sparql_endpoint_url,
        config.research_theme_qid,
        arxiv_query_property=config.arxiv_query_property,
        openalex_query_property=config.openalex_query_property,
        zbmath_query_property=config.zbmath_query_property,
        since_days_property=config.since_days_property,
        auto_classify_keywords_property=config.auto_classify_keywords_property,
        maintainer_qid=config.maintainer_qid,
    )
    if config.maintainer_qid:
        total = count_all_topics(config.sparql_endpoint_url, config.research_theme_qid)
        ignored = total - len(topics)
        log.info(
            "Loaded %d research theme(s) for automated updates (%d ignored — P19 not set to %s)",
            len(topics), ignored, config.maintainer_qid,
        )
    else:
        log.info("Loaded %d registered research theme(s)", len(topics))
    for t in topics:
        log.info("  %s: %s", t.qid, t.label)
        log.info("    Description: %s", t.description or "(none)")

    kg = None if config.dry_run else make_kg_client(config)
    publisher = None if config.dry_run else make_publisher(config)

    if not args.themes_only:
        try:
            imported = pipeline.harvest_step(
                config, State(), topics=topics, kg=kg, model=model, publisher=publisher
            )
        except pipeline.PipelineError as exc:
            log.error("Pipeline terminated: %s", exc)
            sys.exit(1)
        log.info("Imported %d papers", imported)

    pages = pipeline.ensure_theme_pages_step(
        config, topics=topics, publisher=publisher, kg=kg
    )
    log.info("Ensured %d research theme pages", len(pages))



if __name__ == "__main__":
    main()
