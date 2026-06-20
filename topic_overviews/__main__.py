"""CLI entry point: run the harvest + page-generation pipeline once."""
from __future__ import annotations

import argparse
import dataclasses
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

    topics = load_registered_topics(config.sparql_endpoint_url, config.research_theme_qid)
    log.info("Loaded %d registered research themes", len(topics))

    kg = None if config.dry_run else make_kg_client(config)

    if not args.themes_only:
        state = load_state(config.state_path)
        imported = pipeline.harvest_step(config, state, topics=topics, kg=kg)
        log.info("Imported %d papers", imported)
        save_state(config.state_path, state)

    publisher = None if config.dry_run else _make_publisher(config)
    pages = pipeline.ensure_theme_pages_step(
        config, topics=topics, publisher=publisher, kg=kg
    )
    log.info("Ensured %d research theme pages", len(pages))


def _make_publisher(config) -> WikiPublisher:
    pub = WikiPublisher(config.mediawiki_api_url, config.mediawiki_bot_user, config.mediawiki_bot_password)
    pub.login()
    return pub


if __name__ == "__main__":
    main()
