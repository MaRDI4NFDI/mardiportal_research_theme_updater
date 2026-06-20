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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("topic_overviews")

    config = load_config()
    if args.dry_run:
        config = dataclasses.replace(config, dry_run=True)

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
