"""Prefect flow entry point for the topic-overviews pipeline.

Secrets are read from Prefect Secret blocks (create them once with
workflow_create_secrets.py). All other config is passed as flow parameters
so individual runs can be adjusted from the Prefect UI.

Local dev: use run_locally.sh + .env instead.
"""
from __future__ import annotations

import logging
import subprocess

from prefect import flow, get_run_logger
from prefect.blocks.system import Secret

from topic_overviews.config import Config
from topic_overviews.state import State
from topic_overviews.kg.topics import load_registered_topics
from topic_overviews.kg.client import make_kg_client
from topic_overviews.kg.model_items import get_llm_model_identifier
from topic_overviews.llm.client import assert_openai_compatible_server_available
from topic_overviews.wiki.publisher import make_publisher
from topic_overviews import pipeline

# Non-secret, deployment-specific constants.
_MEDIAWIKI_API_URL = "https://portal.mardi4nfdi.de/w/api.php"
_WIKIBASE_URL = "https://portal.mardi4nfdi.de"
_SPARQL_ENDPOINT_URL = "https://query.portal.mardi4nfdi.de/sparql"
_RESEARCH_THEME_QID = "Q7266523"
_MODEL_QID = "Q7269921"
_OPENAI_BASE_URL = "https://ollama.zib.de/api"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("topic_overviews").setLevel(logging.INFO)


@flow(name="topic-overviews", log_prints=True)
def topic_overviews(
    since_days: int = 10,
    harvest_limit: int = 0,
    theme_max_papers: int = 100,
    dry_run: bool = False,
    themes_only: bool = False,
) -> None:
    logger = get_run_logger()

    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    sha = result.stdout.strip() if result.returncode == 0 else "unknown"
    logger.info("Running commit: %s", sha)

    bot_user = Secret.load("topic-overviews-bot-user").get()
    bot_password = Secret.load("topic-overviews-bot-password").get()
    openai_api_key = Secret.load("topic-overviews-openai-api-key").get()
    try:
        s2_api_key = Secret.load("topic-overviews-s2-api-key").get()
    except Exception:
        s2_api_key = ""

    config = Config(
        arxiv_query="",
        arxiv_query_property="P1965",
        openalex_query_property="P1967",
        openalex_email="",
        zbmath_query="",
        zbmath_query_property="P1979",
        since_days_property="P1968",
        since_days=since_days,
        llm_provider="openai",
        openai_base_url=_OPENAI_BASE_URL,
        openai_api_key=openai_api_key,
        research_theme_qid=_RESEARCH_THEME_QID,
        model_qid=_MODEL_QID,
        harvest_limit=harvest_limit,
        theme_max_papers=theme_max_papers,
        anthropic_api_key="",
        mediawiki_api_url=_MEDIAWIKI_API_URL,
        mediawiki_bot_user=bot_user,
        mediawiki_bot_password=bot_password,
        wikibase_url=_WIKIBASE_URL,
        sparql_endpoint_url=_SPARQL_ENDPOINT_URL,
        s2_api_key=s2_api_key,
        dry_run=dry_run,
    )

    model = ""
    if not themes_only:
        assert_openai_compatible_server_available(_OPENAI_BASE_URL)
        model = get_llm_model_identifier(_MEDIAWIKI_API_URL, _MODEL_QID)
        logger.info("LLM model identifier: %s", model)

    topics = load_registered_topics(
        _SPARQL_ENDPOINT_URL,
        _RESEARCH_THEME_QID,
        arxiv_query_property=config.arxiv_query_property,
        openalex_query_property=config.openalex_query_property,
        zbmath_query_property=config.zbmath_query_property,
        since_days_property=config.since_days_property,
    )
    logger.info("Loaded %d research themes", len(topics))
    for t in topics:
        logger.info("  %s: %s", t.qid, t.label)

    kg = None if dry_run else make_kg_client(config)
    publisher = None if dry_run else make_publisher(config)

    if not themes_only:
        # State is intentionally fresh per run: find_existing_paper checks all
        # known identifiers against the KG to skip already-imported papers.
        state = State()
        imported = pipeline.harvest_step(
            config, state, topics=topics, kg=kg, model=model, publisher=publisher,
        )
        logger.info("Harvest complete — imported %d paper(s)", imported)

    pages = pipeline.ensure_theme_pages_step(
        config, topics=topics, publisher=publisher, kg=kg,
    )
    logger.info("Theme pages ensured: %d", len(pages))


if __name__ == "__main__":
    topic_overviews()
