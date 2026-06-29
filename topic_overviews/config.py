"""Environment-variable-first configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Config:
    arxiv_query: str
    arxiv_query_property: str
    openalex_query_property: str
    openalex_email: str
    zbmath_query: str
    zbmath_query_property: str
    since_days_property: str
    auto_classify_keywords_property: str
    maintainer_qid: str
    since_days: int
    llm_provider: str
    openai_base_url: str
    openai_api_key: str
    research_theme_qid: str
    model_qid: str
    harvest_limit: int
    theme_max_papers: int
    anthropic_api_key: str
    mediawiki_api_url: str
    mediawiki_bot_user: str
    mediawiki_bot_password: str
    wikibase_url: str
    sparql_endpoint_url: str
    s2_api_key: str
    dry_run: bool
    lakefs_url: str
    lakefs_user: str
    lakefs_password: str
    lakefs_repo: str
    lakefs_branch: str


def _flag(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def load_config(env: Mapping[str, str] = os.environ) -> Config:
    return Config(
        arxiv_query=env.get("TOPIC_OVERVIEWS_ARXIV_QUERY", ""),
        arxiv_query_property=env.get("TOPIC_OVERVIEWS_ARXIV_QUERY_PROPERTY", "P1965"),
        openalex_query_property=env.get("TOPIC_OVERVIEWS_OPENALEX_QUERY_PROPERTY", ""),
        openalex_email=env.get("TOPIC_OVERVIEWS_OPENALEX_EMAIL", ""),
        zbmath_query=env.get("TOPIC_OVERVIEWS_ZBMATH_QUERY", ""),
        zbmath_query_property=env.get("TOPIC_OVERVIEWS_ZBMATH_QUERY_PROPERTY", "P1979"),
        since_days_property=env.get("TOPIC_OVERVIEWS_SINCE_DAYS_PROPERTY", ""),
        auto_classify_keywords_property=env.get("TOPIC_OVERVIEWS_AUTO_CLASSIFY_KEYWORDS_PROPERTY", "P1990"),
        maintainer_qid=env.get("TOPIC_OVERVIEWS_MAINTAINER_QID", "Q7270033"),
        since_days=int(env.get("TOPIC_OVERVIEWS_SINCE_DAYS", "10")),
        llm_provider=env.get("TOPIC_OVERVIEWS_LLM_PROVIDER", "anthropic"),
        openai_base_url=env.get("TOPIC_OVERVIEWS_OPENAI_BASE_URL", "https://ollama.zib.de/api"),
        openai_api_key=env.get("TOPIC_OVERVIEWS_OPENAI_API_KEY", ""),
        research_theme_qid=env.get("TOPIC_OVERVIEWS_RESEARCH_THEME_QID", "Q0"),
        model_qid=env.get("TOPIC_OVERVIEWS_MODEL_QID", ""),  # KG item of the LLM → P1966 model string; P1642 now always Q7270033
        harvest_limit=int(env.get("TOPIC_OVERVIEWS_HARVEST_LIMIT", "0")),  # 0 = unlimited
        theme_max_papers=int(env.get("TOPIC_OVERVIEWS_THEME_MAX_PAPERS", "100")),
        anthropic_api_key=env.get("ANTHROPIC_API_KEY", ""),
        mediawiki_api_url=env.get("MEDIAWIKI_API_URL", ""),
        mediawiki_bot_user=env.get("MEDIAWIKI_BOT_USER", ""),
        mediawiki_bot_password=env.get("MEDIAWIKI_BOT_PASSWORD", ""),
        wikibase_url=env.get("WIKIBASE_URL", ""),
        sparql_endpoint_url=env.get("SPARQL_ENDPOINT_URL", ""),
        s2_api_key=env.get("TOPIC_OVERVIEWS_S2_API_KEY", ""),
        dry_run=_flag(env.get("TOPIC_OVERVIEWS_DRY_RUN", "false")),
        lakefs_url=env.get("LAKEFS_URL", ""),
        lakefs_user=env.get("LAKEFS_USER", ""),
        lakefs_password=env.get("LAKEFS_PASSWORD", ""),
        lakefs_repo=env.get("LAKEFS_REPO", ""),
        lakefs_branch=env.get("LAKEFS_BRANCH", "main"),
    )
