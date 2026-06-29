"""Prefect flow for uploading arXiv HTML5→Markdown to lakeFS.

Secrets are read from Prefect Secret blocks (shared with the main workflow).
All other config is passed as flow parameters so individual runs can be
adjusted from the Prefect UI.

Local dev: use run_locally_arxiv_update.sh with env vars set.
"""
from __future__ import annotations

import logging
import os

from prefect import flow, get_run_logger
from prefect.blocks.system import Secret

from topic_overviews.arxiv_update import run_arxiv_update

_SPARQL_ENDPOINT_URL = "https://query.portal.mardi4nfdi.de/sparql"
_RESEARCH_THEME_QID = "Q7266523"
_LAKEFS_URL = os.environ.get("LAKEFS_URL", "https://lake-bioinfmed.zib.de")
_LAKEFS_REPO = os.environ.get("LAKEFS_REPO", "mardi-fdo-data")
_LAKEFS_BRANCH = os.environ.get("LAKEFS_BRANCH", "main")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("topic_overviews").setLevel(logging.INFO)


@flow(name="topic-overviews-arxiv-update", log_prints=True)
def arxiv_update(
    limit: int = 50,
    dry_run: bool = False,
) -> None:
    logger = get_run_logger()

    try:
        lakefs_user = Secret.load("lakefs-user").get()
        lakefs_password = Secret.load("lakefs-password").get()
    except Exception as exc:
        logger.error("Could not load lakeFS secrets: %s", exc)
        raise

    result = run_arxiv_update(
        lakefs_url=_LAKEFS_URL,
        lakefs_user=lakefs_user,
        lakefs_password=lakefs_password,
        lakefs_repo=_LAKEFS_REPO,
        lakefs_branch=_LAKEFS_BRANCH,
        sparql_endpoint=_SPARQL_ENDPOINT_URL,
        research_theme_qid=_RESEARCH_THEME_QID,
        limit=limit,
        dry_run=dry_run,
    )

    logger.info(
        "Done. Uploaded: %d  Already existed: %d  Errors: %d",
        result.uploaded, result.skipped_exists, result.skipped_error,
    )


if __name__ == "__main__":
    arxiv_update()
