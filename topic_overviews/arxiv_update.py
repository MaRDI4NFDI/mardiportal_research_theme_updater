"""Core logic for uploading arXiv HTML5→Markdown to lakeFS.

Called by workflow_arxiv_update.py (Prefect) and can be invoked directly
from a local run script.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import lakefs
import requests

from .arxiv_to_md import fetch_and_convert
from .lakefs_upload import component_path, upload_markdown, commit_upload
from .kg import model as M

log = logging.getLogger(__name__)

_SPARQL_ENDPOINT_DEFAULT = "https://query.portal.mardi4nfdi.de/sparql"
_RESEARCH_THEME_QID_DEFAULT = "Q7266523"


@dataclass
class ArxivUpdateResult:
    uploaded: int
    skipped_exists: int
    skipped_error: int


def get_theme_papers_with_arxiv_id(
    endpoint: str,
    research_theme_qid: str,
    session: requests.Session,
) -> list[tuple[str, str]]:
    """Return [(paper_qid, arxiv_id), ...] for all theme papers that have an arXiv ID."""
    query = f"""
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT DISTINCT ?paper ?arxivId WHERE {{
  ?theme wdt:{M.P_INSTANCE_OF} wd:{research_theme_qid} .
  ?theme wdt:{M.P_HAS_PART} ?paper .
  ?paper wdt:{M.P_ARXIV_ID} ?arxivId .
}}
ORDER BY ?paper
"""
    resp = session.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=120,
    )
    resp.raise_for_status()
    rows = [
        {var: cell["value"] for var, cell in row.items()}
        for row in resp.json()["results"]["bindings"]
    ]
    return [(_qid(row["paper"]), row["arxivId"]) for row in rows]


def _qid(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]


def _already_uploaded(qid: str, repo: str, branch: str, client: lakefs.Client) -> bool:
    path = component_path(qid)
    try:
        lakefs.repository(repo, client=client).branch(branch).object(path).stat()
        return True
    except Exception:
        return False


def run_arxiv_update(
    *,
    lakefs_url: str,
    lakefs_user: str,
    lakefs_password: str,
    lakefs_repo: str,
    lakefs_branch: str = "main",
    sparql_endpoint: str = _SPARQL_ENDPOINT_DEFAULT,
    research_theme_qid: str = _RESEARCH_THEME_QID_DEFAULT,
    limit: int = 0,
    dry_run: bool = False,
) -> ArxivUpdateResult:
    """Fetch arXiv HTML5 for all theme papers and upload Markdown to lakeFS.

    Papers already present in lakeFS are skipped. A single commit is made at
    the end of the run.
    """
    log.info("SPARQL endpoint : %s", sparql_endpoint)
    log.info("Research theme  : %s", research_theme_qid)
    log.info("lakeFS          : %s  repo=%s  branch=%s", lakefs_url, lakefs_repo, lakefs_branch)
    log.info("dry_run=%s  limit=%s", dry_run, limit or "unlimited")

    session = requests.Session()
    lf_client = (
        lakefs.Client(host=lakefs_url, username=lakefs_user, password=lakefs_password)
        if not dry_run else None
    )

    log.info("Querying KG for theme papers with arXiv IDs...")
    papers = get_theme_papers_with_arxiv_id(sparql_endpoint, research_theme_qid, session)
    log.info("Found %d paper(s) with an arXiv ID across all themes", len(papers))

    if limit:
        papers = papers[:limit]
        log.info("Limited to first %d paper(s)", limit)

    uploaded = skipped_exists = skipped_error = 0

    for i, (qid, arxiv_id) in enumerate(papers, 1):
        log.info("[%d/%d] %s  arXiv:%s", i, len(papers), qid, arxiv_id)

        if not dry_run and _already_uploaded(qid, lakefs_repo, lakefs_branch, lf_client):
            log.info("  -> already uploaded, skipping")
            skipped_exists += 1
            continue

        try:
            markdown = fetch_and_convert(arxiv_id)
        except RuntimeError as exc:
            log.warning("  -> fetch failed: %s", exc)
            skipped_error += 1
            time.sleep(1)
            continue

        if dry_run:
            log.info("  -> [dry-run] would upload %d chars to %s/%s",
                     len(markdown), lakefs_branch, component_path(qid))
            uploaded += 1
            continue

        try:
            lakefs_path = upload_markdown(
                qid, markdown,
                url=lakefs_url, user=lakefs_user, password=lakefs_password,
                repo=lakefs_repo, branch=lakefs_branch,
            )
            log.info("  -> uploaded to %s", lakefs_path)
            uploaded += 1
        except Exception as exc:
            log.warning("  -> lakeFS upload failed: %s", exc)
            skipped_error += 1

        time.sleep(0.5)

    if not dry_run and uploaded > 0:
        log.info("Committing %d uploaded file(s)...", uploaded)
        try:
            commit_id = commit_upload(
                f"Add arXiv paper markdown ({uploaded} papers)",
                url=lakefs_url, user=lakefs_user, password=lakefs_password,
                repo=lakefs_repo, branch=lakefs_branch,
            )
            log.info("Committed: %s", commit_id)
        except Exception as exc:
            log.warning("Commit failed: %s", exc)

    return ArxivUpdateResult(
        uploaded=uploaded,
        skipped_exists=skipped_exists,
        skipped_error=skipped_error,
    )
