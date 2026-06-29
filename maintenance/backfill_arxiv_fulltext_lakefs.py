#!/usr/bin/env python3
"""Backfill arXiv HTML5→Markdown uploads to lakeFS for papers already in the KG.

Finds all paper items linked to any research theme (via P265) that carry a P21
(arXiv ID) claim, converts the arXiv HTML5 page to Markdown with clean LaTeX
formulas, and uploads the result to lakeFS.

Papers whose lakeFS object already exists are skipped (idempotent).

Usage:
    python maintenance/backfill_arxiv_fulltext_lakefs.py [--dry-run] [--limit N]

Environment variables:
    SPARQL_ENDPOINT_URL                  (default: https://query.portal.mardi4nfdi.de/sparql)
    TOPIC_OVERVIEWS_RESEARCH_THEME_QID   research theme class QID (default: Q7266523)
    LAKEFS_URL                           lakeFS endpoint URL
    LAKEFS_USER                          lakeFS access key ID
    LAKEFS_PASSWORD                      lakeFS secret access key
    LAKEFS_REPO                          lakeFS repository name
    LAKEFS_BRANCH                        lakeFS branch (default: main)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import lakefs
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topic_overviews.arxiv_to_md import fetch_and_convert
from topic_overviews.lakefs_upload import component_path, upload_markdown
from topic_overviews.kg import model as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPARQL_ENDPOINT = os.environ.get("SPARQL_ENDPOINT_URL", "https://query.portal.mardi4nfdi.de/sparql")
RESEARCH_THEME_QID = os.environ.get("TOPIC_OVERVIEWS_RESEARCH_THEME_QID", "Q7266523")


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def _run_sparql(endpoint: str, query: str, session: requests.Session) -> list[dict]:
    resp = session.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=120,
    )
    resp.raise_for_status()
    return [
        {var: cell["value"] for var, cell in row.items()}
        for row in resp.json()["results"]["bindings"]
    ]


def _qid(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]


def get_theme_papers_with_arxiv_id(
    endpoint: str,
    research_theme_qid: str,
    session: requests.Session,
) -> list[tuple[str, str]]:
    """Return [(paper_qid, arxiv_id), ...] for all theme papers that have P21."""
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
    rows = _run_sparql(endpoint, query, session)
    return [(_qid(row["paper"]), row["arxivId"]) for row in rows]


# ---------------------------------------------------------------------------
# lakeFS helpers
# ---------------------------------------------------------------------------

def _lakefs_client(url: str, user: str, password: str) -> lakefs.Client:
    return lakefs.Client(host=url, username=user, password=password)


def already_uploaded(qid: str, repo: str, branch: str, client: lakefs.Client) -> bool:
    """Return True if the lakeFS object for this QID already exists."""
    path = component_path(qid)
    try:
        lakefs.repository(repo, client=client).branch(branch).object(path).stat()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Skip actual uploads")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N papers (0 = unlimited)")
    args = parser.parse_args()

    lakefs_url = os.environ.get("LAKEFS_URL", "")
    lakefs_user = os.environ.get("LAKEFS_USER", "")
    lakefs_password = os.environ.get("LAKEFS_PASSWORD", "")
    lakefs_repo = os.environ.get("LAKEFS_REPO", "")
    lakefs_branch = os.environ.get("LAKEFS_BRANCH", "main")

    if not args.dry_run:
        missing = [k for k, v in {
            "LAKEFS_URL": lakefs_url,
            "LAKEFS_USER": lakefs_user,
            "LAKEFS_PASSWORD": lakefs_password,
            "LAKEFS_REPO": lakefs_repo,
        }.items() if not v]
        if missing:
            log.error("Missing required environment variables: %s", ", ".join(missing))
            sys.exit(1)

    session = requests.Session()
    lf_client = _lakefs_client(lakefs_url, lakefs_user, lakefs_password) if not args.dry_run else None

    log.info("Querying KG for theme papers with arXiv IDs...")
    papers = get_theme_papers_with_arxiv_id(SPARQL_ENDPOINT, RESEARCH_THEME_QID, session)
    log.info("Found %d paper(s) with an arXiv ID across all themes", len(papers))

    if args.limit:
        papers = papers[: args.limit]
        log.info("Limited to first %d paper(s)", args.limit)

    processed = skipped_exists = skipped_error = uploaded = 0

    for i, (qid, arxiv_id) in enumerate(papers, 1):
        log.info("[%d/%d] %s  arXiv:%s", i, len(papers), qid, arxiv_id)

        if not args.dry_run and already_uploaded(qid, lakefs_repo, lakefs_branch, lf_client):
            log.info("  → already uploaded, skipping")
            skipped_exists += 1
            continue

        try:
            markdown = fetch_and_convert(arxiv_id)
        except RuntimeError as exc:
            log.warning("  → fetch failed: %s", exc)
            skipped_error += 1
            time.sleep(1)
            continue

        if args.dry_run:
            path = component_path(qid)
            log.info("  → [dry-run] would upload %d chars to %s/%s", len(markdown), lakefs_branch, path)
            uploaded += 1
            continue

        try:
            lakefs_path = upload_markdown(
                qid,
                markdown,
                url=lakefs_url,
                user=lakefs_user,
                password=lakefs_password,
                repo=lakefs_repo,
                branch=lakefs_branch,
            )
            log.info("  → uploaded to %s", lakefs_path)
            uploaded += 1
        except Exception as exc:
            log.warning("  → lakeFS upload failed: %s", exc)
            skipped_error += 1

        time.sleep(0.5)  # be polite to arXiv

    print(
        f"\nDone. "
        f"{'Would upload' if args.dry_run else 'Uploaded'}: {uploaded}  "
        f"Already existed: {skipped_exists}  "
        f"Errors: {skipped_error}"
    )


if __name__ == "__main__":
    main()
