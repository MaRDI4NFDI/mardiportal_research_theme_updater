#!/usr/bin/env python3
"""Extract mathematical formulas from arXiv paper Markdown stored in lakeFS.

Scans the lakeFS repository for available .md.txt files, displays the
corresponding papers (QID + title), and — when a QID is given — sends the
fulltext to an LLM to extract all formulas as structured JSON.

Usage:
    python maintenance/extract_formulas.py [QID]

    Without QID: list all available papers and exit.
    With QID:    extract formulas and write maintenance/{QID}_formulas.json

Environment variables:
    LAKEFS_URL          (default: https://lake-bioinfmed.zib.de)
    LAKEFS_USER         lakeFS access key ID
    LAKEFS_PASSWORD     lakeFS secret access key
    LAKEFS_REPO         (default: mardi-fdo-data)
    LAKEFS_BRANCH       (default: main)
    SPARQL_ENDPOINT_URL (default: https://query.portal.mardi4nfdi.de/sparql)
    OPENROUTER_API_KEY  API key for openrouter.ai
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import lakefs
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topic_overviews.lakefs_upload import component_path
from topic_overviews.kg import model as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_LAKEFS_URL = "https://lake-bioinfmed.zib.de"
_LAKEFS_REPO = "mardi-fdo-data"
_LAKEFS_BRANCH = "main"
_SPARQL_ENDPOINT = "https://query.portal.mardi4nfdi.de/sparql"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "z-ai/glm-4.7-flash"


def list_lakefs_papers(
    url: str,
    user: str,
    password: str,
    repo: str,
    branch: str,
) -> list[str]:
    """Return deduplicated list of QIDs that have a .md.txt file in lakeFS."""
    client = lakefs.Client(host=url, username=user, password=password)
    objects = lakefs.repository(repo, client=client).branch(branch).objects(
        max_amount=100_000
    )
    seen: set[str] = set()
    result: list[str] = []
    for obj in objects:
        parts = obj.path.split("/")
        # path: pp/qq/rr/QXXXX/fulltext/QXXXX.md.txt  → 6 parts
        if len(parts) < 6 or not obj.path.endswith(".md.txt"):
            continue
        qid = parts[3]
        if qid not in seen:
            seen.add(qid)
            result.append(qid)
    return result


def get_paper_titles(
    qids: list[str],
    endpoint: str,
    session: requests.Session,
) -> dict[str, str]:
    """Return {QID: title} for the given QIDs. Missing items map to empty string."""
    if not qids:
        return {}
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT ?paper ?title WHERE {{
  VALUES ?paper {{ {values} }}
  OPTIONAL {{ ?paper wdt:{M.P_TITLE} ?title }}
}}
"""
    resp = session.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=120,
    )
    resp.raise_for_status()
    rows = resp.json()["results"]["bindings"]
    result: dict[str, str] = {q: "" for q in qids}
    for row in rows:
        qid = row["paper"]["value"].rstrip("/").rsplit("/", 1)[-1]
        result[qid] = row.get("title", {}).get("value", "")
    return result


def main():
    pass


if __name__ == "__main__":
    main()
