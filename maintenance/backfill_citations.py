#!/usr/bin/env python3
"""Backfill P223 (cites work) citation links for papers already in the KG.

Finds all paper items linked to research themes via P265 (has part), resolves
their citations using zbMATH, OpenAlex, or Semantic Scholar (fallback), and
writes P223 claims for any cited papers that are already in the KG.

Papers that already have at least one P223 claim are skipped (idempotent).
P223 claims written via Semantic Scholar carry a P98=Q56440 reference block.

Usage:
    python maintenance/backfill_citations.py [--dry-run] [--limit N]

Environment variables:
    MEDIAWIKI_API_URL                    e.g. https://portal.mardi4nfdi.de/w/api.php
    MEDIAWIKI_BOT_USER
    MEDIAWIKI_BOT_PASSWORD
    SPARQL_ENDPOINT_URL                  e.g. https://query.portal.mardi4nfdi.de/sparql
    TOPIC_OVERVIEWS_RESEARCH_THEME_QID   research theme class QID, e.g. Q7266523
    TOPIC_OVERVIEWS_OPENALEX_EMAIL       optional, for OpenAlex polite pool
    TOPIC_OVERVIEWS_S2_API_KEY           optional, Semantic Scholar API key
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import requests

# Allow running from repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topic_overviews.kg.citation_linker import (
    fetch_zbmath_references,
    fetch_openalex_referenced_works,
    fetch_openalex_referenced_works_by_arxiv,
    resolve_qids_by_zbmath_doc_ids,
    resolve_qids_by_openalex_ids,
    fetch_s2_references,
    resolve_s2_references,
)
from topic_overviews.kg import model as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://query.portal.mardi4nfdi.de/sparql"
MEDIAWIKI_API = "https://portal.mardi4nfdi.de/w/api.php"
RESEARCH_THEME_QID = "Q7266523"


# ---------------------------------------------------------------------------
# KG helpers
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


def _qid_from_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]


def get_theme_paper_qids(sparql_endpoint: str, research_theme_qid: str, session: requests.Session) -> list[str]:
    """Return QIDs of all paper items linked from any research theme via P265."""
    query = f"""
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT DISTINCT ?paper WHERE {{
  ?theme wdt:{M.P_INSTANCE_OF} wd:{research_theme_qid} .
  ?theme wdt:{M.P_HAS_PART} ?paper .
}}
"""
    rows = _run_sparql(sparql_endpoint, query, session)
    return [_qid_from_uri(row["paper"]) for row in rows]


def get_paper_identifiers_batch(
    sparql_endpoint: str,
    paper_qids: list[str],
    session: requests.Session,
) -> dict[str, dict]:
    """Return {qid: {"zbmath_id": str, "openalex_id": str, "arxiv_id": str}} for a small batch.

    Fetches P225, P388, and P21 — no join with P223 so the query stays cheap.
    Caller is responsible for keeping batch size small (≤ 50).
    """
    if not paper_qids:
        return {}
    values = " ".join(f"wd:{q}" for q in paper_qids)
    query = f"""
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT ?paper ?zbmathId ?openalexId ?arxivId WHERE {{
  VALUES ?paper {{ {values} }}
  OPTIONAL {{ ?paper wdt:{M.P_ZBMATH_ID} ?zbmathId . }}
  OPTIONAL {{ ?paper wdt:{M.P_OPENALEX_ID} ?openalexId . }}
  OPTIONAL {{ ?paper wdt:{M.P_ARXIV_ID} ?arxivId . }}
}}
"""
    rows = _run_sparql(sparql_endpoint, query, session)
    result: dict[str, dict] = {}
    for row in rows:
        qid = _qid_from_uri(row["paper"])
        entry = result.setdefault(qid, {"zbmath_id": "", "openalex_id": "", "arxiv_id": ""})
        if row.get("zbmathId"):
            entry["zbmath_id"] = row["zbmathId"]
        if row.get("openalexId"):
            entry["openalex_id"] = row["openalexId"]
        if row.get("arxivId"):
            entry["arxiv_id"] = row["arxivId"]
    return result


def get_papers_with_citations_batch(
    sparql_endpoint: str,
    paper_qids: list[str],
    session: requests.Session,
) -> set[str]:
    """Return the subset of paper_qids that already have at least one P223 claim.

    Uses a simple triple pattern (no OPTIONAL, no multi-join) to stay cheap.
    Caller is responsible for keeping batch size small (≤ 50).
    """
    if not paper_qids:
        return set()
    values = " ".join(f"wd:{q}" for q in paper_qids)
    query = f"""
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT DISTINCT ?paper WHERE {{
  VALUES ?paper {{ {values} }}
  ?paper wdt:{M.P_CITES_WORK} [] .
}}
"""
    rows = _run_sparql(sparql_endpoint, query, session)
    return {_qid_from_uri(row["paper"]) for row in rows}


def _login(session: requests.Session, api: str, user: str, password: str) -> None:
    r = session.get(
        api,
        params={"action": "query", "meta": "tokens", "type": "login", "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    login_token = r.json()["query"]["tokens"]["logintoken"]
    r = session.post(api, data={
        "action": "login", "lgname": user, "lgpassword": password,
        "lgtoken": login_token, "format": "json",
    }, timeout=30)
    r.raise_for_status()
    if r.json().get("login", {}).get("result") != "Success":
        print(f"Login failed: {r.json()}", file=sys.stderr)
        sys.exit(1)


def _csrf(session: requests.Session, api: str) -> str:
    r = session.get(api, params={"action": "query", "meta": "tokens", "format": "json"}, timeout=30)
    r.raise_for_status()
    return r.json()["query"]["tokens"]["csrftoken"]


def write_citations(
    session: requests.Session,
    api: str,
    paper_qid: str,
    cited_qids: list[str],
    existing_cited: set[str],
    dry_run: bool,
    reference_qid: str = "",
) -> int:
    """Write P223 claims for cited_qids not already present. Returns number added.

    If ``reference_qid`` is given, a P98=<reference_qid> reference block is
    added to each new claim (used for Semantic Scholar-sourced citations).
    """
    to_add = [q for q in cited_qids if q not in existing_cited]
    if not to_add:
        return 0
    if dry_run:
        log.info("  [dry-run] would add %d P223 claim(s) to %s: %s", len(to_add), paper_qid, to_add)
        return len(to_add)
    reference_snaks = json.dumps({
        M.P_STATED_IN: [{
            "snaktype": "value",
            "property": M.P_STATED_IN,
            "datavalue": {"type": "wikibase-entityid",
                          "value": {"entity-type": "item", "id": reference_qid}},
        }]
    }) if reference_qid else None
    token = _csrf(session, api)
    added = 0
    for cited_qid in to_add:
        r = session.post(api, data={
            "action": "wbcreateclaim",
            "entity": paper_qid,
            "snaktype": "value",
            "property": M.P_CITES_WORK,
            "value": json.dumps({"entity-type": "item", "id": cited_qid}),
            "token": token,
            "format": "json",
            "bot": "1",
        }, timeout=30)
        r.raise_for_status()
        if reference_snaks:
            claim_guid = (r.json().get("claim") or {}).get("id")
            if claim_guid:
                session.post(api, data={
                    "action": "wbsetreference", "statement": claim_guid,
                    "snaks": reference_snaks,
                    "token": _csrf(session, api),
                    "format": "json", "bot": "1",
                }, timeout=30).raise_for_status()
        added += 1
    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Resolve citations but do not write to KG")
    parser.add_argument("--limit", type=int, default=0, metavar="N", help="Stop after processing N papers (0 = all)")
    args = parser.parse_args()

    api = os.environ.get("MEDIAWIKI_API_URL", MEDIAWIKI_API).strip()
    user = os.environ.get("MEDIAWIKI_BOT_USER", "").strip()
    password = os.environ.get("MEDIAWIKI_BOT_PASSWORD", "").strip()
    sparql_endpoint = os.environ.get("SPARQL_ENDPOINT_URL", SPARQL_ENDPOINT).strip()
    research_theme_qid = os.environ.get("TOPIC_OVERVIEWS_RESEARCH_THEME_QID", RESEARCH_THEME_QID).strip()
    openalex_email = os.environ.get("TOPIC_OVERVIEWS_OPENALEX_EMAIL", "").strip()
    s2_api_key = os.environ.get("TOPIC_OVERVIEWS_S2_API_KEY", "").strip()

    if not args.dry_run:
        missing = [name for name, val in [
            ("MEDIAWIKI_BOT_USER", user),
            ("MEDIAWIKI_BOT_PASSWORD", password),
        ] if not val]
        if missing:
            print(f"Missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)

    session = requests.Session()
    if not args.dry_run:
        _login(session, api, user, password)
        log.info("Logged in as %s", user)

    # --- Discover all theme-linked papers ---
    log.info("Querying papers linked from research themes (class %s)…", research_theme_qid)
    paper_qids = get_theme_paper_qids(sparql_endpoint, research_theme_qid, session)
    log.info("Found %d paper item(s)", len(paper_qids))

    if args.limit:
        paper_qids = paper_qids[: args.limit]
        log.info("Limiting to %d paper(s)", len(paper_qids))

    # --- Batch-fetch identifiers and existing citation state (two separate cheap queries) ---
    BATCH = 50  # keep VALUES clauses small to avoid SPARQL timeouts
    log.info("Fetching identifiers for all papers (batch size %d)…", BATCH)
    identifiers: dict[str, dict] = {}
    for i in range(0, len(paper_qids), BATCH):
        batch = paper_qids[i : i + BATCH]
        identifiers.update(get_paper_identifiers_batch(sparql_endpoint, batch, session))
        log.info("  identifiers %d/%d", min(i + BATCH, len(paper_qids)), len(paper_qids))
        time.sleep(0.5)

    log.info("Checking which papers already have P223 citations…")
    papers_with_citations: set[str] = set()
    for i in range(0, len(paper_qids), BATCH):
        batch = paper_qids[i : i + BATCH]
        papers_with_citations |= get_papers_with_citations_batch(sparql_endpoint, batch, session)
        log.info("  citation check %d/%d", min(i + BATCH, len(paper_qids)), len(paper_qids))
        time.sleep(0.5)

    already_done = len(papers_with_citations)
    log.info(
        "%d paper(s) already have P223 — skipping. %d to process.",
        already_done,
        len(paper_qids) - already_done,
    )

    # --- Process each paper ---
    total_added = 0
    processed = 0
    skipped_no_id = 0
    skipped_already = 0

    for paper_qid in paper_qids:
        info = identifiers.get(paper_qid, {})
        zbmath_id = info.get("zbmath_id", "")
        openalex_id = info.get("openalex_id", "")
        arxiv_id = info.get("arxiv_id", "")

        if paper_qid in papers_with_citations:
            skipped_already += 1
            continue

        if not zbmath_id and not openalex_id and not arxiv_id:
            log.debug("Skipping %s: no zbMATH, OpenAlex, or arXiv ID", paper_qid)
            skipped_no_id += 1
            continue

        processed += 1
        log.info(
            "[%d] Processing %s (zbMATH=%s OA=%s arXiv=%s)",
            processed, paper_qid, zbmath_id or "-", openalex_id or "-", arxiv_id or "-",
        )

        cited_qids: list[str] = []
        reference_qid = ""
        if zbmath_id:
            doc_ids = fetch_zbmath_references(zbmath_id, session=session)
            if doc_ids:
                cited_qids = resolve_qids_by_zbmath_doc_ids(doc_ids, sparql_endpoint, session)
        elif openalex_id:
            oa_work_ids = fetch_openalex_referenced_works(openalex_id, session=session, email=openalex_email)
            if oa_work_ids:
                cited_qids = resolve_qids_by_openalex_ids(oa_work_ids, sparql_endpoint, session)
        elif arxiv_id:
            oa_work_ids = fetch_openalex_referenced_works_by_arxiv(arxiv_id, session=session, email=openalex_email)
            if oa_work_ids:
                cited_qids = resolve_qids_by_openalex_ids(oa_work_ids, sparql_endpoint, session)

        # Semantic Scholar fallback when zbMATH/OpenAlex yielded nothing
        if not cited_qids:
            time.sleep(2)  # respect S2 unauthenticated rate limit
            s2_refs = fetch_s2_references(
                arxiv_id=arxiv_id, doi="", api_key=s2_api_key, session=session,
            )
            if s2_refs:
                cited_qids = resolve_s2_references(s2_refs, sparql_endpoint, api, session)
                if cited_qids:
                    reference_qid = M.Q_SEMANTIC_SCHOLAR

        if not cited_qids:
            log.info("  → no KG citations resolved")
            continue

        log.info("  → %d cited KG item(s) resolved%s", len(cited_qids),
                 " [via Semantic Scholar]" if reference_qid else "")
        added = write_citations(session, api, paper_qid, cited_qids, set(), args.dry_run,
                                reference_qid=reference_qid)
        total_added += added
        if added:
            log.info("  → wrote %d P223 claim(s)", added)

        time.sleep(0.5)  # be polite to both zbMATH/OpenAlex and the KG write API

    print(
        f"\nDone. Processed {processed} paper(s), "
        f"skipped {skipped_already} (already had citations), "
        f"skipped {skipped_no_id} (no identifier). "
        f"{'Would add' if args.dry_run else 'Added'} {total_added} P223 claim(s) in total."
    )


if __name__ == "__main__":
    main()
