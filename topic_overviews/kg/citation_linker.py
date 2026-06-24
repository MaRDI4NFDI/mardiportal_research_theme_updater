"""Resolve citation links between paper items in the MaRDI KG.

Strategy by source:
- zbMATH papers: fetch references via zbMATH detail API → resolve via P1451 (zbMATH DE Number)
- OpenAlex papers: fetch referenced_works from OpenAlex → resolve via P388 (OpenAlex ID)
- arXiv-only papers that were zbMATH-enriched: use the zbMATH path
- Fallback (no P223 written yet): Semantic Scholar API → resolve via P21/P27/P388/title search
"""
from __future__ import annotations

import logging

import requests

from . import model as M
from .sparql import run_sparql

ZBMATH_API_BASE = "https://api.zbmath.org/v1"
OPENALEX_API_URL = "https://api.openalex.org/works"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
MARDI_API_URL = "https://portal.mardi4nfdi.de/w/api.php"

log = logging.getLogger(__name__)


def fetch_zbmath_references(zbmath_id: str, *, session=None) -> list[int]:
    """Return zbMATH DE Numbers (numeric document_ids) referenced by this paper.

    Calls the zbMATH detail endpoint and extracts references[].zbmath.document_id.
    Returns an empty list on error or if no structured references are present.
    """
    if not zbmath_id:
        return []
    sess = session or requests.Session()
    try:
        resp = sess.get(f"{ZBMATH_API_BASE}/document/{zbmath_id}", timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        result = resp.json().get("result") or {}
        refs = result.get("references") or []
        doc_ids = [
            ref["zbmath"]["document_id"]
            for ref in refs
            if isinstance(ref.get("zbmath"), dict) and ref["zbmath"].get("document_id")
        ]
        log.debug("zbMATH %s: %d reference(s) with document_id", zbmath_id, len(doc_ids))
        return doc_ids
    except Exception as exc:
        log.warning("zbMATH references fetch for %s failed: %s", zbmath_id, exc)
        return []


def fetch_openalex_referenced_works(openalex_id: str, *, session=None, email: str = "") -> list[str]:
    """Return OpenAlex work IDs (e.g. 'W2741809809') cited by this paper.

    Fetches the single-work endpoint and extracts referenced_works.
    Returns an empty list for papers < ~6 months old (OpenAlex has no data yet) or on error.
    """
    if not openalex_id:
        return []
    sess = session or requests.Session()
    bare_id = openalex_id if openalex_id.startswith("W") else f"W{openalex_id}"
    params = {"mailto": email} if email else {}
    try:
        resp = sess.get(f"{OPENALEX_API_URL}/{bare_id}", params=params or None, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        refs = resp.json().get("referenced_works") or []
        # Normalize: strip URL prefix, keep bare work ID ("W1234567")
        work_ids = [r.rstrip("/").rsplit("/", 1)[-1] for r in refs if r]
        log.debug("OpenAlex %s: %d referenced_work(s)", openalex_id, len(work_ids))
        return work_ids
    except Exception as exc:
        log.warning("OpenAlex referenced_works fetch for %s failed: %s", openalex_id, exc)
        return []


def fetch_openalex_referenced_works_by_arxiv(arxiv_id: str, *, session=None, email: str = "") -> list[str]:
    """Return OpenAlex work IDs cited by an arXiv paper, looked up via its DOI.

    Uses the canonical arXiv DOI (10.48550/arXiv.<id>) as the OpenAlex lookup key.
    Returns an empty list if the paper is not indexed by OpenAlex yet or on error.
    """
    if not arxiv_id:
        return []
    sess = session or requests.Session()
    doi_url = f"https://doi.org/10.48550/arXiv.{arxiv_id}"
    params = {"mailto": email} if email else {}
    try:
        resp = sess.get(f"{OPENALEX_API_URL}/{doi_url}", params=params or None, timeout=30)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        refs = resp.json().get("referenced_works") or []
        work_ids = [r.rstrip("/").rsplit("/", 1)[-1] for r in refs if r]
        log.debug("OpenAlex (arXiv:%s): %d referenced_work(s)", arxiv_id, len(work_ids))
        return work_ids
    except Exception as exc:
        log.warning("OpenAlex referenced_works (arXiv:%s) failed: %s", arxiv_id, exc)
        return []


def _batch_sparql_lookup(prop: str, values: list[str], sparql_endpoint: str, session=None) -> dict[str, str]:
    """Return {value → QID} for KG items that have `prop` set to one of `values`.

    Uses a VALUES clause for efficient batch lookup.
    """
    if not values:
        return {}
    values_clause = " ".join(f'"{v}"' for v in values)
    query = f"""
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT ?item ?value WHERE {{
  VALUES ?value {{ {values_clause} }}
  ?item wdt:{prop} ?value .
}}
"""
    try:
        rows = run_sparql(sparql_endpoint, query, session)
        result: dict[str, str] = {}
        for row in rows:
            item_uri = row.get("item", "")
            value = row.get("value", "")
            if item_uri and value:
                qid = item_uri.rstrip("/").rsplit("/", 1)[-1]
                result[value] = qid
        return result
    except Exception as exc:
        log.warning("Batch SPARQL lookup (prop=%s, n=%d) failed: %s", prop, len(values), exc)
        return {}


def resolve_qids_by_zbmath_doc_ids(
    doc_ids: list[int],
    sparql_endpoint: str,
    session=None,
) -> list[str]:
    """Return KG QIDs for papers identified by zbMATH DE Number (P1451)."""
    if not doc_ids:
        return []
    mapping = _batch_sparql_lookup(M.P_ZBMATH_DE_NUMBER, [str(d) for d in doc_ids], sparql_endpoint, session)
    found = len(mapping)
    log.info("Citation resolve (zbMATH): %d/%d doc_ids matched in KG", found, len(doc_ids))
    return list(mapping.values())


def resolve_qids_by_openalex_ids(
    openalex_ids: list[str],
    sparql_endpoint: str,
    session=None,
) -> list[str]:
    """Return KG QIDs for papers identified by OpenAlex work ID (P388)."""
    if not openalex_ids:
        return []
    mapping = _batch_sparql_lookup(M.P_OPENALEX_ID, openalex_ids, sparql_endpoint, session)
    found = len(mapping)
    log.info("Citation resolve (OpenAlex): %d/%d work IDs matched in KG", found, len(openalex_ids))
    return list(mapping.values())


# ---------------------------------------------------------------------------
# Semantic Scholar fallback
# ---------------------------------------------------------------------------

def fetch_s2_references(
    arxiv_id: str = "",
    doi: str = "",
    *,
    api_key: str = "",
    session=None,
) -> list[dict]:
    """Fetch references for a paper from the Semantic Scholar API.

    Looks up by arXiv ID first, then DOI. Returns a list of dicts, each with
    keys ``arxiv_id``, ``doi``, ``openalex_id``, and ``title`` (any may be empty).
    Returns an empty list on error or if the paper is not found.
    """
    sess = session or requests.Session()
    headers = {"x-api-key": api_key} if api_key else {}

    paper_id = f"arXiv:{arxiv_id}" if arxiv_id else (f"DOI:{doi}" if doi else "")
    if not paper_id:
        return []

    try:
        resp = sess.get(
            f"{S2_API_BASE}/paper/{paper_id}/references",
            params={"fields": "externalIds,title", "limit": 500},
            headers=headers,
            timeout=30,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        refs = []
        for entry in resp.json().get("data", []):
            cited = entry.get("citedPaper") or {}
            ext = cited.get("externalIds") or {}
            refs.append({
                "arxiv_id": ext.get("ArXiv", ""),
                "doi": ext.get("DOI", ""),
                "openalex_id": ext.get("MAG", "") or "",  # S2 uses MAG id ≠ OpenAlex W-id
                "title": cited.get("title", ""),
            })
        log.debug("Semantic Scholar %s: %d reference(s)", paper_id, len(refs))
        return refs
    except Exception as exc:
        log.warning("Semantic Scholar references fetch for %s failed: %s", paper_id, exc)
        return []


def _search_kg_by_title(title: str, mediawiki_api_url: str, session=None) -> str:
    """Return a KG QID for a scholarly article whose label exactly matches title, or ''."""
    if not title or not mediawiki_api_url:
        return ""
    sess = session or requests.Session()
    try:
        resp = sess.get(
            mediawiki_api_url,
            params={"action": "wbsearchentities", "search": title,
                    "language": "en", "type": "item", "limit": 5, "format": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        for hit in resp.json().get("search", []):
            if hit.get("label", "").lower() == title.lower():
                return hit["id"]
    except Exception as exc:
        log.warning("KG title search for %r failed: %s", title[:60], exc)
    return ""


def resolve_s2_references(
    refs: list[dict],
    sparql_endpoint: str,
    mediawiki_api_url: str = "",
    session=None,
) -> list[str]:
    """Resolve a Semantic Scholar reference list to KG QIDs.

    Tries in order: arXiv ID (P21) → DOI (P27) → title search.
    Returns a deduplicated list of matched QIDs.
    """
    if not refs:
        return []

    found: dict[str, str] = {}  # QID → canonical match value (for dedup logging)

    # Batch identifier lookups
    arxiv_ids = [r["arxiv_id"] for r in refs if r["arxiv_id"]]
    dois = [r["doi"] for r in refs if r["doi"]]

    if arxiv_ids:
        for val, qid in _batch_sparql_lookup(M.P_ARXIV_ID, arxiv_ids, sparql_endpoint, session).items():
            found[qid] = f"arXiv:{val}"
    if dois:
        for val, qid in _batch_sparql_lookup(M.P_DOI, dois, sparql_endpoint, session).items():
            if qid not in found:
                found[qid] = f"DOI:{val}"

    matched_by_id = set(found.keys())

    # Title search for refs still unresolved
    resolved_arxiv = {r["arxiv_id"] for r in refs if r["arxiv_id"] and any(
        f"arXiv:{r['arxiv_id']}" == v for v in found.values()
    )}
    resolved_dois = {r["doi"] for r in refs if r["doi"] and any(
        f"DOI:{r['doi']}" == v for v in found.values()
    )}

    for ref in refs:
        if ref["arxiv_id"] in resolved_arxiv or ref["doi"] in resolved_dois:
            continue
        if not ref["title"]:
            continue
        qid = _search_kg_by_title(ref["title"], mediawiki_api_url, session)
        if qid and qid not in found:
            found[qid] = f"title:{ref['title'][:60]}"

    log.info(
        "Citation resolve (S2): %d matched (%d by id, %d by title)",
        len(found), len(matched_by_id), len(found) - len(matched_by_id),
    )
    return list(found.keys())
