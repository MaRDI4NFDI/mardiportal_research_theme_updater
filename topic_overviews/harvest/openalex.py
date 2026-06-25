"""Harvest recent works from OpenAlex by query string.

The query_str format mirrors URL params: ``search=mardi&filter=funders.id:f4320320879``.
Supported keys: ``search``, ``filter``, ``sort``.
The harvester appends ``from_publication_date:{cutoff}`` to the filter automatically
and uses cursor-based pagination. When sorted by ``publication_date:desc`` (the
default), iteration stops as soon as a result is older than the date window.
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Iterator
from urllib.parse import parse_qs

import requests

from .arxiv_oai import PaperRecord

OPENALEX_API_URL = "https://api.openalex.org/works"

log = logging.getLogger(__name__)


def _wikidata_qid(url: str) -> str:
    """Extract bare QID from a Wikidata entity URL, e.g. '.../wiki/Q2539' → 'Q2539'."""
    if not url:
        return ""
    part = url.rstrip("/").rsplit("/", 1)[-1]
    return part if part.startswith("Q") else ""


def _reconstruct_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    positions: dict[int, str] = {}
    for word, pos_list in inv.items():
        for p in pos_list:
            positions[p] = word
    return " ".join(positions[k] for k in sorted(positions))


def _strip_prefix(url: str, prefix: str) -> str:
    return url[len(prefix):] if url and url.startswith(prefix) else (url or "")


def _arxiv_id_from_work(w: dict) -> str:
    """Extract arXiv ID from ids.arxiv first, then fall back to locations."""
    ids = w.get("ids") or {}
    arxiv_raw = ids.get("arxiv", "") or ""
    if arxiv_raw:
        return _strip_prefix(arxiv_raw, "https://arxiv.org/abs/").strip()
    for loc in (w.get("locations") or []):
        url = (loc.get("landing_page_url") or "").strip()
        if url.startswith("https://arxiv.org/abs/"):
            return _strip_prefix(url, "https://arxiv.org/abs/").strip()
    return ""


def parse_works_page(works: list[dict]) -> list[PaperRecord]:
    records = []
    for w in works:
        ids = w.get("ids") or {}
        arxiv_id = _arxiv_id_from_work(w)
        openalex_raw = w.get("id", "") or ""
        openalex_id = openalex_raw.rstrip("/").rsplit("/", 1)[-1]
        doi_raw = (w.get("doi") or ids.get("doi") or "")
        doi = _strip_prefix(doi_raw, "https://doi.org/") or None
        authors = [
            (a.get("author") or {}).get("display_name") or a.get("raw_author_name", "")
            for a in (w.get("authorships") or [])
        ]
        categories: list[str] = []  # OpenAlex topics are not arXiv category codes
        primary = w.get("primary_location") or {}
        source = primary.get("source") or {}
        journal_title = (
            source.get("display_name", "").strip()
            if source.get("type") == "journal"
            else ""
        )
        oa = w.get("open_access") or {}
        oa_status = oa.get("oa_status") or ""
        concepts = [
            (c["display_name"], _wikidata_qid(c.get("wikidata") or ""))
            for c in (w.get("concepts") or [])
            if c.get("display_name") and c.get("score", 0) >= 0.3
        ]
        openalex_keywords = [
            k["display_name"] for k in (w.get("keywords") or [])
            if k.get("display_name")
        ]
        records.append(PaperRecord(
            arxiv_id=arxiv_id,
            title=(w.get("title") or "").strip(),
            abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
            authors=[a for a in authors if a],
            categories=categories,
            published=(w.get("publication_date") or "")[:10],
            doi=doi,
            openalex_id=openalex_id,
            journal_title=journal_title,
            oa_status=oa_status,
            concepts=concepts,
            openalex_keywords=openalex_keywords,
        ))
    return records


def fetch_openalex_records(
    query_str: str,
    since_days: int,
    *,
    email: str = "",
    page_size: int = 200,
    session=None,
    sleep=time.sleep,
    today: datetime.date | None = None,
) -> Iterator[PaperRecord]:
    """Yield PaperRecords from OpenAlex matching ``query_str`` published within
    ``since_days`` days. Uses cursor-based pagination; stops early when results
    are sorted by publication_date:desc and a result falls outside the window."""
    session = session or requests.Session()
    cutoff = (today or datetime.date.today()) - datetime.timedelta(days=since_days)
    cutoff_s = cutoff.isoformat()

    parsed = parse_qs(query_str, keep_blank_values=True)
    params: dict[str, str] = {k: v[0] for k, v in parsed.items()}

    sort = params.pop("sort", "publication_date:desc")
    early_stop = sort == "publication_date:desc"

    user_filter = params.pop("filter", "")
    date_filter = f"from_publication_date:{cutoff_s}"
    params["filter"] = f"{user_filter},{date_filter}" if user_filter else date_filter
    params["sort"] = sort
    params["per_page"] = str(page_size)
    if email:
        params["mailto"] = email

    cursor = "*"
    while True:
        params["cursor"] = cursor
        log.info(
            "Fetching OpenAlex results query=%r cursor=%r cutoff=%s",
            query_str, cursor, cutoff_s,
        )
        resp = session.get(OPENALEX_API_URL, params=dict(params), timeout=60)
        resp.raise_for_status()
        data = resp.json()
        works = data.get("results") or []
        log.info("Got %d OpenAlex results", len(works))
        if not works:
            return
        for record in parse_works_page(works):
            if early_stop and record.published and record.published < cutoff_s:
                log.info(
                    "Stopping OpenAlex at %s (%s): older than cutoff %s",
                    record.record_id if (record.arxiv_id or record.openalex_id) else "?",
                    record.published,
                    cutoff_s,
                )
                return
            yield record
        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            return
        cursor = next_cursor
        sleep(1)  # OpenAlex polite pool


def lookup_publication_date(
    doi: str | None = None,
    arxiv_id: str = "",
    *,
    session=None,
    email: str = "",
) -> str | None:
    """Return the full publication_date (YYYY-MM-DD) from OpenAlex for a paper
    identified by DOI or arXiv ID. Returns None if not found or on error.

    Used to enrich zbMATH records, which only carry a publication year.
    Tries DOI first (direct single-item lookup), then falls back to arXiv ID filter.
    """
    if not doi and not arxiv_id:
        return None
    sess = session or requests.Session()
    extra = {"mailto": email} if email else {}
    try:
        if doi:
            resp = sess.get(
                f"{OPENALEX_API_URL}/https://doi.org/{doi}",
                params=extra or None,
                timeout=20,
            )
            if resp.status_code == 200:
                date = (resp.json().get("publication_date") or "")[:10]
                if date:
                    return date
        if arxiv_id:
            params = {
                "filter": f"ids.arxiv:https://arxiv.org/abs/{arxiv_id}",
                "per_page": "1",
                **extra,
            }
            resp = sess.get(OPENALEX_API_URL, params=params, timeout=20)
            if resp.status_code == 200:
                works = resp.json().get("results") or []
                if works:
                    date = (works[0].get("publication_date") or "")[:10]
                    if date:
                        return date
    except Exception as exc:
        log.warning("OpenAlex date lookup failed (doi=%s arxiv=%s): %s", doi, arxiv_id, exc)
    return None


def lookup_openalex_enrichment(
    doi: str | None = None,
    arxiv_id: str = "",
    *,
    session=None,
    email: str = "",
) -> dict | None:
    """Fetch OpenAlex enrichment fields for a paper identified by DOI or arXiv ID.

    Returns a dict with keys ``published``, ``oa_status``, ``concepts``,
    ``openalex_keywords``, or None if the paper is not found.
    Makes a single API call reused for all fields.
    """
    if not doi and not arxiv_id:
        return None
    sess = session or requests.Session()
    extra = {"mailto": email} if email else {}
    work: dict | None = None
    try:
        if doi:
            resp = sess.get(
                f"{OPENALEX_API_URL}/https://doi.org/{doi}",
                params=extra or None,
                timeout=20,
            )
            if resp.status_code == 200:
                work = resp.json()
        if work is None and arxiv_id:
            params = {
                "filter": f"ids.arxiv:https://arxiv.org/abs/{arxiv_id}",
                "per_page": "1",
                **extra,
            }
            resp = sess.get(OPENALEX_API_URL, params=params, timeout=20)
            if resp.status_code == 200:
                results = resp.json().get("results") or []
                work = results[0] if results else None
    except Exception as exc:
        log.warning("OpenAlex enrichment lookup failed (doi=%s arxiv=%s): %s", doi, arxiv_id, exc)
        return None
    if work is None:
        return None
    oa = work.get("open_access") or {}
    concepts = [
        (c["display_name"], _wikidata_qid(c.get("wikidata") or ""))
        for c in (work.get("concepts") or [])
        if c.get("display_name") and c.get("score", 0) >= 0.3
    ]
    openalex_keywords = [
        k["display_name"] for k in (work.get("keywords") or [])
        if k.get("display_name")
    ]
    return {
        "published": (work.get("publication_date") or "")[:10],
        "oa_status": oa.get("oa_status") or "",
        "concepts": concepts,
        "openalex_keywords": openalex_keywords,
    }
