"""Harvest recent documents from zbMATH Open by query string.

Uses the zbMATH Open REST API search endpoint:
  GET https://api.zbmath.org/v1/document/_search?search_string=<query>

Query syntax follows zbMATH's own search:
  ``cc:68W25``   — MSC classification code
  ``ti:algebra`` — title keyword
  ``ab:mardi``   — abstract keyword
  ``arxiv:1403.6207`` — arXiv ID (used for single-paper lookup)

Since zbMATH stores only the publication *year*, the date window from
``since_days`` is approximated to the earliest calendar year covered:
``py:{cutoff_year}-{current_year}`` is appended to the query automatically
(unless the query already contains a ``py:`` filter).
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Iterator

import requests

from .arxiv_oai import PaperRecord

ZBMATH_API_BASE = "https://api.zbmath.org/v1"
PAGE_SIZE = 100

log = logging.getLogger(__name__)


def _year_range_filter(since_days: int, today: datetime.date) -> str:
    cutoff = today - datetime.timedelta(days=since_days)
    if cutoff.year == today.year:
        return f"py:{today.year}"
    return f"py:{cutoff.year}-{today.year}"


def parse_documents_page(docs: list[dict]) -> list[PaperRecord]:
    """Parse a list of document dicts from the zbMATH _search response."""
    records = []
    for d in docs:
        # P225: string document identifier, e.g. "1541.68445"
        zbmath_id = (d.get("identifier") or "").strip()
        # P1451: numeric DE Number, e.g. "7860651"
        zbmath_de_number = str(d.get("id") or "").strip()

        title_raw = d.get("title") or {}
        title = (
            title_raw.get("title") if isinstance(title_raw, dict) else str(title_raw or "")
        ).strip()

        contributors = d.get("contributors") or {}
        authors_raw = contributors.get("authors") or []
        authors = [a["name"] for a in authors_raw if a.get("name")]
        zbmath_author_ids = [
            (a["name"], a["codes"][0])
            for a in authors_raw
            if a.get("name") and a.get("codes")
        ]

        # MSC classification codes → P226 (not arXiv categories → P22)
        msc_codes = [m["code"] for m in (d.get("msc") or []) if m.get("code")]

        doi = None
        arxiv_id = ""
        for link in (d.get("links") or []):
            ltype = link.get("type", "")
            lid = (link.get("identifier") or "").strip()
            if ltype == "doi" and not doi:
                doi = lid or None
            elif ltype == "arxiv" and not arxiv_id:
                arxiv_id = lid

        year = str(d.get("year") or "").strip()
        published = f"{year}-01-01" if year else ""

        zbmath_keywords = [kw for kw in (d.get("keywords") or []) if kw]
        source = d.get("source") or {}
        series = source.get("series") or []
        journal_title = (series[0].get("title") or "").strip() if series else ""

        if not doi and arxiv_id:
            doi = f"10.48550/arXiv.{arxiv_id}"
        records.append(PaperRecord(
            arxiv_id=arxiv_id,
            title=title,
            abstract="",
            authors=authors,
            categories=[],       # zbMATH records carry no arXiv category codes
            published=published,
            doi=doi,
            zbmath_id=zbmath_id,
            zbmath_de_number=zbmath_de_number,
            zbmath_author_ids=zbmath_author_ids,
            zbmath_keywords=zbmath_keywords,
            journal_title=journal_title,
            msc_codes=msc_codes,
        ))
    return records


def fetch_zbmath_records(
    query_str: str,
    since_days: int,
    *,
    page_size: int = PAGE_SIZE,
    session=None,
    sleep=time.sleep,
    today: datetime.date | None = None,
) -> Iterator[PaperRecord]:
    """Yield PaperRecords from zbMATH Open matching ``query_str``.

    Appends a year-range filter derived from ``since_days`` unless the query
    already contains ``py:``. Paginates until all matching results are consumed.
    """
    session = session or requests.Session()
    _today = today or datetime.date.today()
    query = query_str.strip()
    if query and "py:" not in query:
        query = f"{query} {_year_range_filter(since_days, _today)}"

    page = 0
    fetched = 0
    while True:
        log.info("Fetching zbMATH query=%r page=%d", query_str, page)
        resp = session.get(
            f"{ZBMATH_API_BASE}/document/_search",
            params={"search_string": query, "page": page, "results_per_page": page_size},
            timeout=60,
        )
        if resp.status_code == 404:
            log.info("zbMATH query=%r returned 404 (no results for this period)", query_str)
            return
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status") or {}
        docs = data.get("result")
        if not isinstance(docs, list) or not docs:
            return
        total = int(status.get("nr_total_results") or 0)
        log.info("Got %d zbMATH results (total=%d, page=%d)", len(docs), total, page)
        yield from parse_documents_page(docs)
        fetched += len(docs)
        if fetched >= total:
            return
        page += 1
        sleep(1)


def lookup_by_arxiv_id(
    arxiv_id: str,
    *,
    session=None,
) -> PaperRecord | None:
    """Return the zbMATH record for ``arxiv_id``, or None if not found.

    Used for inline enrichment of arXiv/OpenAlex papers: adds P225 (zbMATH ID)
    and zbMATH author codes (for P676-based P16 resolution) to newly imported items.
    """
    if not arxiv_id:
        return None
    sess = session or requests.Session()
    try:
        resp = sess.get(
            f"{ZBMATH_API_BASE}/document/_search",
            params={"search_string": f"arxiv:{arxiv_id}", "page": 0, "results_per_page": 3},
            timeout=30,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        docs = resp.json().get("result") or []
        if not isinstance(docs, list):
            return None
        for record in parse_documents_page(docs):
            if record.arxiv_id == arxiv_id:
                return record
        return None
    except Exception as exc:
        log.warning("zbMATH lookup for arXiv:%s failed: %s", arxiv_id, exc)
        return None
