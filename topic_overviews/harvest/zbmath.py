"""Harvest recent documents from zbMATH Open by query string.

Query syntax follows zbMATH's own search:
  ``cc:68W25``   — MSC classification code
  ``ti:algebra`` — title keyword
  ``ab:mardi``   — abstract keyword

Since zbMATH stores only the publication *year*, the date window from
``since_days`` is approximated to the earliest calendar year covered:
``py:{cutoff_year}-{current_year}`` is appended to the query automatically
(unless the query already contains a ``py:`` filter).
"""
from __future__ import annotations

import datetime
import logging
import time
import urllib.parse
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
    records = []
    for d in docs:
        zbmath_id = str(d.get("document_id", "")).strip()
        title = (d.get("title") or "").strip()
        abstract = (d.get("abstract") or "").strip()
        authors = [
            b.get("name", "")
            for b in (d.get("author_biographies") or [])
            if b.get("name")
        ]
        classifications = [str(c) for c in (d.get("classifications") or [])]
        doi = (d.get("doi") or "").strip() or None
        links = d.get("links") or {}
        arxiv_id = (links.get("arxiv") or "").strip()
        year = str(d.get("year") or "").strip()
        published = f"{year}-01-01" if year else ""
        records.append(PaperRecord(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            categories=classifications,
            published=published,
            doi=doi,
            zbmath_id=zbmath_id,
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

    start = 0
    while True:
        encoded = urllib.parse.quote(query, safe=":+-")
        url = f"{ZBMATH_API_BASE}/document/_{encoded}"
        params = {"start": start, "count": page_size}
        log.info("Fetching zbMATH query=%r start=%d", query_str, start)
        resp = session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result") or {}
        docs = result.get("results") or []
        total = int(result.get("total") or 0)
        log.info("Got %d zbMATH results (total=%d)", len(docs), total)
        if not docs:
            return
        yield from parse_documents_page(docs)
        start += len(docs)
        if start >= total:
            return
        sleep(1)
