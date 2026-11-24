"""Harvest recent arXiv papers by keyword query via the arXiv API.

Unlike the OAI harvester (which returns records by last-modified datestamp and is
dominated by re-indexed old papers), this queries the arXiv *search* API sorted
by submission date descending, so it yields the genuinely newest papers matching
a keyword query, and stops once papers fall outside the date window.
"""
from __future__ import annotations

import datetime
import logging
import re
import time
from typing import Iterator
from xml.etree import ElementTree as ET

import requests

from .arxiv_oai import PaperRecord

ARXIV_API_URL = "http://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


# arXiv throttles with 429; 502/503/504 are outages. Both are retried.
_RETRY_STATUS = (429, 502, 503, 504)
_MAX_ATTEMPTS = 5
# Pauses between the attempts: ten minutes in total, then the query is given up
# (it is harvested again on the next run).
_BACKOFF = (60, 120, 180, 240)
_MAX_RETRY_AFTER = 300  # cap on a server-requested Retry-After wait


def _retry_after(resp) -> int | None:
    """Seconds from a numeric Retry-After header, capped; None if absent or unusable."""
    try:
        seconds = int(float(resp.headers.get("Retry-After", "")))
    except (AttributeError, TypeError, ValueError):
        return None
    return min(seconds, _MAX_RETRY_AFTER) if seconds > 0 else None


def _norm(text: str | None) -> str:
    return " ".join(text.split()) if text else ""


def parse_atom(xml: str) -> list[PaperRecord]:
    root = ET.fromstring(xml)
    records: list[PaperRecord] = []
    for e in root.findall("a:entry", NS):
        raw_id = e.findtext("a:id", default="", namespaces=NS)
        arxiv_id = re.sub(r"^https?://arxiv\.org/abs/", "", raw_id)
        arxiv_id = re.sub(r"v\d+$", "", arxiv_id).strip()
        authors = [
            _norm(a.findtext("a:name", default="", namespaces=NS))
            for a in e.findall("a:author", NS)
        ]
        records.append(
            PaperRecord(
                arxiv_id=arxiv_id,
                title=_norm(e.findtext("a:title", default="", namespaces=NS)),
                abstract=_norm(e.findtext("a:summary", default="", namespaces=NS)),
                authors=[a for a in authors if a],
                categories=[c.get("term") for c in e.findall("a:category", NS) if c.get("term")],
                published=(e.findtext("a:published", default="", namespaces=NS) or "")[:10],
                doi=e.findtext("arxiv:doi", default=None, namespaces=NS) or f"10.48550/arXiv.{arxiv_id}",
            )
        )
    return records


def search_records(
    query: str,
    since_days: int,
    *,
    page_size: int = 100,
    session=None,
    sleep=time.sleep,
    today: datetime.date | None = None,
) -> Iterator[PaperRecord]:
    """Yield newest-first arXiv papers matching ``query`` submitted within the
    last ``since_days`` days. Stops as soon as a paper is older than the window
    (results are sorted by submission date descending)."""
    session = session or requests.Session()
    log = logging.getLogger(__name__)
    cutoff = (today or datetime.date.today()) - datetime.timedelta(days=since_days)
    cutoff_s = cutoff.isoformat()
    start = 0
    while True:
        log.info(
            "Fetching arXiv results for query=%r start=%d page_size=%d cutoff=%s",
            query,
            start,
            page_size,
            cutoff_s,
        )
        for attempt in range(_MAX_ATTEMPTS):
            resp = session.get(
                ARXIV_API_URL,
                params={
                    "search_query": query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                    "start": start,
                    "max_results": page_size,
                },
                timeout=60,
            )
            if resp.status_code in _RETRY_STATUS:
                if attempt == _MAX_ATTEMPTS - 1:
                    # Out of attempts: fail now instead of waiting once more.
                    log.warning(
                        "arXiv returned %d for query=%r (attempt %d/%d); giving up",
                        resp.status_code, query, attempt + 1, _MAX_ATTEMPTS,
                    )
                    break
                wait = _retry_after(resp) or _BACKOFF[attempt]
                log.warning(
                    "arXiv returned %d for query=%r (attempt %d/%d); waiting %ds",
                    resp.status_code, query, attempt + 1, _MAX_ATTEMPTS, wait,
                )
                sleep(wait)
                continue
            break
        resp.raise_for_status()
        records = parse_atom(resp.text)
        log.info("Got %d arXiv results for query=%r", len(records), query)
        if not records:
            return
        for r in records:
            if r.published and r.published < cutoff_s:
                log.info(
                    "Stopping arXiv query=%r at %s because it is older than cutoff %s",
                    query,
                    r.arxiv_id,
                    cutoff_s,
                )
                return  # everything after this is older too
            yield r
        start += page_size
        sleep(3)  # arXiv API politeness
