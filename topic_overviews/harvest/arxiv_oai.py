"""Incremental arXiv harvest via OAI-PMH (ListRecords, arXiv metadata format)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator
from xml.etree import ElementTree as ET

import requests

OAI_URL = "http://export.arxiv.org/oai2"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}


@dataclass
class PaperRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    doi: str | None = None
    openalex_id: str = ""
    zbmath_id: str = ""          # zbMATH Open document ID string, e.g. "1541.68445" → P225
    zbmath_de_number: str = ""   # zbMATH DE Number (numeric), e.g. "7860651" → P1451
    zbmath_author_ids: list[tuple[str, str]] = field(default_factory=list)
    zbmath_keywords: list[str] = field(default_factory=list)
    journal_title: str = ""
    msc_codes: list[str] = field(default_factory=list)
    oa_status: str = ""                                          # OpenAlex oa_status string
    concepts: list[tuple[str, str]] = field(default_factory=list)  # OpenAlex (display_name, wikidata_qid)
    openalex_keywords: list[str] = field(default_factory=list)  # OpenAlex keyword display names
    license_url: str = ""                                        # license URL from zbMATH or Crossref

    @property
    def record_id(self) -> str:
        if self.arxiv_id:
            return self.arxiv_id
        if self.openalex_id:
            return f"openalex:{self.openalex_id}"
        if self.zbmath_id:
            return f"zbmath:{self.zbmath_id}"
        raise ValueError("PaperRecord has no identifier")


def _text(el, path: str) -> str | None:
    found = el.find(path, NS)
    if found is None or found.text is None:
        return None
    return " ".join(found.text.split())


def parse_oai_response(xml: str) -> tuple[list[PaperRecord], str | None]:
    root = ET.fromstring(xml)
    records: list[PaperRecord] = []
    for rec in root.findall(".//oai:record", NS):
        meta = rec.find(".//arxiv:arXiv", NS)
        if meta is None:
            continue  # deleted / no metadata
        authors = []
        for a in meta.findall("arxiv:authors/arxiv:author", NS):
            name = " ".join(
                p for p in [_text(a, "arxiv:forenames"), _text(a, "arxiv:keyname")] if p
            )
            if name:
                authors.append(name)
        records.append(
            PaperRecord(
                arxiv_id=_text(meta, "arxiv:id") or "",
                title=_text(meta, "arxiv:title") or "",
                abstract=_text(meta, "arxiv:abstract") or "",
                authors=authors,
                categories=(_text(meta, "arxiv:categories") or "").split(),
                published=_text(meta, "arxiv:created") or "",
                doi=_text(meta, "arxiv:doi") or (f"10.48550/arXiv.{_text(meta, 'arxiv:id')}" if _text(meta, "arxiv:id") else None),
            )
        )
    token_el = root.find(".//oai:resumptionToken", NS)
    token = token_el.text.strip() if token_el is not None and token_el.text and token_el.text.strip() else None
    return records, token


def fetch_records(
    from_date: str | None,
    set_spec: str = "math",
    session=None,
    sleep=time.sleep,
) -> Iterator[PaperRecord]:
    session = session or requests.Session()
    params: dict = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_spec}
    if from_date:
        params["from"] = from_date
    while True:
        resp = session.get(OAI_URL, params=params, timeout=60)
        resp.raise_for_status()
        records, token = parse_oai_response(resp.text)
        yield from records
        if not token:
            return
        params = {"verb": "ListRecords", "resumptionToken": token}
        sleep(3)  # arXiv OAI politeness
