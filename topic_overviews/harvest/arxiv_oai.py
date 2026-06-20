"""Incremental arXiv harvest via OAI-PMH (ListRecords, arXiv metadata format)."""
from __future__ import annotations

import time
from dataclasses import dataclass
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
                doi=_text(meta, "arxiv:doi"),
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
