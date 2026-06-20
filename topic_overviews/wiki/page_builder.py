"""Render topic overview pages and the index as MediaWiki wikitext (pure)."""
from __future__ import annotations

from ..kg.pagedata import TopicPageData

TOPIC_PAGE_PREFIX = "Topic:"
INDEX_PAGE_TITLE = "Topic overview"


def build_topic_page(data: TopicPageData) -> str:
    lines = [
        f"= {data.label} =",
        "",
        data.description,
        "",
        '{| class="wikitable sortable"',
        "! Title !! Authors !! Year !! arXiv",
    ]
    for p in data.papers:
        link = f"[https://arxiv.org/abs/{p.arxiv_id} {p.arxiv_id}]" if p.arxiv_id else ""
        lines.append("|-")
        lines.append(f"| {p.title} || {'; '.join(p.authors)} || {p.year} || {link}")
    lines.append("|}")
    return "\n".join(lines) + "\n"


def build_index_page(topics: list[TopicPageData]) -> str:
    lines = [f"= {INDEX_PAGE_TITLE} =", ""]
    for t in topics:
        lines.append(f"* [[{TOPIC_PAGE_PREFIX}{t.label}|{t.label}]] ({len(t.papers)} papers)")
    return "\n".join(lines) + "\n"
