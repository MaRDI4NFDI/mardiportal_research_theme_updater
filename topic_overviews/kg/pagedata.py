"""Read per-topic paper lists from the KG for page generation."""
from __future__ import annotations

from dataclasses import dataclass

from . import model as M
from .sparql import run_sparql
from .topics import Topic


@dataclass
class PaperEntry:
    title: str
    authors: list[str]
    year: str
    arxiv_id: str


@dataclass
class TopicPageData:
    qid: str
    label: str
    description: str
    papers: list[PaperEntry]


_QUERY = """SELECT ?title ?year ?arxiv (GROUP_CONCAT(?author; SEPARATOR="; ") AS ?authors) WHERE {{
  wd:{topic} wdt:{p_haspart} ?paper .
  ?paper wdt:{p_title} ?title .
  OPTIONAL {{ ?paper wdt:{p_date} ?year }}
  OPTIONAL {{ ?paper wdt:{p_arxiv} ?arxiv }}
  OPTIONAL {{ ?paper wdt:{p_author} ?author }}
}} GROUP BY ?title ?year ?arxiv ORDER BY DESC(?year)"""


def fetch_topic_page_data(sparql_endpoint: str, topic: Topic, run=run_sparql) -> TopicPageData:
    query = _QUERY.format(
        p_haspart=M.P_HAS_PART, topic=topic.qid, p_title=M.P_TITLE,
        p_date=M.P_PUBLICATION_DATE, p_arxiv=M.P_ARXIV_ID, p_author=M.P_AUTHOR_NAME_STRING,
    )
    rows = run(sparql_endpoint, query)
    papers = [
        PaperEntry(
            title=row.get("title", ""),
            authors=[a for a in row.get("authors", "").split("; ") if a],
            year=(row.get("year", "") or "")[:4],
            arxiv_id=row.get("arxiv", ""),
        )
        for row in rows
    ]
    return TopicPageData(qid=topic.qid, label=topic.label, description=topic.description, papers=papers)
