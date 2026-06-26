"""Read the topic registry from the KG (items that are instance-of the
``research theme`` class). Each such item is one research theme to classify into."""
from __future__ import annotations

from dataclasses import dataclass

from .model import P_INSTANCE_OF, qid_from_uri
from .sparql import run_sparql


@dataclass
class Topic:
    qid: str
    label: str
    description: str
    arxiv_query: str = ""
    openalex_query: str = ""
    zbmath_query: str = ""
    since_days: int | None = None
    auto_classify_keywords: list[str] = None

    def __post_init__(self):
        if self.auto_classify_keywords is None:
            self.auto_classify_keywords = []

    def matches_keywords(self, title: str, abstract: str) -> str | None:
        """Return the first matching keyword if title or abstract contains it, else None."""
        text = (title + " " + abstract).lower()
        for kw in self.auto_classify_keywords:
            if kw.lower() in text:
                return kw
        return None


_QUERY = """SELECT ?topic ?label ?desc {query_select} WHERE {{
  ?topic wdt:{p_inst} wd:{cls} .
  ?topic rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL {{ ?topic schema:description ?desc . FILTER(LANG(?desc) = "en") }}
  {query_optionals}
}}"""

_KEYWORDS_QUERY = """SELECT ?topic ?keyword WHERE {{
  ?topic wdt:{p_inst} wd:{cls} .
  ?topic wdt:{p_kw} ?keyword .
}}"""


def load_registered_topics(
    sparql_endpoint: str,
    research_theme_qid: str,
    *,
    arxiv_query_property: str = "",
    openalex_query_property: str = "",
    zbmath_query_property: str = "",
    since_days_property: str = "",
    auto_classify_keywords_property: str = "",
    run=run_sparql,
) -> list[Topic]:
    select_parts = []
    optional_parts = []
    if arxiv_query_property:
        select_parts.append("?arxivQuery")
        optional_parts.append(
            f"OPTIONAL {{ ?topic wdt:{arxiv_query_property} ?arxivQuery. }}"
        )
    if openalex_query_property:
        select_parts.append("?openalexQuery")
        optional_parts.append(
            f"OPTIONAL {{ ?topic wdt:{openalex_query_property} ?openalexQuery. }}"
        )
    if zbmath_query_property:
        select_parts.append("?zbmathQuery")
        optional_parts.append(
            f"OPTIONAL {{ ?topic wdt:{zbmath_query_property} ?zbmathQuery. }}"
        )
    if since_days_property:
        select_parts.append("?sinceDays")
        optional_parts.append(
            f"OPTIONAL {{ ?topic wdt:{since_days_property} ?sinceDays. }}"
        )

    query = _QUERY.format(
        p_inst=P_INSTANCE_OF,
        cls=research_theme_qid,
        query_select=" ".join(select_parts),
        query_optionals="\n  ".join(optional_parts),
    )
    rows = run(sparql_endpoint, query)

    keywords_by_qid: dict[str, list[str]] = {}
    if auto_classify_keywords_property:
        kw_rows = run(
            sparql_endpoint,
            _KEYWORDS_QUERY.format(
                p_inst=P_INSTANCE_OF,
                cls=research_theme_qid,
                p_kw=auto_classify_keywords_property,
            ),
        )
        for row in kw_rows:
            qid = qid_from_uri(row["topic"])
            keywords_by_qid.setdefault(qid, []).append(row["keyword"])

    return [
        Topic(
            qid=qid_from_uri(row["topic"]),
            label=row["label"],
            description=row.get("desc", ""),
            arxiv_query=row.get("arxivQuery", ""),
            openalex_query=row.get("openalexQuery", ""),
            zbmath_query=row.get("zbmathQuery", ""),
            since_days=int(row["sinceDays"]) if row.get("sinceDays") else None,
            auto_classify_keywords=keywords_by_qid.get(qid_from_uri(row["topic"]), []),
        )
        for row in rows
    ]
