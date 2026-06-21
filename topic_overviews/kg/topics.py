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


_QUERY = """SELECT ?topic ?label ?desc {query_select} WHERE {{
  ?topic wdt:{p_inst} wd:{cls} .
  ?topic rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL {{ ?topic schema:description ?desc . FILTER(LANG(?desc) = "en") }}
  {query_optional}
}}"""


def load_registered_topics(
    sparql_endpoint: str,
    research_theme_qid: str,
    *,
    arxiv_query_property: str = "",
    run=run_sparql,
) -> list[Topic]:
    query_select = "?arxivQuery" if arxiv_query_property else ""
    query_optional = (
        f"OPTIONAL {{ ?topic wdt:{arxiv_query_property} ?arxivQuery. }}"
        if arxiv_query_property
        else ""
    )
    query = _QUERY.format(
        p_inst=P_INSTANCE_OF,
        cls=research_theme_qid,
        query_select=query_select,
        query_optional=query_optional,
    )
    rows = run(sparql_endpoint, query)
    return [
        Topic(
            qid=qid_from_uri(row["topic"]),
            label=row["label"],
            description=row.get("desc", ""),
            arxiv_query=row.get("arxivQuery", ""),
        )
        for row in rows
    ]
