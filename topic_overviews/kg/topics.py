"""Read the topic registry from the KG (items that are instance-of overview-topic)."""
from __future__ import annotations

from dataclasses import dataclass

from .model import P_INSTANCE_OF, qid_from_uri
from .sparql import run_sparql


@dataclass
class Topic:
    qid: str
    label: str
    description: str


_QUERY = """SELECT ?topic ?label ?desc WHERE {{
  ?topic wdt:{p_inst} wd:{cls} .
  ?topic rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL {{ ?topic schema:description ?desc . FILTER(LANG(?desc) = "en") }}
}}"""


def load_registered_topics(sparql_endpoint: str, overview_topic_qid: str, run=run_sparql) -> list[Topic]:
    query = _QUERY.format(p_inst=P_INSTANCE_OF, cls=overview_topic_qid)
    rows = run(sparql_endpoint, query)
    return [
        Topic(
            qid=qid_from_uri(row["topic"]),
            label=row["label"],
            description=row.get("desc", ""),
        )
        for row in rows
    ]
