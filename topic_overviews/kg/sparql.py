"""Thin SPARQL JSON-results helper."""
from __future__ import annotations

import requests

from ..http_utils import http_get


def run_sparql(endpoint: str, query: str, session=None) -> list[dict[str, str]]:
    session = session or requests.Session()
    resp = http_get(
        session,
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=120,
    )
    data = resp.json()
    return [
        {var: cell["value"] for var, cell in row.items()}
        for row in data["results"]["bindings"]
    ]
