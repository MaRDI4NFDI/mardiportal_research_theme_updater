#!/usr/bin/env python3
"""Extract mathematical formulas from arXiv paper Markdown stored in lakeFS.

Scans the lakeFS repository for available .md.txt files, displays the
corresponding papers (QID + title), and — when a QID is given — sends the
fulltext to an LLM to extract all formulas as structured JSON.

Usage:
    python maintenance/extract_formulas.py [QID]

    Without QID: list all available papers and exit.
    With QID:    extract formulas and write maintenance/{QID}_formulas.json

Environment variables:
    LAKEFS_URL          (default: https://lake-bioinfmed.zib.de)
    LAKEFS_USER         lakeFS access key ID
    LAKEFS_PASSWORD     lakeFS secret access key
    LAKEFS_REPO         (default: mardi-fdo-data)
    LAKEFS_BRANCH       (default: main)
    SPARQL_ENDPOINT_URL (default: https://query.portal.mardi4nfdi.de/sparql)
    OPENROUTER_API_KEY  API key for openrouter.ai
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import lakefs
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topic_overviews.lakefs_upload import component_path
from topic_overviews.kg import model as M

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_LAKEFS_URL = "https://lake-bioinfmed.zib.de"
_LAKEFS_REPO = "mardi-fdo-data"
_LAKEFS_BRANCH = "main"
_SPARQL_ENDPOINT = "https://query.portal.mardi4nfdi.de/sparql"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_MODEL = "z-ai/glm-4.7-flash"


def list_lakefs_papers(
    url: str,
    user: str,
    password: str,
    repo: str,
    branch: str,
) -> list[str]:
    """Return deduplicated list of QIDs that have a .md.txt file in lakeFS."""
    client = lakefs.Client(host=url, username=user, password=password)
    objects = lakefs.repository(repo, client=client).branch(branch).objects(
        max_amount=100_000
    )
    seen: set[str] = set()
    result: list[str] = []
    for obj in objects:
        parts = obj.path.split("/")
        # path: pp/qq/rr/QXXXX/fulltext/QXXXX.md.txt  → 6 parts
        if len(parts) < 6 or not obj.path.endswith(".md.txt"):
            continue
        qid = parts[3]
        if qid not in seen:
            seen.add(qid)
            result.append(qid)
    return result


def get_paper_titles(
    qids: list[str],
    endpoint: str,
    session: requests.Session,
) -> dict[str, str]:
    """Return {QID: title} for the given QIDs. Missing items map to empty string."""
    if not qids:
        return {}
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT ?paper ?title WHERE {{
  VALUES ?paper {{ {values} }}
  OPTIONAL {{ ?paper wdt:{M.P_TITLE} ?title }}
}}
"""
    resp = session.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=120,
    )
    resp.raise_for_status()
    rows = resp.json()["results"]["bindings"]
    result: dict[str, str] = {q: "" for q in qids}
    for row in rows:
        qid = row["paper"]["value"].rstrip("/").rsplit("/", 1)[-1]
        result[qid] = row.get("title", {}).get("value", "")
    return result


_SCHEMA_EXAMPLE = """{
  "item_type": "mathematical object",
  "label": "human-readable label",
  "defining_formula": "normalized LaTeX",
  "description_long": "full explanation of what the formula means",
  "formula_type": "equation",
  "classification": "standard",
  "conditions": "conditions/assumptions, or empty string",
  "is_numbered": true,
  "equation_number": "(3.2)",
  "symbols": [
    {
      "symbol": "LaTeX symbol",
      "represents": "meaning",
      "type": "variable",
      "domain": "\\\\mathbb{R}"
    }
  ],
  "notation_variants": [],
  "related_concepts": ["concept1"],
  "msc_codes_suggested": ["65N30"],
  "cross_references": {
    "dlmf": "",
    "wikidata_qid": ""
  },
  "source": {
    "section": "section name",
    "formula_as_found": "verbatim LaTeX from the paper",
    "source_text": "surrounding sentence(s)"
  },
  "confidence": {
    "formula_extraction": 0.99,
    "classification": 0.95,
    "description": 0.96
  },
  "review_status": "unreviewed"
}"""

_SYSTEM_PROMPT = f"""You are a mathematical knowledge extraction assistant.
Your task is to extract every distinct mathematical formula from the provided paper Markdown and return them as a JSON array.

Rules:
- Output ONLY a valid JSON array. No prose, no markdown fences, no explanation.
- Each element of the array represents one formula and must match this schema:
{_SCHEMA_EXAMPLE}

Field constraints:
- formula_type: one of "equation", "inequality", "identity", "definition", "theorem", "lemma", "approximation", "bound", "recurrence"
- classification: one of "standard" (well-known formula), "novel" (introduced in this paper), "variant", "generalization"
- symbol.type: one of "variable", "constant", "operator", "function", "set", "index"
- conditions: empty string when none are stated
- is_numbered: true if the paper assigns an equation number like (3.2), else false
- equation_number: the number string if is_numbered, else empty string
- msc_codes_suggested: MSC codes you can infer from context; empty list if uncertain
- cross_references.dlmf: DLMF section/URL if you recognise the formula; empty string otherwise
- cross_references.wikidata_qid: Wikidata QID (e.g. Q11518) if recognisable; empty string otherwise
- confidence values are your own calibrated estimates (0.0–1.0)
- review_status is always "unreviewed"

If the paper contains no mathematical formulas, return an empty JSON array: []
"""


def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text.

    Handles responses that may have markdown fences or leading prose.
    """
    text = text.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    # Find first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def extract_formulas_llm(markdown: str, api_key: str) -> list[dict]:
    """Send *markdown* to OpenRouter and return extracted formulas as a list of dicts.

    Raises ValueError if the response cannot be parsed as a JSON array.
    """
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        "temperature": 0.1,
    }
    resp = requests.post(
        _OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _extract_json_array(content)


def download_markdown(
    qid: str,
    url: str,
    user: str,
    password: str,
    repo: str,
    branch: str,
) -> str:
    """Download the .md.txt content for *qid* from lakeFS and return as string.

    Raises FileNotFoundError if the object does not exist.
    """
    client = lakefs.Client(host=url, username=user, password=password)
    path = component_path(qid)
    obj = lakefs.repository(repo, client=client).branch(branch).object(path)
    try:
        with obj.reader(mode="rb") as f:
            return f.read().decode("utf-8")
    except Exception as exc:
        raise FileNotFoundError(
            f"No lakeFS object for {qid} at {branch}/{path}: {exc}"
        ) from exc


def main():
    pass


if __name__ == "__main__":
    main()
