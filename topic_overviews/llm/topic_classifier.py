"""Classify a paper against the KG-registered topics using Claude."""
from __future__ import annotations

import json
import re

from ..harvest.arxiv_oai import PaperRecord
from ..kg.topics import Topic

_SYSTEM = (
    "You classify a mathematics paper into a fixed list of research topics. "
    "Return ONLY a JSON object of the form {\"topics\": [\"Q123\", ...]} listing the "
    "QIDs of the topics the paper clearly belongs to. Use an empty list if none fit. "
    "Never invent QIDs that are not in the provided list."
)


def _build_prompt(paper: PaperRecord, topics: list[Topic]) -> str:
    topic_lines = "\n".join(f"- {t.qid}: {t.label} — {t.description}" for t in topics)
    return (
        f"{_SYSTEM}\n\nTOPICS:\n{topic_lines}\n\n"
        f"PAPER TITLE: {paper.title}\n"
        f"ABSTRACT: {paper.abstract}\n\n"
        'Respond with JSON only, e.g. {"topics": ["Q11"]}.'
    )


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return json.loads(match.group(0))


def classify_paper(
    paper: PaperRecord,
    topics: list[Topic],
    *,
    model: str,
    api_key: str,
    client=None,
) -> list[str]:
    if client is None:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

    valid = {t.qid for t in topics}
    prompt = _build_prompt(paper, topics)

    for _ in range(2):
        resp = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        try:
            data = _extract_json(text)
            return [q for q in data.get("topics", []) if q in valid]
        except (ValueError, json.JSONDecodeError):
            continue
    return []
