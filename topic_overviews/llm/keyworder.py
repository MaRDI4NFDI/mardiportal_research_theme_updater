"""Generate a few topical keywords/tags for a paper using Claude."""
from __future__ import annotations

import json
import re

from ..harvest.arxiv_oai import PaperRecord

_SYSTEM = (
    "Extract 4 to 6 short topical keywords/tags (1-3 words each, Title Case) that "
    "characterize this paper, suitable as filter chips. Return ONLY a JSON array of "
    "strings, e.g. [\"Opinion Dynamics\", \"Multi-Agent\"]. No preamble, no markdown."
)


def keywords_paper(
    paper: PaperRecord,
    *,
    model: str,
    api_key: str,
    client=None,
) -> list[str]:
    """Return a short keyword list for the paper (empty list on failure)."""
    if client is None:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

    prompt = f"{_SYSTEM}\n\nTITLE: {paper.title}\nABSTRACT: {paper.abstract}"
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        data = json.loads(match.group(0)) if match else []
        return [str(k).strip() for k in data if str(k).strip()]
    except Exception:
        return []
