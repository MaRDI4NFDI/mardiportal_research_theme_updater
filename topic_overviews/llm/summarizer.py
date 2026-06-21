"""Generate a one-sentence TL;DR for a paper using Claude."""
from __future__ import annotations

from ..harvest.arxiv_oai import PaperRecord

_SYSTEM = (
    "Write a two-sentence TL;DR (about 40-55 words total). The first sentence says "
    "what the paper does; the second states its main contribution, key result, or "
    "what is novel. Plain prose, no preamble, no markdown, no trailing newline."
)


def summarize_paper(
    paper: PaperRecord,
    *,
    model: str,
    api_key: str,
    client=None,
) -> str:
    """Return a one-sentence TL;DR for the paper (empty string on failure)."""
    if client is None:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

    prompt = f"{_SYSTEM}\n\nTITLE: {paper.title}\nABSTRACT: {paper.abstract}"
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return ""
