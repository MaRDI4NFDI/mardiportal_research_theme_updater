"""Generate a one-sentence TL;DR for a paper using Claude."""
from __future__ import annotations

import logging

from ..harvest.arxiv_oai import PaperRecord

log = logging.getLogger(__name__)

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
    llm=None,
) -> str:
    """Return a one-sentence TL;DR for the paper (empty string on failure)."""
    if llm is None and client is None:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

    prompt = f"{_SYSTEM}\n\nTITLE: {paper.title}\nABSTRACT: {paper.abstract}"
    try:
        if llm is not None:
            text = llm.complete(prompt, model=model, max_tokens=4096)
        else:
            resp = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
        log.info("TL;DR raw response: %s", text[:200])
        return " ".join(text.split())
    except Exception as exc:
        log.warning("TL;DR generation failed: %s", exc)
        return ""
