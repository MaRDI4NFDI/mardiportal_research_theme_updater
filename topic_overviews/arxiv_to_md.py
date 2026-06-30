"""Download arXiv HTML5 and convert to Markdown with clean $...$ LaTeX formulas."""
from __future__ import annotations

import time

import requests
from bs4 import BeautifulSoup
import html2text as _html2text


_ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
_RETRIES = 2
_RETRY_PAUSE = 3  # seconds


def fetch_and_convert(arxiv_id: str) -> str:
    """Fetch arXiv HTML5 for *arxiv_id* and return Markdown with LaTeX formulas.

    Retries up to _RETRIES times with a _RETRY_PAUSE second pause on failure.
    """
    url = _ARXIV_HTML_URL.format(arxiv_id=arxiv_id)
    last_exc: Exception | None = None
    for attempt in range(1 + _RETRIES):
        try:
            resp = requests.get(url, headers={"User-Agent": "MaRDI-topic-overviews/1.0"}, timeout=30)
            resp.raise_for_status()
            return _convert_html(resp.text)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _RETRIES:
                time.sleep(_RETRY_PAUSE)
    raise RuntimeError(
        f"Failed to fetch arXiv HTML5 for {arxiv_id} after {1 + _RETRIES} attempts: {last_exc}"
    ) from last_exc


def _convert_html(html: str) -> str:
    """Convert arXiv HTML5 string to Markdown, replacing MathML with $...$."""
    soup = BeautifulSoup(html, "html.parser")

    # arXiv HTML5 wraps the paper in <article>; use it to skip site header/nav
    article = soup.find("article")
    if article:
        soup = BeautifulSoup(str(article), "html.parser")

    # Numbered display equations: <table class="ltx_equation ...">
    for table in soup.find_all("table", class_="ltx_equation"):
        math_tag = table.find("math")
        latex = math_tag.get("alttext", "") if math_tag else ""
        table.replace_with(f"\n\n$${latex}$$\n\n")

    # Remaining <math> elements (inline, or display="block" unnumbered)
    for tag in soup.find_all("math"):
        latex = tag.get("alttext", "")
        delim = "$$" if tag.get("display") == "block" else "$"
        tag.replace_with(f"{delim}{latex}{delim}")

    h = _html2text.HTML2Text()
    h.body_width = 0
    h.protect_links = True
    return h.handle(str(soup))
