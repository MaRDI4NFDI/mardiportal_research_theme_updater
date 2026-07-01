"""Download arXiv HTML5 and convert to Markdown with clean $...$ LaTeX formulas."""
from __future__ import annotations

import json
import re
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
            return _convert_html(resp.text, arxiv_id=arxiv_id)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _RETRIES:
                time.sleep(_RETRY_PAUSE)
    raise RuntimeError(
        f"Failed to fetch arXiv HTML5 for {arxiv_id} after {1 + _RETRIES} attempts: {last_exc}"
    ) from last_exc


def _equation_metadata(source_id: str, number: str) -> str:
    """Return a visible, machine-readable marker for an arXiv equation."""
    return (
        "[Equation metadata: "
        f"source_id={source_id or 'unknown'}; "
        f"number={number or 'unnumbered'}]"
    )


def _clean_text(value: str) -> str:
    """Collapse HTML formatting whitespace in document metadata."""
    return " ".join(value.split())


def _text_after_label(soup: BeautifulSoup, prefix: str) -> str:
    """Return text following a bold metadata label such as ``Keywords:``."""
    label = soup.find(
        "span",
        string=lambda text: bool(text and text.strip().lower().startswith(prefix.lower())),
    )
    if not label or not label.parent:
        return ""
    text = label.parent.get_text(" ", strip=True)
    return _clean_text(text.split(":", 1)[1]) if ":" in text else ""


def _combined_author_records(name_tag) -> list[dict]:
    """Parse LaTeXML's fallback layout where all authors share one personname."""
    if not name_tag.find("br"):
        return []

    names: list[tuple[str, bool]] = []
    current = ""
    after_break = False
    trailing = ""
    for child in name_tag.contents:
        child_name = getattr(child, "name", None)
        if child_name == "br":
            after_break = True
            continue
        if after_break:
            if child_name != "sup":
                trailing += (
                    child.get_text(" ", strip=True)
                    if hasattr(child, "get_text")
                    else str(child)
                )
            continue
        if child_name == "sup":
            name = re.sub(r"^(?:and\s+)", "", _clean_text(current), flags=re.I)
            if name:
                names.append((name, "∗" in child.get_text() or "*" in child.get_text()))
            current = ""
        else:
            current += (
                child.get_text(" ", strip=True)
                if hasattr(child, "get_text")
                else str(child)
            )

    final_name = re.sub(r"^(?:and\s+)", "", _clean_text(current), flags=re.I)
    if final_name:
        names.append((final_name, False))
    if len(names) < 2:
        return []

    trailing = _clean_text(trailing)
    trailing = re.sub(r"^\d+\s*", "", trailing)
    affiliation = re.split(
        r"\s*[∗*]?\s*Correspondence\s*:",
        trailing,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", trailing)
    email = email_match.group(0) if email_match else ""

    return [
        {
            "name": name,
            "affiliation": affiliation,
            "email": email if corresponding else "",
        }
        for name, corresponding in names
    ]


def _extract_document_metadata(soup: BeautifulSoup, arxiv_id: str) -> dict:
    """Extract stable document metadata from LaTeXML's semantic HTML classes."""
    title_tag = soup.select_one("h1.ltx_title_document")
    title = _clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""

    detected = re.search(
        r"\barXiv:(\d{4}\.\d{4,5}(?:v\d+)?)\b",
        soup.get_text(" ", strip=True),
        re.IGNORECASE,
    )
    resolved_arxiv_id = arxiv_id or (detected.group(1) if detected else "")

    authors = []
    for creator in soup.select(".ltx_creator.ltx_role_author"):
        name_tag = creator.select_one(".ltx_personname")
        combined = _combined_author_records(name_tag) if name_tag else []
        if combined:
            authors.extend(combined)
            continue
        name = _clean_text(name_tag.get_text(" ", strip=True)) if name_tag else ""
        affiliations = [
            _clean_text(tag.get_text(" ", strip=True))
            for tag in creator.select(".ltx_contact.ltx_role_address")
            if tag.get_text(" ", strip=True)
        ]
        email_tag = creator.select_one(".ltx_contact.ltx_role_email")
        email = _clean_text(email_tag.get_text(" ", strip=True)) if email_tag else ""
        if name:
            authors.append(
                {
                    "name": name,
                    "affiliation": "; ".join(affiliations),
                    "email": email,
                }
            )

    keyword_text = _text_after_label(soup, "keywords")
    keywords = [
        value.strip().rstrip(".")
        for value in re.split(r"[,;]", keyword_text)
        if value.strip()
    ]
    msc_text = _text_after_label(soup, "mathematics subject classification")
    msc_codes = re.findall(r"\b\d{2}[A-Z]\d{2}\b", msc_text)

    return {
        "arxiv_id": resolved_arxiv_id,
        "title": title,
        "authors": authors,
        "keywords": keywords,
        "msc_2020": msc_codes,
    }


def _normalize_document_sections(soup: BeautifulSoup, metadata: dict) -> None:
    """Turn LaTeXML metadata containers into explicit Markdown sections."""
    authors_tag = soup.select_one(".ltx_authors")
    if authors_tag and metadata["authors"]:
        section = soup.new_tag("section")
        heading = soup.new_tag("h2")
        heading.string = "Authors"
        section.append(heading)
        listing = soup.new_tag("ul")
        for author in metadata["authors"]:
            parts = [author["name"]]
            if author["affiliation"]:
                parts.append(author["affiliation"])
            if author["email"]:
                parts.append(author["email"])
            item = soup.new_tag("li")
            item.string = " — ".join(parts)
            listing.append(item)
        section.append(listing)
        authors_tag.replace_with(section)

    abstract_heading = soup.select_one(".ltx_title_abstract")
    if abstract_heading:
        abstract_heading.name = "h2"
        abstract_heading.string = "Abstract"

    for prefix, heading_text in (
        ("keywords", "Keywords"),
        ("mathematics subject classification", "Mathematics Subject Classification 2020"),
    ):
        label = soup.find(
            "span",
            string=lambda text, prefix=prefix: bool(
                text and text.strip().lower().startswith(prefix)
            ),
        )
        if not label or not label.parent:
            continue
        paragraph = label.parent
        label.extract()
        heading = soup.new_tag("h2")
        heading.string = heading_text
        paragraph.insert_before(heading)


def _front_matter(metadata: dict) -> str:
    """Render deterministic YAML front matter using JSON-compatible strings."""
    quote = lambda value: json.dumps(value, ensure_ascii=False)
    lines = [
        "---",
        f"arxiv_id: {quote(metadata['arxiv_id'])}",
        f"title: {quote(metadata['title'])}",
        "authors:",
    ]
    if metadata["authors"]:
        for author in metadata["authors"]:
            lines.extend(
                [
                    f"  - name: {quote(author['name'])}",
                    f"    affiliation: {quote(author['affiliation'])}",
                    f"    email: {quote(author['email'])}",
                ]
            )
    else:
        lines[-1] = "authors: []"

    for field in ("keywords", "msc_2020"):
        values = metadata[field]
        if values:
            lines.append(f"{field}:")
            lines.extend(f"  - {quote(value)}" for value in values)
        else:
            lines.append(f"{field}: []")
    lines.extend(["---", ""])
    return "\n".join(lines)


def _convert_html(html: str, arxiv_id: str = "") -> str:
    """Convert arXiv HTML5 string to Markdown, replacing MathML with $...$."""
    soup = BeautifulSoup(html, "html.parser")

    # arXiv HTML5 wraps the paper in <article>; use it to skip site header/nav
    article = soup.find("article")
    if article:
        soup = BeautifulSoup(str(article), "html.parser")

    metadata = _extract_document_metadata(soup, arxiv_id)
    _normalize_document_sections(soup, metadata)

    # Multi-row equation groups are best left as tables so their row structure
    # survives html2text. Insert metadata immediately before each group; the
    # remaining <math> pass below converts every cell to LaTeX.
    for group in soup.select("table.ltx_equationgroup"):
        tag = group.select_one(".ltx_tag_equation")
        number = tag.get_text(" ", strip=True) if tag else ""
        marker = soup.new_tag("p")
        marker.string = _equation_metadata(group.get("id", ""), number)
        group.insert_before(marker)

    # Numbered display equations: <table class="ltx_equation ...">
    for table in soup.find_all("table", class_="ltx_equation"):
        math_tag = table.find("math")
        latex = math_tag.get("alttext", "") if math_tag else ""
        tag = table.select_one(".ltx_tag_equation")
        number = tag.get_text(" ", strip=True) if tag else ""
        marker = _equation_metadata(table.get("id", ""), number)
        table.replace_with(f"\n\n{marker}\n\n$${latex}$$\n\n")

    # Remaining <math> elements (inline, or display="block" unnumbered)
    for tag in soup.find_all("math"):
        latex = tag.get("alttext", "")
        delim = "$$" if tag.get("display") == "block" else "$"
        tag.replace_with(f"{delim}{latex}{delim}")

    h = _html2text.HTML2Text()
    h.body_width = 0
    h.protect_links = True
    return _front_matter(metadata) + h.handle(str(soup))
