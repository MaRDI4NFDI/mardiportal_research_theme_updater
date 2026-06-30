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
    OPENROUTER_MODEL    model to use (default: z-ai/glm-4.7-flash)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

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


_VALID_ESCAPE_CHARS = set('"' + "\\" + "/bfnrtu")


_CTRL_ESCAPE = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def _fix_invalid_escapes(s: str) -> str:
    """Double any backslash not followed by a valid JSON escape character.

    Steps over valid escape sequences as a pair so that e.g. \\\\int is not
    re-processed and the trailing \\i misidentified as invalid.
    """
    result: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\":
            if i + 1 < len(s) and s[i + 1] in _VALID_ESCAPE_CHARS:
                result.append(s[i])
                result.append(s[i + 1])
                if s[i + 1] == "u":
                    result.append(s[i + 2 : i + 6])
                    i += 6
                else:
                    i += 2
            else:
                result.append("\\\\")
                i += 1
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def _fix_control_chars(s: str) -> str:
    """Escape raw control characters that appear inside JSON string literals."""
    result: list[str] = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if in_string:
            if c == "\\":
                result.append(c)
                i += 1
                if i < len(s):
                    result.append(s[i])
                    i += 1
                continue
            elif c == '"':
                in_string = False
                result.append(c)
            elif ord(c) < 0x20:
                result.append(_CTRL_ESCAPE.get(c, f"\\u{ord(c):04x}"))
            else:
                result.append(c)
        else:
            if c == '"':
                in_string = True
                result.append(c)
            else:
                result.append(c)
        i += 1
    return "".join(result)


def _extract_json_array(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text.

    Handles responses that may have markdown fences or leading prose.
    """
    text = text.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence (first line) and closing fence (last line if it starts with ```)
        start_idx = 1
        end_idx = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[start_idx:end_idx]).strip()
    # Find first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response: {text[:200]!r}")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as first_exc:
        for label, fn in [
            ("backslash fix", _fix_invalid_escapes),
            ("control-char fix", _fix_control_chars),
            ("both fixes", lambda s: _fix_control_chars(_fix_invalid_escapes(s))),
        ]:
            log.warning("JSON parse error (%s), attempting %s…", first_exc.msg, label)
            try:
                return json.loads(fn(candidate))
            except json.JSONDecodeError:
                pass
        raw_path = os.path.join(os.path.dirname(__file__), "_last_llm_response.txt")
        with open(raw_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        context_start = max(0, first_exc.pos - 120)
        context_end = min(len(candidate), first_exc.pos + 120)
        log.error("JSON parse error at char %d: %s", first_exc.pos, first_exc.msg)
        log.error("Context: …%r…", candidate[context_start:context_end])
        log.error("Raw LLM response saved to %s", raw_path)
        raise ValueError(str(first_exc)) from first_exc


def extract_formulas_llm(markdown: str, api_key: str, model: str = _MODEL) -> list[dict]:
    """Send *markdown* to OpenRouter and return extracted formulas as a list of dicts.

    Raises ValueError if the response cannot be parsed as a JSON array.
    """
    payload = {
        "model": model,
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
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected OpenRouter response structure: {exc}") from exc
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


def _output_path(qid: str) -> str:
    return os.path.join(os.path.dirname(__file__), f"{qid}_formulas.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "qid",
        nargs="?",
        default=None,
        metavar="QID",
        help="QID of the paper to process (e.g. Q6190920). Omit to list all available papers.",
    )
    args = parser.parse_args()

    lakefs_url = os.environ.get("LAKEFS_URL", _LAKEFS_URL).strip()
    lakefs_user = os.environ.get("LAKEFS_USER", "").strip()
    lakefs_password = os.environ.get("LAKEFS_PASSWORD", "").strip()
    lakefs_repo = os.environ.get("LAKEFS_REPO", _LAKEFS_REPO).strip()
    lakefs_branch = os.environ.get("LAKEFS_BRANCH", _LAKEFS_BRANCH).strip()
    sparql_endpoint = os.environ.get("SPARQL_ENDPOINT_URL", _SPARQL_ENDPOINT).strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    openrouter_model = os.environ.get("OPENROUTER_MODEL", _MODEL).strip()

    for name, val in [("LAKEFS_USER", lakefs_user), ("LAKEFS_PASSWORD", lakefs_password)]:
        if not val:
            sys.exit(f"Missing environment variable: {name}")

    if not args.qid:
        # --- List mode: scan lakeFS, fetch titles, display table ---
        log.info("Scanning lakeFS %s/%s for .md.txt files…", lakefs_repo, lakefs_branch)
        qids = list_lakefs_papers(lakefs_url, lakefs_user, lakefs_password, lakefs_repo, lakefs_branch)
        log.info("Found %d paper(s)", len(qids))
        session = requests.Session()
        log.info("Fetching paper titles from SPARQL…")
        titles = get_paper_titles(qids, sparql_endpoint, session)
        print(f"\n{'QID':<15} {'Title'}")
        print("-" * 80)
        for q in qids:
            title = titles.get(q, "")
            print(f"{q:<15} {title or '(no title in KG)'}")
        print(f"\n{len(qids)} paper(s) available in lakeFS.\n")
        sys.exit(0)

    # --- Stage 4: validate requested QID ---
    target = args.qid.upper()
    session = requests.Session()

    if not openrouter_key:
        sys.exit("Missing environment variable: OPENROUTER_API_KEY")

    # --- Stage 5: download Markdown ---
    log.info("Downloading Markdown for %s…", target)
    try:
        markdown = download_markdown(target, lakefs_url, lakefs_user, lakefs_password, lakefs_repo, lakefs_branch)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    log.info("Downloaded %d characters", len(markdown))

    fulltext_path = os.path.join(os.path.dirname(__file__), f"{target}_fulltext.md")
    with open(fulltext_path, "w", encoding="utf-8") as fh:
        fh.write(markdown)
    log.info("Saved fulltext to %s", fulltext_path)

    # --- Stage 6: call LLM ---
    log.info("Sending to %s via OpenRouter…", openrouter_model)
    try:
        formulas = extract_formulas_llm(markdown, openrouter_key, openrouter_model)
    except (ValueError, requests.RequestException) as exc:
        sys.exit(f"LLM extraction failed: {exc}")
    log.info("Extracted %d formula(s)", len(formulas))

    # --- Stage 7: save output ---
    out_path = _output_path(target)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(formulas, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(formulas)} formula(s) to {out_path}")


if __name__ == "__main__":
    main()
