#!/usr/bin/env python3
"""Extract mathematical formulas from arXiv paper Markdown stored in lakeFS.

Scans the lakeFS repository for available .md.txt files, displays the
corresponding papers (QID + title), and — when a QID is given — sends the
fulltext to an LLM to extract all formulas as structured JSON.

Usage:
    python maintenance/extract_formulas.py [QID]

    Without QID: list all available papers and exit.
    With QID:    extract formulas and write maintenance/{QID}_formulas__{model}.json

Environment variables:
    LAKEFS_URL          (default: https://lake-bioinfmed.zib.de)
    LAKEFS_USER         lakeFS access key ID
    LAKEFS_PASSWORD     lakeFS secret access key
    LAKEFS_REPO         (default: mardi-fdo-data)
    LAKEFS_BRANCH       (default: main)
    SPARQL_ENDPOINT_URL (default: https://query.portal.mardi4nfdi.de/sparql)
    OPENROUTER_API_KEY  API key for openrouter.ai
    OPENROUTER_MODEL    model to use (default: z-ai/glm-4.7-flash)
    OPENROUTER_MAX_TOKENS  completion token cap sent to OpenRouter (default: 131072)
    OPENROUTER_PROVIDER    comma-separated provider slug order to pin to, no fallback
                            (e.g. "cloudflare,novita"; case-insensitive); default lets
                            OpenRouter choose
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
# Successful full-paper extractions have used 60-75k output tokens (e.g. deepseekv4flash
# at ~61.6k, mimo-v2.5-pro at ~62.2k, qwen3.6-flash truncated at 73.8k). Set well above
# that observed range; OpenRouter/the provider will still apply their own true ceiling
# if it's lower than this.
_MAX_COMPLETION_TOKENS = 131072


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


_SYSTEM_PROMPT = r"""Extract the mathematical formula occurrences from the supplied paper Markdown and return a single valid JSON array.

Output requirements:

* Output only the JSON array.
* Do not use Markdown fences.
* Do not include explanatory text.
* The output must parse as strict JSON.
* Preserve LaTeX backslashes using valid JSON escaping.

## 0. Document coverage

The input is the full text of one paper, from its title through its final section (proofs, appendices, and all). You must process it section by section, in order, all the way to the end of the supplied Markdown — never stop after the introduction or after a "statement of results" section.

Sections titled "introduction," "statement of results," "main results," or similar name only the *headline* theorems; they are not the scope boundary for extraction. Proof sections ("Proof of Theorem ...", appendices, etc.) routinely contain lemmas, recurrences, and intermediate identities that meet the inclusion criteria in Section 1 below and must be extracted with the same rigor as the headline theorems.

Do not plan to extract only the "main" theorems and definitions while treating lemmas or proof-derivation formulas as out of scope — Section 1's inclusion list explicitly names lemma and corollary formulas, and that applies uniformly across the whole document, not just the opening section.

Before emitting output, confirm internally that your extraction reaches the last numbered or displayed formula in the supplied Markdown. If the paper has proof sections after the statement of results, your output must include formulas from those sections too.

This is a real extraction job whose output is consumed programmatically, not an illustrative sample, a demonstration, or a simulation. If you privately enumerate candidate formulas before writing the JSON array, every candidate that meets the inclusion criteria in Section 1 must appear in the output — do not decide partway through that you will emit only "the most important ones," "a representative subset," or "the first N as an example." There is no length limit on the array: a long, exhaustive array is the correct and expected output, not a problem to economize around.

If you are genuinely short on output budget, degrade gracefully rather than dropping formulas silently: keep every formula record but shorten low-value optional fields first (`description_long`, `notation_variants`, `related_concepts`, secondary `symbols` entries) before omitting any formula entirely. Never omit a formula from the output array merely to save space or time.

## 1. Extraction scope

Extract a formula when it expresses at least one mathematically meaningful relation, definition, construction, bound, recurrence, or named mathematical object.

Include:

* displayed equations;
* numbered equations;
* formulas in `align`, `aligned`, `split`, array, or table structures;
* inline formulas that define an object or state a substantive mathematical relation;
* theorem, lemma, corollary, and assumption formulas — including those that appear only inside proof sections;
* formulas in captions when they contain substantive mathematical content.

Normally exclude:

* isolated variables such as `$n$`;
* bare parameter lists such as `$r=3,4$`, unless they are needed as conditions for an extracted formula;
* purely typographical fragments;
* bibliographic expressions;
* formulas occurring only in the reference list.

## 2. Formula granularity

Create one record for each logically independent mathematical statement.

* Split a display containing several unrelated definitions or equations into separate records.
* Keep chained inequalities together when they represent one bound.
* Keep a recurrence and its initial condition in the same record when both are required to define the sequence.
* Keep cases in a piecewise definition together.
* Keep aligned derivation steps together only when they represent a single derivation rather than several independently reusable formulas.

## 3. Deduplication

Deduplicate formulas at the semantic level only when they express the same mathematical statement with the same symbols and assumptions.

Do not merge formulas merely because they are algebraically similar.

When the same formula occurs more than once:

* create one formula record;
* include every occurrence in `occurrences`;
* use the clearest occurrence for `formula_as_found_primary`.

Do not rename variables during deduplication.

## 4. LaTeX normalization

For `defining_formula`:

* preserve the mathematical meaning and original symbols;
* remove presentation-only commands such as `\displaystyle`;
* normalize whitespace;
* normalize equivalent delimiter commands when safe;
* retain equation environments only when structurally necessary;
* do not expand, simplify, rearrange, or algebraically transform the formula;
* do not silently repair malformed source LaTeX.

If the Markdown appears corrupted, preserve the source text, provide the best conservative normalization, add a warning, and reduce extraction confidence.

## 5. Classification policy

Use `formula_type_primary` for the principal role of the formula and `formula_types_secondary` for additional applicable roles.

Allowed formula types:

* `equation`
* `inequality`
* `identity`
* `definition`
* `theorem_statement`
* `lemma_statement`
* `approximation`
* `bound`
* `recurrence`
* `representation`
* `optimization_problem`
* `condition`

Use the following novelty classifications:

* `standard`: a well-known established formula or definition;
* `paper_result`: derived or proved as a result of this paper;
* `cited_result`: attributed to earlier work;
* `adaptation`: an adjusted or specialized form of an established formula;
* `unknown`: insufficient evidence to decide.

Do not infer novelty solely because a formula occurs in a theorem.

## 6. Schema

Each array element must have this structure:

{
  "formula_id": "F0001",
  "item_type": "mathematical formula",
  "label": "concise human-readable label",
  "defining_formula": "normalized LaTeX",
  "formula_as_found_primary": "verbatim LaTeX from the clearest occurrence",
  "description_long": "precise explanation of the mathematical statement and its role in the paper",
  "formula_type_primary": "definition",
  "formula_types_secondary": ["equation"],
  "classification": "standard",
  "classification_basis": "brief reason based on the paper, or empty string",
  "conditions": [
    {
      "latex": "n \\geq 2",
      "description": "the number of mesh intervals is at least two",
      "explicit": true
    }
  ],
  "is_numbered": true,
  "equation_number": "(3.2)",
  "equation_label_raw": "(3.2)",
  "symbols": [
    {
      "symbol": "n",
      "canonical_symbol": "n",
      "represents": "number of mesh intervals",
      "type": "index",
      "domain": "\\mathbb{N}",
      "codomain": "",
      "definition_status": "explicit",
      "scope": "global",
      "confidence": 0.99
    }
  ],
  "notation_variants": [
    {
      "formula": "alternative LaTeX notation",
      "context": "where or why this variant is used"
    }
  ],
  "related_concepts": [
    {
      "label": "Peano kernel",
      "relation": "uses",
      "wikidata_qid": "",
      "dlmf": ""
    }
  ],
  "msc_codes_suggested": ["65D32"],
  "occurrences": [
    {
      "section": "1. Introduction and statement of the results",
      "subsection": "",
      "context_type": "definition",
      "equation_number": "(1.5)",
      "formula_as_found": "verbatim LaTeX at this occurrence",
      "source_text_before": "sentence immediately before the formula",
      "source_text_after": "sentence immediately after the formula",
      "markdown_locator": "heading and local occurrence index"
    }
  ],
  "source_integrity": "clean",
  "warnings": [],
  "confidence": {
    "formula_extraction": 0.99,
    "deduplication": 0.95,
    "classification": 0.90,
    "description": 0.95,
    "symbols": 0.93
  },
  "requires_source_check": false,
  "review_status": "unreviewed"
}

## 7. Field constraints

`formula_type_primary` and entries in `formula_types_secondary` must be selected from:

* `equation`
* `inequality`
* `identity`
* `definition`
* `theorem_statement`
* `lemma_statement`
* `approximation`
* `bound`
* `recurrence`
* `representation`
* `optimization_problem`
* `condition`

`classification` must be one of:

* `standard`
* `paper_result`
* `cited_result`
* `adaptation`
* `unknown`

`symbol.type` must be one of:

* `variable`
* `constant`
* `operator`
* `function`
* `set`
* `index`
* `parameter`
* `functional`

`symbol.definition_status` must be one of:

* `explicit`
* `inferred`
* `unknown`

`source_integrity` must be one of:

* `clean`
* `possibly_corrupted`
* `corrupted`

`conditions` must be an empty array when no conditions are stated or reliably inferred.

`is_numbered` is true only when the paper visibly assigns a number or label to the formula.

`equation_number` must contain the normalized equation number when available; otherwise use an empty string.

`equation_label_raw` must preserve the exact source label; otherwise use an empty string.

`msc_codes_suggested` must be empty when no formula-specific or concept-specific MSC assignment is reasonably supported.

Do not invent DLMF references or Wikidata QIDs. Use empty strings when not known with high confidence.

All confidence values must be numbers between 0.0 and 1.0.

`review_status` must always be `"unreviewed"`.

## 8. Source fidelity

The fields `formula_as_found_primary` and `occurrences[].formula_as_found` must reproduce the source formula verbatim, including apparent errors.

The normalized field `defining_formula` may correct only unambiguous formatting artifacts. Any correction must be mentioned in `warnings`.

When a formula cannot be reconstructed reliably from the Markdown:

* preserve the damaged source;
* set `source_integrity` to `possibly_corrupted` or `corrupted`;
* set `requires_source_check` to true;
* lower `formula_extraction` confidence;
* explain the problem in `warnings`.

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


def _salvage_partial_json_array(text: str) -> str:
    """Close a truncated JSON array after the last complete top-level object.

    Scans the array contents tracking JSON string/escape state and brace
    depth, and remembers the position of the last '}' that closes a
    top-level array element (depth 1 -> 0). A naive `rfind("}")` is unsafe
    here because formula content is full of LaTeX braces (e.g. `B_{4}`,
    `Q_{n+1}`) inside string literals, which are not JSON structural braces
    and would make the salvage cut at a nonsensical, mid-string position.
    """
    start = text.find("[")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False
    last_complete_end = -1
    for i in range(start + 1, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                last_complete_end = i

    if last_complete_end == -1:
        return text
    return text[: last_complete_end + 1] + "]"


def _log_generation_stats(generation_id: str, api_key: str) -> None:
    """Fetch and log OpenRouter's per-generation stats (best-effort, never raises).

    Surfaces which upstream provider actually served the request and its native
    token breakdown (completion vs. reasoning) — the data needed to diagnose
    provider-specific output caps like the one hit on DeepInfra for glm-4.7-flash.
    """
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/generation",
            params={"id": generation_id},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        stats = resp.json().get("data", {})
        log.info(
            "Generation stats: provider=%s native_completion=%s native_reasoning=%s "
            "finish_reason=%s cost=$%s",
            stats.get("provider_name"),
            stats.get("native_tokens_completion"),
            stats.get("native_tokens_reasoning"),
            stats.get("native_finish_reason"),
            stats.get("usage"),
        )
    except (requests.RequestException, ValueError) as exc:
        log.warning("Could not fetch generation stats for %s: %s", generation_id, exc)


def extract_formulas_llm(
    markdown: str,
    api_key: str,
    model: str = _MODEL,
    max_tokens: int = _MAX_COMPLETION_TOKENS,
    providers: list[str] | None = None,
) -> list[dict]:
    """Send *markdown* to OpenRouter and return extracted formulas as a list of dicts.

    *providers*, if given, pins the OpenRouter request to that ordered list of
    upstream providers (e.g. ["Cloudflare", "NovitaAI"]) with no fallback to
    others — useful since the same model can have very different completion
    token caps and reasoning-token behavior depending on which provider serves it.

    Raises ValueError if the response cannot be parsed as a JSON array.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        # Cap hidden reasoning so reasoning-capable models don't spend their whole
        # completion budget "thinking" and leave nothing for the actual JSON array
        # (observed with z-ai/glm-4.7-flash: 14374 of a 16384-token provider cap went
        # to reasoning, leaving an empty `content` field). Providers that don't support
        # reasoning controls ignore this field.
        "reasoning": {"max_tokens": 2000},
    }
    if providers:
        # OpenRouter's provider.order expects lowercase provider slugs (e.g.
        # "cloudflare", "deepinfra"), not the capitalized display names shown on the
        # pricing page ("Cloudflare", "DeepInfra") — passing the display name matches
        # no provider and looks identical to "this provider has no live endpoint"
        # (404 "No endpoints found"). Normalize defensively so either form works.
        #
        # allow_fallbacks=False: hard pin to this provider list, no silent fallback.
        # With fallback allowed, OpenRouter routed straight to DeepInfra and ignored
        # the preference entirely whenever a listed provider had no live endpoint —
        # giving no error and no signal that the pin had no effect. Failing loudly
        # is more useful than silently getting the wrong provider's much smaller
        # output cap.
        payload["provider"] = {
            "order": [p.lower() for p in providers],
            "allow_fallbacks": False,
        }
    log.info("Requesting model=%s provider=%s", model, payload.get("provider"))
    resp = requests.post(
        _OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise ValueError(f"{exc}. Response body: {resp.text[:2000]}") from exc
    try:
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason", "")
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected OpenRouter response structure: {exc}") from exc
    generation_id = data.get("id")
    if generation_id:
        _log_generation_stats(generation_id, api_key)
    if not content:
        raise ValueError(
            f"Model returned no completion content (finish_reason={finish_reason!r}). "
            "The model likely exhausted its output budget on hidden reasoning before "
            "writing any JSON; try a lower 'reasoning.max_tokens' or a non-reasoning model."
        )
    if finish_reason == "length":
        log.warning(
            "Model hit output token limit (finish_reason=length) — response is truncated. "
            "Attempting to salvage partial JSON array."
        )
        content = _salvage_partial_json_array(content)
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


def _model_slug(model: str) -> str:
    """Turn an OpenRouter model id (e.g. "z-ai/glm-4.7-flash") into a short
    filesystem-safe slug (e.g. "glm47flash") for use in output filenames."""
    name = model.rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _output_path(qid: str, model: str) -> str:
    slug = _model_slug(model)
    return os.path.join(os.path.dirname(__file__), f"{qid}_formulas__{slug}.json")


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
    openrouter_max_tokens = int(
        os.environ.get("OPENROUTER_MAX_TOKENS", str(_MAX_COMPLETION_TOKENS)).strip()
    )
    openrouter_providers = [
        p.strip() for p in os.environ.get("OPENROUTER_PROVIDER", "").split(",") if p.strip()
    ]

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
        formulas = extract_formulas_llm(
            markdown, openrouter_key, openrouter_model, openrouter_max_tokens, openrouter_providers
        )
    except (ValueError, requests.RequestException) as exc:
        sys.exit(f"LLM extraction failed: {exc}")
    log.info("Extracted %d formula(s)", len(formulas))

    # --- Stage 7: save output ---
    out_path = _output_path(target, openrouter_model)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(formulas, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(formulas)} formula(s) to {out_path}")


if __name__ == "__main__":
    main()
