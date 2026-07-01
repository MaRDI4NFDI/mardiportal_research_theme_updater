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
    OPENROUTER_REASONING_TOKENS  cap on hidden reasoning tokens sent to OpenRouter
                            (default: 8000)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
# Successful full-paper extractions have used 60-75k output tokens (e.g. deepseekv4flash
# at ~61.6k, mimo-v2.5-pro at ~62.2k, qwen3.6-flash truncated at 73.8k). Set well above
# that observed range; OpenRouter/the provider will still apply their own true ceiling
# if it's lower than this.
_MAX_COMPLETION_TOKENS = 131072
# Cap on hidden reasoning tokens. Too low starves reasoning-heavy models of planning
# room (observed: grok-4.3 stopped after 2 formulas using only 722 of an available
# 2000); too high lets a model burn its whole completion budget "thinking" and leave
# nothing for the actual JSON (observed: glm-4.7-flash on DeepInfra spent 14374 of a
# 16384-token hard cap on reasoning). 8000 is a higher default to test against grok-style
# under-extraction; tune per model via OPENROUTER_REASONING_TOKENS.
_REASONING_MAX_TOKENS = 8000


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

Process the entire supplied paper in order, including proofs, appendices, and
captions. An introduction or "statement of results" section is not the scope
boundary. Include every formula meeting Section 1; never return a representative
sample or only the most important results. Before responding, confirm internally
that extraction reaches the paper's final mathematical content.

This is a real extraction job whose output is consumed programmatically, not a
demo, example, illustration, or simulation. Every candidate you identify that
meets Section 1 must appear in the JSON array. Never output only the first N
records or a subset intended to illustrate the schema.

If output space is limited, keep every formula record and shorten or omit
`description`, enrichment fields, and other optional metadata first. Never omit
a formula record merely to save output space.

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

If the Markdown appears corrupted, preserve the source text, provide the best
conservative normalization, add a warning, and set `requires_source_check`.

## 5. Classification policy

Use `formula_type` for the mathematical form of the formula. Allowed values:

* `equation`
* `inequality`
* `identity`
* `definition`
* `approximation`
* `bound`
* `recurrence`
* `representation`
* `optimization_problem`
* `condition`

Use the most specific applicable type. Prefer `definition`, `recurrence`,
`approximation`, `bound`, `identity`, `inequality`, `representation`,
`condition`, then `equation`. Use `bound` for an upper or lower estimate and
`identity` only for a relation asserted for all admissible values.

Record origin independently of the formula's role:

* `standard`: established mathematical knowledge not attributed here to a
  specific source;
* `cited_source`: explicitly attributed to earlier work;
* `current_paper`: introduced, derived, or proved by this paper;
* `unclear`: insufficient evidence.

Set `is_adaptation` to true only for an adjusted or specialized form of an
established or cited formula. Preserve explicit citation identifiers and
attribution wording. Do not infer familiarity or attribution from the formula
alone.

Set `established_name` to the established mathematical name explicitly used by
the paper, such as `"Peano kernel theorem"`. Use an empty string when the paper
does not supply an established name; do not create a descriptive name for this
field. The ordinary `label` field may still contain a generated descriptive
label.

Use `statement_role` to describe how the formula functions in the paper:

* `headline_result`: a principal result emphasized by the paper;
* `supporting_result`: an independently meaningful result used to support the
  main argument, including cited or standard results used by the paper;
* `definition`: a definition of an object or notation;
* `assumption`: a hypothesis or condition imposed on later results;
* `derivation_step`: an intermediate proof or calculation step that is not an
  independently meaningful result.

Use `statement_environment` for an explicit `theorem`, `lemma`, `proposition`,
or `corollary` environment; otherwise use `none`. A formula's environment does
not determine its mathematical type or origin.

In `derived_from_formula_ids`, record only formulas that are direct premises or
derivation inputs. Do not link formulas merely because they are nearby, share
symbols, or discuss the same topic. Python derives reverse support links.

The Markdown may place a source marker immediately before a formula:

`[Equation metadata: source_id=S1.E1; number=(1.1)]`

For every formula associated with such a marker, copy `source_id` exactly into
`occurrences[].source_id` and copy `number` exactly into
`occurrences[].equation_number`. When the marker says `number=unnumbered`, use
an empty `equation_number`. Do not infer, normalize, or renumber these values.
For a multi-row equation group, the marker applies to the complete group.

## 6. Schema

Each array element must have this structure:

{
  "formula_id": "F0001",
  "label": "concise human-readable label",
  "defining_formula": "normalized LaTeX",
  "formula_as_found_primary": "verbatim LaTeX from the clearest occurrence",
  "description": "precise explanation of the statement and its role",
  "formula_type": "definition",
  "origin": "standard",
  "is_adaptation": false,
  "statement_role": "definition",
  "statement_environment": "none",
  "classification_evidence": [
    "The surrounding text explicitly calls this a standard definition"
  ],
  "provenance": {
    "citation_keys": [],
    "attribution_text": ""
  },
  "established_name": "Peano kernel theorem",
  "derived_from_formula_ids": [],
  "conditions": [
    {
      "latex": "n \\geq 2",
      "description": "the number of mesh intervals is at least two",
      "explicit": true
    }
  ],
  "symbols": [
    {
      "symbol": "n",
      "represents": "number of mesh intervals",
      "type": "index",
      "introduced_in_formula_id": "F0001",
      "is_paper_local": false
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
      "context_type": "definition",
      "source_id": "S1.E1",
      "equation_number": "(1.1)",
      "formula_as_found": "verbatim LaTeX at this occurrence",
      "markdown_locator": "heading and local occurrence index"
    }
  ],
  "source_integrity": "clean",
  "warnings": [],
  "requires_source_check": false
}

## 7. Field constraints

`formula_type` must be one of:

* `equation`
* `inequality`
* `identity`
* `definition`
* `approximation`
* `bound`
* `recurrence`
* `representation`
* `optimization_problem`
* `condition`

`origin` must be one of:

* `standard`
* `cited_source`
* `current_paper`
* `unclear`

`classification_evidence` must contain one or more concise, source-grounded
observations supporting `origin`, `is_adaptation`, and `statement_role`. Do not
use unsupported claims such as "this is well known." For `origin: "unclear"`,
state what evidence is missing.

`provenance.citation_keys` must contain only citation identifiers explicitly
associated with the formula in the supplied paper. Preserve their source form,
for example `"12"` or `"Smith2020"`, and use an empty array when none is given.

`provenance.attribution_text` must preserve the concise source wording that
attributes the formula, theorem, or result. Use an empty string when there is no
explicit attribution.

`is_adaptation` must be a JSON boolean.

`established_name` must contain only a name explicitly supported by the paper;
otherwise use an empty string.

`statement_role` must be one of:

* `headline_result`
* `supporting_result`
* `definition`
* `assumption`
* `derivation_step`

`statement_environment` must be one of:

* `theorem`
* `lemma`
* `proposition`
* `corollary`
* `none`

`derived_from_formula_ids` must contain only valid, non-duplicate `formula_id`
values from the same output array and must not contain the record's own ID.

`occurrences[].source_id` must contain the exact `source_id` from the associated
equation metadata marker, or an empty string when no marker is present.

`occurrences[].equation_number` must contain the exact parenthesized `number`
from the associated marker. Use an empty string when the marker says
`number=unnumbered` or no equation number is present.

`symbol.type` must be one of:

* `variable`
* `constant`
* `operator`
* `function`
* `set`
* `index`
* `parameter`
* `functional`

`symbol.introduced_in_formula_id` must contain the `formula_id` of the extracted
formula that introduces or defines the symbol, including the current record's
ID when applicable. Use an empty string when the symbol is introduced only in
prose, is established notation not introduced by this paper, or cannot be
identified reliably.

`symbol.is_paper_local` must be a JSON boolean. Set it to true only for notation
introduced specifically for this paper's argument or construction, and false
for established mathematical notation and generic bound variables.

`source_integrity` must be one of:

* `clean`
* `possibly_corrupted`
* `corrupted`

`conditions` must be an empty array when no conditions are stated or reliably inferred.

`msc_codes_suggested` must be empty when no formula-specific or concept-specific MSC assignment is reasonably supported.

Do not invent DLMF references or Wikidata QIDs. Use empty strings when not known with high confidence.

## 8. Source fidelity

The fields `formula_as_found_primary` and `occurrences[].formula_as_found` must reproduce the source formula verbatim, including apparent errors.

The normalized field `defining_formula` may correct only unambiguous formatting artifacts. Any correction must be mentioned in `warnings`.

When a formula cannot be reconstructed reliably from the Markdown:

* preserve the damaged source;
* set `source_integrity` to `possibly_corrupted` or `corrupted`;
* set `requires_source_check` to true;
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


def _derive_metadata(formulas: list[dict]) -> list[dict]:
    """Add deterministic fields and reverse links to LLM-extracted records."""
    by_id: dict[str, dict] = {}
    for formula in formulas:
        formula_id = formula.get("formula_id")
        if not isinstance(formula_id, str) or not formula_id:
            raise ValueError("Every formula must have a non-empty string formula_id")
        if formula_id in by_id:
            raise ValueError(f"Duplicate formula_id: {formula_id}")
        by_id[formula_id] = formula

    for formula_id, formula in by_id.items():
        formula["item_type"] = "mathematical formula"
        formula["review_status"] = "unreviewed"

        origin = formula.get("origin")
        role = formula.get("statement_role")
        if formula.get("is_adaptation") is True:
            classification = "adaptation"
        elif origin == "standard":
            classification = "standard"
        elif origin == "cited_source":
            classification = "cited_result"
        elif origin == "current_paper" and role in {
            "headline_result",
            "supporting_result",
        }:
            classification = "paper_result"
        elif origin == "current_paper":
            classification = "paper_internal"
        else:
            classification = "unknown"
        formula["classification"] = classification
        formula["standalone_statement"] = role != "derivation_step"
        formula["supports_formula_ids"] = []

        occurrences = formula.get("occurrences")
        primary_occurrence = occurrences[0] if isinstance(occurrences, list) and occurrences else {}
        equation_number = (
            primary_occurrence.get("equation_number", "")
            if isinstance(primary_occurrence, dict)
            else ""
        )
        formula["equation_number"] = equation_number
        formula["equation_label_raw"] = equation_number
        formula["is_numbered"] = bool(equation_number)

        dependencies = formula.get("derived_from_formula_ids", [])
        if not isinstance(dependencies, list):
            raise ValueError(
                f"{formula_id}.derived_from_formula_ids must be a JSON array"
            )
        clean_dependencies: list[str] = []
        for dependency_id in dependencies:
            if (
                not isinstance(dependency_id, str)
                or dependency_id == formula_id
                or dependency_id not in by_id
            ):
                formula.setdefault("warnings", []).append(
                    f"Ignored invalid derived_from_formula_ids entry: {dependency_id!r}"
                )
                formula["requires_source_check"] = True
                continue
            if dependency_id not in clean_dependencies:
                clean_dependencies.append(dependency_id)
        formula["derived_from_formula_ids"] = clean_dependencies

        for symbol in formula.get("symbols", []):
            if not isinstance(symbol, dict):
                continue
            introduced_in = symbol.get("introduced_in_formula_id", "")
            if introduced_in and introduced_in not in by_id:
                symbol["introduced_in_formula_id"] = ""
                formula.setdefault("warnings", []).append(
                    f"Ignored invalid symbol introduction formula ID: {introduced_in!r}"
                )
                formula["requires_source_check"] = True

    for formula_id, formula in by_id.items():
        for dependency_id in formula["derived_from_formula_ids"]:
            by_id[dependency_id]["supports_formula_ids"].append(formula_id)

    return formulas


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

    The stats record isn't immediately queryable right after the chat completion
    response comes back (eventual consistency on OpenRouter's side, undocumented
    delay), so a 404 on the first attempt is expected and retried with backoff
    rather than treated as a real failure.
    """
    last_exc: Exception | None = None
    for attempt, delay in enumerate((0, 2, 5, 10, 15)):
        if delay:
            time.sleep(delay)
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
            return
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
    log.warning("Could not fetch generation stats for %s: %s", generation_id, last_exc)


def extract_formulas_llm(
    markdown: str,
    api_key: str,
    model: str = _MODEL,
    max_tokens: int = _MAX_COMPLETION_TOKENS,
    providers: list[str] | None = None,
    reasoning_max_tokens: int = _REASONING_MAX_TOKENS,
) -> list[dict]:
    """Send *markdown* to OpenRouter and return extracted formulas as a list of dicts.

    *providers*, if given, pins the OpenRouter request to that ordered list of
    upstream providers (e.g. ["cloudflare", "novita"]) with no fallback to
    others — useful since the same model can have very different completion
    token caps and reasoning-token behavior depending on which provider serves it.

    *reasoning_max_tokens* caps hidden reasoning so reasoning-capable models don't
    spend their whole completion budget "thinking" and leave nothing for the actual
    JSON array (observed with z-ai/glm-4.7-flash: 14374 of a 16384-token provider cap
    went to reasoning, leaving an empty `content` field) — but too low a cap can
    instead starve a model of planning room and cause early, sparse extraction
    (observed with x-ai/grok-4.3: stopped after 2 formulas using only a fraction of
    a 2000-token reasoning budget). Providers that don't support reasoning controls
    ignore this field.

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
        "reasoning": {"max_tokens": reasoning_max_tokens},
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
    except json.JSONDecodeError as exc:
        snippet_start = max(0, exc.pos - 300)
        raise ValueError(
            f"OpenRouter response body is not valid JSON ({exc}). This usually means the "
            "response came back as SSE/streamed chunks or was cut off mid-transfer rather "
            f"than a single JSON object. Body around the failure: {resp.text[snippet_start:exc.pos + 300]!r}"
        ) from exc
    try:
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
    return _derive_metadata(_extract_json_array(content))


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
    openrouter_reasoning_tokens = int(
        os.environ.get("OPENROUTER_REASONING_TOKENS", str(_REASONING_MAX_TOKENS)).strip()
    )

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
            markdown,
            openrouter_key,
            openrouter_model,
            openrouter_max_tokens,
            openrouter_providers,
            openrouter_reasoning_tokens,
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
