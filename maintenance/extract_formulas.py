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
    OPENROUTER_MODEL    model to use. Required unless OLLAMA_MODEL is set — see
                            "Backend selection" below.
    OPENROUTER_MAX_TOKENS  completion token cap sent to OpenRouter (default: 131072)
    OPENROUTER_PROVIDER    comma-separated provider slug order to pin to, no fallback
                            (e.g. "cloudflare,novita"; case-insensitive); default lets
                            OpenRouter choose
    OPENROUTER_REASONING_TOKENS  cap on hidden reasoning tokens sent to OpenRouter
                            (default: 8000)

    OLLAMA_URL          base URL of an OpenAI-compatible Ollama/LiteLLM proxy, e.g.
                            "https://tllm.science-berlin.de" ("/chat/completions" is
                            appended automatically). When OLLAMA_MODEL is set, this
                            backend is used instead of OpenRouter.
    OLLAMA_API_TOKEN    bearer token for OLLAMA_URL
    OLLAMA_MODEL        model id to request from OLLAMA_URL, e.g. "zib/gpt-oss-20b".
                            Setting this switches the script from OpenRouter to the
                            Ollama/LiteLLM backend.
    OLLAMA_DISABLE_THINKING  manage the target model's hidden reasoning automatically
                            (default: true; set to "false" to leave reasoning at
                            whatever the model/deployment defaults to). The actual
                            field sent is chosen per model via
                            `_reasoning_control_payload()`, not a single hardcoded
                            convention: a vLLM-backed "thinking" model (e.g.
                            Qwen3-style) reliably gets
                            chat_template_kwargs.enable_thinking=false. A model
                            using OpenAI's reasoning_effort convention instead
                            (e.g. gpt-oss) gets nothing by default — testing found
                            no reasoning_effort value reliable for it on this
                            deployment; see OLLAMA_REASONING_EFFORT to opt into one
                            anyway. Only applies to the Ollama/LiteLLM backend.
    OLLAMA_REASONING_EFFORT  value to send for reasoning_effort (e.g. "high") on a
                            non-vLLM model that supports it. Unset by default —
                            deliberately not auto-applied, since it was observed to
                            fix one paper's empty-output failure but break a
                            previously-reliable paper on 3 of 4 identical retries
                            for zib/gpt-oss-20b. Use as a manual retry knob when a
                            run returns empty content, not a default-on setting.

    The Ollama/LiteLLM backend has no max-tokens env var: `preflight_check()` (see
    below) determines the request's max_tokens dynamically per paper, and fails
    the run outright rather than sending a request likely to be truncated.

Backend selection:
    OLLAMA_MODEL set             -> Ollama/LiteLLM backend (OLLAMA_URL, OLLAMA_API_TOKEN required).
    OLLAMA_MODEL unset, OPENROUTER_MODEL set -> OpenRouter backend (OPENROUTER_API_KEY required).
    Neither set                  -> exits with an error; a model must be chosen explicitly.
"""
from __future__ import annotations

import argparse
import enum
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field

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
enrichment fields and other optional metadata first. Never omit a formula
record merely to save output space.

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

Do not rename variables during deduplication.

## 4. LaTeX normalization

For `latex`:

* preserve the mathematical meaning and original symbols;
* remove presentation-only commands such as `\displaystyle`;
* normalize whitespace;
* normalize equivalent delimiter commands when safe;
* retain equation environments only when structurally necessary;
* do not expand, simplify, rearrange, or algebraically transform the formula;
* do not silently repair malformed source LaTeX.

Every `occurrences[].latex_as_found` value must preserve the corresponding
source expression verbatim. If the Markdown appears corrupted, keep `latex`
conservative and preserve the damaged source exactly in `latex_as_found`.

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

Use `classification` for origin and novelty:

* `standard`: established mathematical knowledge not attributed here to a
  specific source;
* `cited_result`: explicitly attributed to earlier work;
* `paper_result`: a headline or independently meaningful supporting result
  introduced, derived, or proved by this paper;
* `paper_internal`: a definition, assumption, or derivation step specific to
  this paper;
* `adaptation`: an adjusted or specialized form of an established or cited
  formula;
* `unknown`: insufficient source evidence.

Preserve explicit citation identifiers and attribution wording. Do not infer
familiarity or attribution from the formula alone.

Set `established_name` to the established mathematical name explicitly used by
the paper, such as `"Peano kernel theorem"` or `"Jacobi operator"`. A name may
refer to the formula itself or to the mathematical object explicitly defined by
the formula. Do not copy the name of a nearby theorem, lemma, principle, or
method when it merely justifies, uses, or follows the formula. Use an empty
string when the paper does not explicitly apply an established name; do not
create a descriptive name for this field. The ordinary `label` field may still
contain a generated descriptive label.

Use `statement_role` to describe how the formula functions in the paper:

* `headline_result`: a principal result emphasized by the paper;
* `supporting_result`: an independently meaningful result used to support the
  main argument, including cited or standard results used by the paper;
* `definition`: a definition of an object or notation;
* `assumption`: a hypothesis or condition imposed on later results;
* `derivation_step`: an intermediate proof or calculation step that is not an
  independently meaningful result.

Set `standalone_statement` to true for a definition, assumption, headline
result, or independently meaningful supporting result. Set it to false for an
intermediate derivation step.

In `derived_from_formula_ids`, record only formulas that are direct premises or
derivation inputs. Do not link formulas merely because they are nearby, share
symbols, or discuss the same topic.

The Markdown may place a source marker immediately before a formula:

`[Equation metadata: source_id=S1.E1; number=(1.1)]`

For every formula associated with such a marker, copy `source_id` exactly into
`occurrences[].source_id` and copy `number` exactly into
`occurrences[].equation_number`. When the marker says `number=unnumbered`, use
an empty `equation_number`. Do not infer, normalize, or renumber these values.
For a multi-row equation group, the marker applies to the complete group. If
an individual table or aligned row visibly carries its own explicit
parenthesized equation label, use that visible row label for the corresponding
occurrence while retaining the group's `source_id`.

## 6. Schema

Each array element must have this structure:

{
  "formula_id": "F0001",
  "label": "concise human-readable label",
  "latex": "normalized LaTeX",
  "formula_type": "definition",
  "classification": "standard",
  "statement_role": "definition",
  "standalone_statement": true,
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
  "symbols_defined": [
    {
      "symbol": "n",
      "represents": "number of mesh intervals",
      "type": "index",
      "is_paper_local": false
    }
  ],
  "related_concepts": [
    {
      "label": "Peano kernel",
      "relation": "uses"
    }
  ],
  "msc_codes_suggested": ["65D32"],
  "occurrences": [
    {
      "source_id": "S1.E1",
      "section": "1. Introduction and statement of the results",
      "equation_number": "(1.1)",
      "latex_as_found": "verbatim LaTeX at this occurrence"
    }
  ]
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

`classification` must be one of:

* `standard`
* `cited_result`
* `paper_result`
* `paper_internal`
* `adaptation`
* `unknown`

`provenance.citation_keys` must contain only citation identifiers explicitly
associated with the formula in the supplied paper. Preserve their source form,
for example `"12"` or `"Smith2020"`, and use an empty array when none is given.

`provenance.attribution_text` must preserve the concise source wording that
attributes the formula, theorem, or result. Use an empty string when there is no
explicit attribution.

`established_name` must contain only a name explicitly supported by the paper;
otherwise use an empty string.

`statement_role` must be one of:

* `headline_result`
* `supporting_result`
* `definition`
* `assumption`
* `derivation_step`

`standalone_statement` must be a JSON boolean and must agree with
`statement_role`: it is false only for `derivation_step`.

`derived_from_formula_ids` must contain only valid, non-duplicate `formula_id`
values from the same output array and must not contain the record's own ID.

`occurrences[].source_id` must contain the exact `source_id` from the associated
equation metadata marker, or an empty string when no marker is present.

`occurrences[].equation_number` must contain the exact parenthesized `number`
from the associated marker. Use an empty string when the marker says
`number=unnumbered` or no equation number is present.

Only include a symbol in `symbols_defined` when this formula explicitly defines
or introduces that symbol. Do not list every symbol merely used by the formula.

`symbols_defined[].type` must be one of:

* `variable`
* `constant`
* `operator`
* `function`
* `set`
* `index`
* `parameter`
* `functional`

`symbols_defined[].is_paper_local` must be a JSON boolean. Set it to true only for notation
introduced specifically for this paper's argument or construction, and false
for established mathematical notation and generic bound variables.

`conditions` must be an empty array when no conditions are stated or reliably inferred.

`msc_codes_suggested` must be empty when no formula-specific or concept-specific MSC assignment is reasonably supported.

`related_concepts` contains only concepts directly relevant to the formula.
Each entry must contain a concise `label` and a `relation` describing how the
formula relates to that concept. Use an empty array when no relation is
supported by the supplied paper.

## 8. Source fidelity

`occurrences[].latex_as_found` must reproduce the source formula verbatim,
including apparent errors. The normalized `latex` field may correct only
unambiguous presentation artifacts. Never silently repair mathematical content.

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


def _validate_and_clean_formula_records(formulas: list[dict]) -> list[dict]:
    """Validate references and keep only the compact extraction schema."""
    cleaned: list[dict] = []
    by_id: dict[str, dict] = {}

    for raw in formulas:
        formula_id = raw.get("formula_id")
        if not isinstance(formula_id, str) or not formula_id:
            raise ValueError("Every formula must have a non-empty string formula_id")
        if formula_id in by_id:
            raise ValueError(f"Duplicate formula_id: {formula_id}")

        provenance = raw.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}

        occurrences = raw.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError(f"{formula_id}.occurrences must be a non-empty JSON array")

        record = {
            "formula_id": formula_id,
            "label": raw.get("label", ""),
            "latex": raw.get("latex", ""),
            "formula_type": raw.get("formula_type", "equation"),
            "classification": raw.get("classification", "unknown"),
            "statement_role": raw.get("statement_role", "derivation_step"),
            "standalone_statement": raw.get("standalone_statement", False),
            "provenance": {
                "citation_keys": provenance.get("citation_keys", []),
                "attribution_text": provenance.get("attribution_text", ""),
            },
            "established_name": raw.get("established_name", ""),
            "derived_from_formula_ids": raw.get("derived_from_formula_ids", []),
            "conditions": raw.get("conditions", []),
            "symbols_defined": raw.get("symbols_defined", []),
            "related_concepts": [
                {
                    "label": concept.get("label", ""),
                    "relation": concept.get("relation", ""),
                }
                for concept in raw.get("related_concepts", [])
                if isinstance(concept, dict)
            ],
            "msc_codes_suggested": raw.get("msc_codes_suggested", []),
            "occurrences": [
                {
                    "source_id": occurrence.get("source_id", ""),
                    "section": occurrence.get("section", ""),
                    "equation_number": occurrence.get("equation_number", ""),
                    "latex_as_found": occurrence.get("latex_as_found", ""),
                }
                for occurrence in occurrences
                if isinstance(occurrence, dict)
            ],
        }
        if not record["occurrences"]:
            raise ValueError(f"{formula_id}.occurrences contains no JSON objects")
        if not isinstance(record["latex"], str) or not record["latex"]:
            raise ValueError(f"{formula_id}.latex must be a non-empty string")
        if record["statement_role"] == "derivation_step":
            record["standalone_statement"] = False

        cleaned.append(record)
        by_id[formula_id] = record

    for formula_id, record in by_id.items():
        dependencies = record["derived_from_formula_ids"]
        if not isinstance(dependencies, list):
            raise ValueError(
                f"{formula_id}.derived_from_formula_ids must be a JSON array"
            )
        valid_dependencies: list[str] = []
        for dependency_id in dependencies:
            if (
                not isinstance(dependency_id, str)
                or dependency_id == formula_id
                or dependency_id not in by_id
            ):
                log.warning(
                    "Ignoring invalid %s.derived_from_formula_ids entry: %r",
                    formula_id,
                    dependency_id,
                )
                continue
            if dependency_id not in valid_dependencies:
                valid_dependencies.append(dependency_id)
        record["derived_from_formula_ids"] = valid_dependencies

    return cleaned


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


_EQUATION_MARKER_RE = re.compile(r"\[Equation metadata: source_id=([^;]+); number=([^\]]+)\]")
# Fallback for Markdown without embedded source markers: count display-math blocks directly.
_DISPLAY_MATH_RE = re.compile(
    r"\$\$.+?\$\$"
    r"|\\\[.+?\\\]"
    r"|\\begin\{(?:equation|align|gather|multline)\*?\}.+?\\end\{(?:equation|align|gather|multline)\*?\}",
    re.DOTALL,
)


def count_potential_equations(markdown: str) -> int:
    """Estimate how many formula records an extraction pass will produce.

    Prefers the `[Equation metadata: source_id=...; number=...]` markers the
    HTML-to-Markdown conversion embeds immediately before each formula, since that
    is the same segmentation the extraction prompt is instructed to follow (one
    record per marker, modulo grouping/splitting). Falls back to counting raw
    display-math blocks (`$$...$$`, `\\[...\\]`, `equation`/`align`/`gather`/
    `multline` environments) for Markdown that has no embedded markers.

    Args:
        markdown: The paper's fulltext Markdown.

    Returns:
        The estimated number of formula records, used as the driver for
        `preflight_check`'s output-token estimate.
    """
    marker_ids = {m[0].strip() for m in _EQUATION_MARKER_RE.findall(markdown)}
    if marker_ids:
        return len(marker_ids)
    return len(_DISPLAY_MATH_RE.findall(markdown))


# Conservative max_tokens used when a model's backend can't be confirmed as vLLM (see
# _is_vllm_backed) and the capacity gate is therefore skipped. 60000 comfortably covers
# both the default (no explicit reasoning_effort — completed Q7269940/Q7269937 fine at
# far less than this) and an OLLAMA_REASONING_EFFORT override, which spends real budget
# on visible reasoning before the JSON array and needs the extra headroom (60000 is what
# produced a genuine, if truncated, result on the largest test paper: Q7270050, 88
# formulas, with reasoning_effort="high").
_UNVERIFIED_BACKEND_MAX_TOKENS = 60000

# Per (api_root, model) — one throwaway request, so cache it and share it between
# _is_vllm_backed and _reasoning_control_payload rather than fetching twice.
_model_info_cache: dict[tuple[str, str], dict] = {}


def _fetch_model_info_entry(api_root: str, api_key: str, model: str) -> dict:
    """Fetch and cache *model*'s entry from LiteLLM's `/model/info`.

    Returns `{}` on any lookup failure (network error, model missing from the
    response) — callers should treat that the same as "unknown," not raise.

    Args:
        api_root: Base URL of the LiteLLM/Ollama proxy (without a path suffix).
        api_key: Bearer token for that endpoint.
        model: Model id to look up.

    Returns:
        The matching entry from `/model/info`'s `data` array, or `{}` if it
        could not be found or fetched.
    """
    cache_key = (api_root, model)
    if cache_key in _model_info_cache:
        return _model_info_cache[cache_key]

    entry: dict = {}
    try:
        resp = requests.get(
            api_root.rstrip("/") + "/model/info",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        resp.raise_for_status()
        for candidate in resp.json().get("data", []):
            if candidate.get("model_name") == model:
                entry = candidate
                break
    except (requests.RequestException, ValueError) as exc:
        log.warning("Could not fetch /model/info entry for %s: %s", model, exc)

    _model_info_cache[cache_key] = entry
    return entry


def _is_vllm_backed(api_root: str, api_key: str, model: str) -> bool:
    """Determine whether *model* is served by vLLM, via LiteLLM's `/model/info`.

    `_discover_context_ceiling`'s probe relies on vLLM's behavior of validating
    `max_tokens` against its configured context window synchronously and
    rejecting an oversized value immediately with a descriptive error. A model
    routed to a plain Ollama backend instead (observed with `zib/gpt-oss-20b`,
    `litellm_params.model == "ollama/gpt-oss:20b"`) does not fail fast the same
    way — sending it `max_tokens=10_000_000` just made it start generating, and
    the probe request hung until it hit the client-side timeout instead of
    returning a parseable error. Since there's no reliable way to discover that
    backend's true ceiling through the API, capacity checking is skipped for it
    entirely (see `preflight_check`).

    `/model/info`'s `litellm_params.model` is prefixed with the underlying
    provider (e.g. `"hosted_vllm/..."` vs. `"ollama/..."`); only the
    `hosted_vllm` prefix is treated as vLLM. A missing/unfetchable entry
    conservatively returns False, since the fast-fail probe is only safe to run
    when the vLLM backend has been positively confirmed.

    Args:
        api_root: Base URL of the LiteLLM/Ollama proxy (without a path suffix).
        api_key: Bearer token for that endpoint.
        model: Model id to look up.

    Returns:
        True if `model` is confirmed vLLM-backed, False otherwise (including
        when this cannot be determined).
    """
    entry = _fetch_model_info_entry(api_root, api_key, model)
    return entry.get("litellm_params", {}).get("model", "").startswith("hosted_vllm/")


def _reasoning_control_payload(
    api_root: str,
    api_key: str,
    model: str,
    is_vllm: bool,
    reasoning_effort_override: str | None = None,
) -> dict:
    """Choose the request fields that steer *model*'s hidden reasoning.

    Reasoning-capable models don't share one API convention for controlling how
    much hidden reasoning they do:

    - vLLM-served "thinking" models (e.g. Qwen3-style) use
      `chat_template_kwargs.enable_thinking`. Setting it `false` reliably
      suppresses reasoning: reasoning otherwise competes with the actual JSON
      output for the same small token budget (observed with
      `zib/qwen3.6-35b-a3b`: reasoning left no room for output at all under a
      small `max_tokens`). Dispatched via the caller-supplied *is_vllm* — the
      same route decision `resolve_route()` makes once per run, rather than
      each function re-querying `_is_vllm_backed` independently — and always
      applies this fix for a confirmed vLLM backend.
    - OpenAI-style reasoning models (e.g. gpt-oss) instead expose a
      `reasoning_effort` parameter (`"low"`/`"medium"`/`"high"`) and silently
      ignore `chat_template_kwargs`. Unlike the vLLM case, this function does
      *not* set a value for this family by default: testing on `zib/gpt-oss-20b`
      found no reliable default. `reasoning_effort="low"` made the model give up
      and return an empty array; `reasoning_effort="high"` fixed one paper that
      otherwise always returned empty (Q7270050) but then made a *previously
      reliable* paper (Q7269940) fail empty on 3 of 4 identical retries — the
      model's behavior on this deployment appears to be unreliable regardless of
      the value chosen, not fixed by picking the "right" one. So the caller must
      opt in via *reasoning_effort_override* (e.g. from an env var) rather than
      this function guessing a default; leaving it unset here matches this
      family's previously-observed reliable behavior on most papers.

    Whether a model exposes `reasoning_effort` at all is read from `/model/info`
    → `model_info.supported_openai_params`, but only for non-vLLM backends: that
    field is *not* reliable for vLLM-backed models — LiteLLM appears to attach a
    generic capability list to the whole `hosted_vllm` provider class (it listed
    `reasoning_effort` for `zib/qwen3.6-35b-a3b` even though sending it there
    just hangs — the request never returned within 250s, unlike the working
    `chat_template_kwargs` call). For a genuinely Ollama-backed model (gpt-oss),
    by contrast, the same field reflects real capability.

    Args:
        api_root: Base URL of the LiteLLM/Ollama proxy (without a path suffix).
        api_key: Bearer token for that endpoint.
        model: Model id to look up.
        is_vllm: Whether *model* is confirmed vLLM-backed (from `resolve_route`).
        reasoning_effort_override: Value to send for `reasoning_effort` on a
            non-vLLM model that supports it (e.g. `"high"`). Only applied when
            given explicitly; otherwise no reasoning-effort field is sent at all
            for this model family, since no default was found reliable.

    Returns:
        A dict of extra fields to merge into the chat-completions request
        payload (empty if there's nothing to suppress/set).
    """
    if is_vllm:
        return {"chat_template_kwargs": {"enable_thinking": False}}

    if reasoning_effort_override:
        entry = _fetch_model_info_entry(api_root, api_key, model)
        supported = entry.get("model_info", {}).get("supported_openai_params") or []
        if "reasoning_effort" in supported:
            return {"reasoning_effort": reasoning_effort_override}
    return {}


# Per (chat_url, model) — discovering the ceiling costs one throwaway request, so cache it.
_context_ceiling_cache: dict[tuple[str, str], int] = {}

_CEILING_ERROR_PATTERNS = (
    re.compile(r"max_total_tokens=(\d+)"),
    re.compile(r"maximum context length is (\d+) tokens"),
    re.compile(r"max_model_len[\"']?\s*[:=]\s*(\d+)"),
)


def _discover_context_ceiling(chat_url: str, api_key: str, model: str) -> int:
    """Find a model's true total context window (input + output combined) by
    deliberately overflowing it.

    There is no OpenAI-compatible endpoint that reports a model's context window
    directly, but vLLM (and LiteLLM proxies in front of it) reject a request whose
    `max_tokens` alone exceeds that window with a 400 error naming the exact
    ceiling, e.g. `"...max_tokens=131072 cannot be greater than
    max_model_len=max_total_tokens=65536..."`. Sending a trivial one-token prompt
    with a deliberately absurd `max_tokens` (10,000,000) triggers that validation
    error immediately, before any real inference runs, so the probe is cheap.

    The result is cached per `(chat_url, model)` for the life of the process,
    since it costs one throwaway request and does not change between calls.

    Args:
        chat_url: Full chat-completions endpoint URL.
        api_key: Bearer token for that endpoint.
        model: Model id to probe.

    Returns:
        The model's total context window in tokens.

    Raises:
        ValueError: If the ceiling could not be parsed from the server's response
            (e.g. the absurd `max_tokens` was, implausibly, accepted, or the error
            text doesn't match any known phrasing).
    """
    cache_key = (chat_url, model)
    if cache_key in _context_ceiling_cache:
        return _context_ceiling_cache[cache_key]

    resp = requests.post(
        chat_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10_000_000,
        },
        timeout=30,
    )
    for pattern in _CEILING_ERROR_PATTERNS:
        match = pattern.search(resp.text)
        if match:
            ceiling = int(match.group(1))
            _context_ceiling_cache[cache_key] = ceiling
            return ceiling
    raise ValueError(
        f"Could not determine context ceiling for model={model!r} from probe response "
        f"(HTTP {resp.status_code}): {resp.text[:300]!r}"
    )


def _count_prompt_tokens(api_root: str, api_key: str, model: str, system_prompt: str, markdown: str) -> int:
    """Count the exact token cost of the system prompt + paper Markdown via
    LiteLLM's `/utils/token_counter` endpoint.

    This avoids bundling a separate tokenizer library client-side. The endpoint
    has been observed to fall back to a generic tokenizer (`tokenizer_type:
    "openai_tokenizer"`) rather than the model's real one when LiteLLM has no
    tokenizer mapping registered for a custom model id — verified against a real
    completion's `usage.prompt_tokens` as ~4% low (27,356 estimated vs. 28,514
    actual), which is treated as close enough for a capacity check, not an exact
    accounting.

    Args:
        api_root: Base URL of the LiteLLM/Ollama proxy (without a path suffix).
        api_key: Bearer token for that endpoint.
        model: Model id to count tokens for.
        system_prompt: The extraction system prompt.
        markdown: The paper's fulltext Markdown (sent as the user message).

    Returns:
        The estimated prompt token count.
    """
    resp = requests.post(
        api_root.rstrip("/") + "/utils/token_counter",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": markdown},
            ],
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["total_tokens"]


# Calibrated from real completion_tokens/record ratios observed on this schema: ~540
# tokens/record (Q7266546, 13 records) and ~444 tokens/record (Q7270050, 45 records
# salvaged from a truncated response). 600 rounds up from both with a small cushion.
_TOKENS_PER_EQUATION_RECORD = 600
_OUTPUT_TOKEN_FIXED_OVERHEAD = 200
_OUTPUT_TOKEN_SAFETY_MARGIN = 1.3


def preflight_check(
    markdown: str,
    api_root: str,
    api_key: str,
    model: str,
    is_vllm: bool,
    system_prompt: str = _SYSTEM_PROMPT,
) -> int:
    """Verify a self-hosted model has enough context budget for this paper before
    spending minutes on a request that would otherwise come back truncated or
    empty, and compute the `max_tokens` value to request.

    Self-hosted Ollama/LiteLLM deployments (unlike OpenRouter) can have a small,
    fixed *total* context window shared between input and output — e.g. 65536
    tokens for `zib/qwen3.6-35b-a3b`. Two distinct failures follow from that:

    1. A `max_tokens` value that alone exceeds the ceiling is rejected outright,
       regardless of how small the prompt is (seen with the OpenRouter default of
       131072 against a 65536-token ceiling).
    2. `prompt_tokens + max_tokens` exceeding the ceiling is rejected once the
       prompt is large enough, or — if `max_tokens` is set conservatively low
       enough to avoid that — the model instead runs out of room mid-extraction
       and returns a truncated result (`finish_reason=length`), as happened on
       Q7270050: 88 equation markers needed far more output than a fixed 20000
       token cap allowed, and the model was cut off having covered only 48.9% of
       them.

    This function computes both piece of the budget and either raises (the
    request has no realistic chance of succeeding, so don't send it) or returns
    the `max_tokens` value to request:

    - **Prompt tokens** via `_count_prompt_tokens` (LiteLLM's `/utils/token_counter`
      endpoint, ~4% accurate against real `usage.prompt_tokens` in testing).
    - **Estimated output tokens** as
      `(200 + n_equations * 600) * 1.3`, where `n_equations` comes from
      `count_potential_equations` and 600 tokens/record is calibrated from real
      `completion_tokens / len(formulas)` ratios observed in testing (~540 and
      ~444 tokens/record across two full-paper runs), 200 is a fixed overhead for
      JSON structure, and 1.3x is a safety margin for verbosity variance.
    - **The ceiling** via `_discover_context_ceiling`, probed dynamically rather
      than hardcoded, since it is a property of how the model was deployed
      (vLLM's `--max-model-len`), not something implied by the model id.

    The ceiling probe only works against a vLLM backend, which fails fast and
    descriptively on an oversized `max_tokens`. A model routed to plain Ollama
    doesn't fail fast the same way (observed with `zib/gpt-oss-20b`: the probe
    just hung until timeout instead of erroring), so the caller-supplied
    *is_vllm* (from `resolve_route()`, which determines this once per run rather
    than each function re-querying `_is_vllm_backed` independently) gates
    whether the probe runs at all; when it isn't confirmed vLLM, this function
    skips the gate entirely and returns `_UNVERIFIED_BACKEND_MAX_TOKENS`
    unchecked rather than risk a hung probe.

    Args:
        markdown: The paper's fulltext Markdown.
        api_root: Base URL of the LiteLLM/Ollama proxy (without a path suffix,
            e.g. "https://tllm.science-berlin.de").
        api_key: Bearer token for that endpoint.
        model: Model id to check capacity for.
        is_vllm: Whether *model* is confirmed vLLM-backed.
        system_prompt: The extraction system prompt that will accompany the
            paper in the real request (defaults to the module's `_SYSTEM_PROMPT`).

    Returns:
        The `max_tokens` value to request: the estimated output token need,
        capped at whatever room remains in the context window after the prompt —
        or the fixed `_UNVERIFIED_BACKEND_MAX_TOKENS` fallback, unchecked, when
        the backend isn't confirmed vLLM.

    Raises:
        ValueError: If the prompt alone meets or exceeds the ceiling, or if the
            prompt plus the estimated output need exceeds it — in either case the
            request should not be sent. Never raised for a non-vLLM backend,
            since capacity isn't checked in that case.
    """
    if not is_vllm:
        log.warning(
            "%s is not confirmed vLLM-backed; skipping capacity check and using a fixed "
            "max_tokens=%d (a hung ceiling-probe is worse than an unchecked request here).",
            model,
            _UNVERIFIED_BACKEND_MAX_TOKENS,
        )
        return _UNVERIFIED_BACKEND_MAX_TOKENS

    chat_url = api_root.rstrip("/") + "/chat/completions"
    ceiling = _discover_context_ceiling(chat_url, api_key, model)
    prompt_tokens = _count_prompt_tokens(api_root, api_key, model, system_prompt, markdown)

    if prompt_tokens >= ceiling:
        raise ValueError(
            f"Prompt alone ({prompt_tokens} tokens) meets or exceeds {model}'s context "
            f"ceiling ({ceiling} tokens); this paper cannot be processed by this model."
        )

    n_equations = count_potential_equations(markdown)
    estimated_output_tokens = int(
        (_OUTPUT_TOKEN_FIXED_OVERHEAD + n_equations * _TOKENS_PER_EQUATION_RECORD)
        * _OUTPUT_TOKEN_SAFETY_MARGIN
    )

    if prompt_tokens + estimated_output_tokens > ceiling:
        raise ValueError(
            f"{model} cannot handle this paper: {prompt_tokens} prompt tokens + an "
            f"estimated {estimated_output_tokens} output tokens (for ~{n_equations} "
            f"detected equations) = {prompt_tokens + estimated_output_tokens}, which "
            f"exceeds its {ceiling}-token context ceiling by "
            f"{prompt_tokens + estimated_output_tokens - ceiling} tokens."
        )

    return min(estimated_output_tokens, ceiling - prompt_tokens)


# No LiteLLM-style token-counter or context-ceiling-probe endpoint exists on a
# plain Ollama/Open WebUI deployment (confirmed against https://ollama.zib.de:
# /model/info and /utils/token_counter both 404 through the native API path).
# Since num_ctx is a value *we* choose for this route rather than one to be
# discovered, a fixed chars-per-token ratio is close enough for a sanity
# check — this guards against an obviously-too-small num_ctx, not an exact
# accounting.
_NATIVE_CHARS_PER_TOKEN = 3.5


def preflight_check_native(
    markdown: str,
    num_ctx: int,
    system_prompt: str = _SYSTEM_PROMPT,
) -> None:
    """Sanity-check that *markdown* is unlikely to overflow *num_ctx* on a
    native-Ollama route before spending minutes on a request.

    Unlike `preflight_check` (which probes a LiteLLM/vLLM deployment's true
    ceiling and computes a `max_tokens` to request), this route sets
    `num_ctx` directly in the request, so there is nothing to discover —
    only a cheap local estimate to catch an obviously-undersized value ahead
    of time rather than after the model returns truncated or empty content.

    Args:
        markdown: The paper's fulltext Markdown.
        num_ctx: The context window that will be requested via
            `options.num_ctx`.
        system_prompt: The extraction system prompt that will accompany the
            paper in the real request.

    Raises:
        ValueError: If the estimated prompt + output token need exceeds
            `num_ctx`.
    """
    prompt_chars = len(system_prompt) + len(markdown)
    estimated_prompt_tokens = int(prompt_chars / _NATIVE_CHARS_PER_TOKEN)

    n_equations = count_potential_equations(markdown)
    estimated_output_tokens = int(
        (_OUTPUT_TOKEN_FIXED_OVERHEAD + n_equations * _TOKENS_PER_EQUATION_RECORD)
        * _OUTPUT_TOKEN_SAFETY_MARGIN
    )

    total = estimated_prompt_tokens + estimated_output_tokens
    if total > num_ctx:
        raise ValueError(
            f"Estimated {estimated_prompt_tokens} prompt tokens + {estimated_output_tokens} "
            f"output tokens (for ~{n_equations} detected equations) = {total}, which exceeds "
            f"num_ctx={num_ctx}. Increase OLLAMA_NUM_CTX or use a paper with fewer equations."
        )


def extract_formulas_llm(
    markdown: str,
    api_key: str,
    model: str = _MODEL,
    max_tokens: int = _MAX_COMPLETION_TOKENS,
    providers: list[str] | None = None,
    reasoning_max_tokens: int = _REASONING_MAX_TOKENS,
    base_url: str = _OPENROUTER_URL,
    reasoning_control: dict | None = None,
) -> list[dict]:
    """Send *markdown* to an OpenAI-compatible chat-completions endpoint and return
    extracted formulas as a list of dicts.

    *base_url* defaults to OpenRouter. Passing a different endpoint (e.g. an
    Ollama/LiteLLM proxy's "/chat/completions") skips the OpenRouter-only
    "provider" pinning, "reasoning" budget, and post-hoc generation-stats lookup,
    since those are OpenRouter-specific extensions not understood by other
    OpenAI-compatible servers.

    *reasoning_control*, if given, is merged directly into the request payload —
    normally the output of `_reasoning_control_payload()`, which picks the right
    field (`chat_template_kwargs.enable_thinking` or `reasoning_effort`) for the
    target model rather than this function hardcoding one convention. On a
    self-hosted model with a small total context window (input + output combined,
    e.g. 65536 tokens), an unmanaged "thinking" model can spend the whole
    completion budget on invisible reasoning and return empty content for
    anything but a trivial paper.

    *providers*, if given, pins the OpenRouter request to that ordered list of
    upstream providers (e.g. ["cloudflare", "novita"]) with no fallback to
    others — useful since the same model can have very different completion
    token caps and reasoning-token behavior depending on which provider serves it.
    Ignored when *base_url* is not OpenRouter.

    *reasoning_max_tokens* caps hidden reasoning so reasoning-capable models don't
    spend their whole completion budget "thinking" and leave nothing for the actual
    JSON array (observed with z-ai/glm-4.7-flash: 14374 of a 16384-token provider cap
    went to reasoning, leaving an empty `content` field) — but too low a cap can
    instead starve a model of planning room and cause early, sparse extraction
    (observed with x-ai/grok-4.3: stopped after 2 formulas using only a fraction of
    a 2000-token reasoning budget). Providers that don't support reasoning controls
    ignore this field. Ignored when *base_url* is not OpenRouter.

    Raises ValueError if the response cannot be parsed as a JSON array.
    """
    is_openrouter = base_url.rstrip("/") == _OPENROUTER_URL.rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": markdown},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if is_openrouter:
        payload["reasoning"] = {"max_tokens": reasoning_max_tokens}
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
    if reasoning_control:
        payload.update(reasoning_control)
    log.info("Requesting model=%s base_url=%s provider=%s", model, base_url, payload.get("provider"))
    resp = requests.post(
        base_url,
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
    if is_openrouter and generation_id:
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
    return _validate_and_clean_formula_records(_extract_json_array(content))


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


class Backend(enum.Enum):
    """Which LLM backend a run uses, and — for Ollama — which capacity/reasoning
    handling applies. See `resolve_route`."""

    OPENROUTER = "openrouter"
    OLLAMA_VLLM = "ollama_vllm"
    OLLAMA_OTHER = "ollama_other"
    OLLAMA_NATIVE = "ollama_native"


@dataclass
class Route:
    """The fully-resolved backend route for one script invocation.

    Produced once by `resolve_route()` so that `preflight_check`,
    `_reasoning_control_payload`, and `extract_formulas_llm` each consume an
    already-decided route instead of independently re-deriving parts of it.
    Before this existed, the same question was answered three different ways in
    three different places: backend selection was read from env vars directly in
    `main()`; vLLM-vs-not was re-queried via `_is_vllm_backed` from two separate
    call sites; and OpenRouter-vs-not was inferred a third way, by comparing
    `base_url` against `_OPENROUTER_URL`, inside `extract_formulas_llm`.

    Attributes:
        backend: Which route this is.
        base_url: Full chat-completions endpoint URL to POST to.
        api_key: Bearer token for that endpoint.
        model: Model id to request.
        api_root: Base URL of the LiteLLM/Ollama proxy (without a path suffix),
            used for the preflight/reasoning-control API calls. `None` for
            OpenRouter, which has no equivalent side-channel endpoints.
        max_tokens: Fixed completion-token cap. Only meaningful for OpenRouter;
            `None` for Ollama, since `preflight_check` computes it per paper.
        providers: OpenRouter provider-pin order. Empty for Ollama.
        reasoning_max_tokens: OpenRouter reasoning-token cap. `0` for Ollama,
            which uses `_reasoning_control_payload` instead.
    """

    backend: Backend
    base_url: str
    api_key: str
    model: str
    api_root: str | None = None
    max_tokens: int | None = None
    providers: list[str] = field(default_factory=list)
    reasoning_max_tokens: int = 0
    num_ctx: int | None = None
    """Context window to request via native Ollama's `options.num_ctx`. Only
    meaningful for `Backend.OLLAMA_NATIVE` — the OpenAI-compat routes either
    don't support this field at all, or (per `preflight_check`) discover a
    ceiling dynamically instead. `None` for the other two backends."""


def resolve_route(
    *,
    ollama_model: str,
    ollama_url: str,
    ollama_token: str,
    ollama_api_style: str = "openai",
    ollama_num_ctx: int = 32768,
    openrouter_model: str,
    openrouter_key: str,
    openrouter_max_tokens: int,
    openrouter_providers: list[str],
    openrouter_reasoning_tokens: int,
) -> Route:
    """Decide which backend/route this run should use — the single place that
    answers "which route to take," so nothing downstream has to re-derive it.

    Precedence: `ollama_model` set -> Ollama backend (requires `ollama_url`
    and `ollama_token`). `ollama_api_style` then picks *how* to talk to it:
    `"native"` -> `Backend.OLLAMA_NATIVE`, talking to `{ollama_url}/chat` in
    Ollama's own request/response shape with an explicit `num_ctx` (for
    deployments — e.g. plain Open WebUI — that have no LiteLLM-compatible
    `/model/info`/`/utils/token_counter` and were observed to silently
    truncate prompts on the OpenAI-compatible route instead of honoring
    `num_ctx` there). Any other value (default `"openai"`) preserves the
    existing behavior: vLLM-vs-not is determined once via `_is_vllm_backed`,
    a single `/model/info` lookup, cached for the process. Otherwise
    `openrouter_model` set -> OpenRouter backend (requires `openrouter_key`).
    Neither set is an error: a model must be chosen explicitly, there is no
    implicit default backend or model.

    Args:
        ollama_model: `OLLAMA_MODEL` env var.
        ollama_url: `OLLAMA_URL` env var.
        ollama_token: `OLLAMA_API_TOKEN` env var.
        ollama_api_style: `OLLAMA_API_STYLE` env var (`"openai"` or
            `"native"`). Ignored unless `ollama_model` is set.
        ollama_num_ctx: Parsed `OLLAMA_NUM_CTX` value. Only used when
            `ollama_api_style == "native"`.
        openrouter_model: `OPENROUTER_MODEL` env var.
        openrouter_key: `OPENROUTER_API_KEY` env var.
        openrouter_max_tokens: Parsed `OPENROUTER_MAX_TOKENS` value.
        openrouter_providers: Parsed `OPENROUTER_PROVIDER` list.
        openrouter_reasoning_tokens: Parsed `OPENROUTER_REASONING_TOKENS` value.

    Returns:
        The resolved `Route`: `Backend.OLLAMA_NATIVE`, `Backend.OLLAMA_VLLM`,
        or `Backend.OLLAMA_OTHER` for the Ollama backend, or `Backend.OPENROUTER`.

    Raises:
        ValueError: If neither `ollama_model` nor `openrouter_model` is set, or
            the backend selected by precedence is missing a required
            credential.
    """
    if ollama_model:
        if not ollama_url or not ollama_token:
            raise ValueError(
                "OLLAMA_URL and OLLAMA_API_TOKEN are required when OLLAMA_MODEL is set."
            )
        if ollama_api_style == "native":
            return Route(
                backend=Backend.OLLAMA_NATIVE,
                base_url=ollama_url.rstrip("/") + "/chat",
                api_key=ollama_token,
                model=ollama_model,
                api_root=ollama_url,
                num_ctx=ollama_num_ctx,
            )
        backend = (
            Backend.OLLAMA_VLLM
            if _is_vllm_backed(ollama_url, ollama_token, ollama_model)
            else Backend.OLLAMA_OTHER
        )
        return Route(
            backend=backend,
            base_url=ollama_url.rstrip("/") + "/chat/completions",
            api_key=ollama_token,
            model=ollama_model,
            api_root=ollama_url,
        )

    if openrouter_model:
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY is required when OPENROUTER_MODEL is set.")
        return Route(
            backend=Backend.OPENROUTER,
            base_url=_OPENROUTER_URL,
            api_key=openrouter_key,
            model=openrouter_model,
            max_tokens=openrouter_max_tokens,
            providers=openrouter_providers,
            reasoning_max_tokens=openrouter_reasoning_tokens,
        )

    raise ValueError(
        "Set OLLAMA_MODEL (self-hosted) or OPENROUTER_MODEL (OpenRouter) to select a backend."
    )


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
    openrouter_model = os.environ.get("OPENROUTER_MODEL", "").strip()
    openrouter_max_tokens = int(
        os.environ.get("OPENROUTER_MAX_TOKENS", str(_MAX_COMPLETION_TOKENS)).strip()
    )
    openrouter_providers = [
        p.strip() for p in os.environ.get("OPENROUTER_PROVIDER", "").split(",") if p.strip()
    ]
    openrouter_reasoning_tokens = int(
        os.environ.get("OPENROUTER_REASONING_TOKENS", str(_REASONING_MAX_TOKENS)).strip()
    )
    ollama_url = os.environ.get("OLLAMA_URL", "").strip()
    ollama_token = os.environ.get("OLLAMA_API_TOKEN", "").strip()
    ollama_model = os.environ.get("OLLAMA_MODEL", "").strip()
    ollama_disable_thinking = os.environ.get("OLLAMA_DISABLE_THINKING", "true").strip().lower() not in (
        "0",
        "false",
        "",
    )
    ollama_reasoning_effort = os.environ.get("OLLAMA_REASONING_EFFORT", "").strip() or None

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

    # --- Stage 4: resolve backend/route, then validate requested QID ---
    try:
        route = resolve_route(
            ollama_model=ollama_model,
            ollama_url=ollama_url,
            ollama_token=ollama_token,
            openrouter_model=openrouter_model,
            openrouter_key=openrouter_key,
            openrouter_max_tokens=openrouter_max_tokens,
            openrouter_providers=openrouter_providers,
            openrouter_reasoning_tokens=openrouter_reasoning_tokens,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    target = args.qid.upper()
    session = requests.Session()

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

    reasoning_control = None
    if route.backend != Backend.OPENROUTER:
        is_vllm = route.backend == Backend.OLLAMA_VLLM
        log.info("Checking %s's capacity for this paper…", route.model)
        try:
            route.max_tokens = preflight_check(markdown, route.api_root, route.api_key, route.model, is_vllm)
        except (ValueError, requests.RequestException) as exc:
            sys.exit(f"Preflight check failed: {exc}")
        log.info("Preflight OK — requesting max_tokens=%d", route.max_tokens)
        if ollama_disable_thinking:
            reasoning_control = _reasoning_control_payload(
                route.api_root, route.api_key, route.model, is_vllm, ollama_reasoning_effort
            )
            log.info("Reasoning control for %s: %s", route.model, reasoning_control)

    # --- Stage 6: call LLM ---
    log.info("Sending to %s via %s…", route.model, route.backend.value)
    try:
        formulas = extract_formulas_llm(
            markdown,
            route.api_key,
            route.model,
            route.max_tokens,
            route.providers,
            route.reasoning_max_tokens,
            base_url=route.base_url,
            reasoning_control=reasoning_control,
        )
    except (ValueError, requests.RequestException) as exc:
        sys.exit(f"LLM extraction failed: {exc}")
    log.info("Extracted %d formula(s)", len(formulas))

    # --- Stage 7: save output ---
    out_path = _output_path(target, route.model)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(formulas, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(formulas)} formula(s) to {out_path}")


if __name__ == "__main__":
    main()
