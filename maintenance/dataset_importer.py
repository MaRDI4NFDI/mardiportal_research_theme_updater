"""
Map a dataset search results JSON file to MaRDI KG dataset item payloads
and optionally import them into the MaRDI Knowledge Graph.

Expected input structure for 'convert':
  { "datasets": [ { "title", "description", "authors", "licenses",
                     "links", "unique_identifier", "updated_date",
                     "dataset_id", ... }, ... ] }

Subcommands:
  convert         Read input JSON, map fields, write a KG-ready JSON file.
  import_to_mardi Read the KG-ready JSON and create items in the MaRDI KG.

Field mapping (convert):
  label   ← title
  P31     = Q56885   (data set) — fixed
  P1460   = Q5984635 (MaRDI dataset profile) — fixed
  P43     ← authors[]          (author name string)
  P1459   ← description        (HTML-decoded, trailing boilerplate stripped)
  P205    ← links[]            (full work available at URL)
  P27     ← unique_identifier  (DOI, stripped to bare form)
  P1476   ← updated_date       (ISO 8601 time, date precision)
  P163    ← licenses[]         (license item QID, resolved via LICENSE_QID_MAP)
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Known license strings → MaRDI KG QIDs.
# Extend this map when new license strings appear in future input files.
LICENSE_QID_MAP: dict[str, str | None] = {
    "MIT License":                                                  "Q56842",
    "Apache License, v2.0":                                        "Q56870",
    "Attribution 4.0 (CC BY 4.0)":                                "Q57056",
    "Attribution-ShareAlike 4.0 (CC BY-SA 4.0)":                  "Q57038",
    "Attribution-NonCommercial 4.0 (CC BY-NC 4.0)":               "Q57074",
    "Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)": "Q57078",
    "Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0)": "Q57076",
    "https://choosealicense.com/licenses/cc0-1.0/":                "Q56468",   # CC0
    "https://creativecommons.org/publicdomain/zero/1.0/":          "Q56468",   # CC0
    "https://choosealicense.com/licenses/gpl-3.0/":                "Q56621",   # GPL v3
    "https://www.gnu.org/licenses/gpl-3.0.html":                   "Q56621",   # GPL v3
    # No KG item found — skipped during import:
    "http://www.gnu.org/licenses/fdl-1.3.html":                    None,       # GNU FDL 1.3
    "https://choosealicense.com/licenses/other/":                   None,       # unspecified
    "http://vocab.nerc.ac.uk/collection/L08/current/CB/":          None,       # domain-specific vocab
}

# Properties whose values are Wikibase item QIDs.
_ITEM_PROPS = {"P31", "P1460", "P163"}
# Properties whose values are ISO 8601 time strings (e.g. "+2021-03-05T00:00:00Z").
_TIME_PROPS = {"P1476"}
# All other properties are stored as plain strings (string, url, external-id).

_MEDIAWIKI_API_URL = "https://portal.mardi4nfdi.de/w/api.php"


# Title keywords that indicate the entry is not a real dataset (case-insensitive).
# Extend when new false-positive patterns appear in future input files.
_NON_DATASET_KEYWORDS = [
    "leaderboard",
    "price-performance tracker",
    "price tracker",
]


def _is_non_dataset(title: str) -> bool:
    t = title.lower()
    return any(kw in t for kw in _NON_DATASET_KEYWORDS)


# ---------------------------------------------------------------------------
# Shared mapping helpers
# ---------------------------------------------------------------------------

def clean_description(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"\s*See the full description on the dataset page:.*$", "", text, flags=re.DOTALL)
    # Wikibase rejects vertical whitespace in string values — collapse to spaces.
    text = re.sub(r"[\r\n\t\v\f]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_doi(uid: str | None) -> str | None:
    if uid and uid.startswith("https://doi.org/"):
        return uid[len("https://doi.org/"):]
    return None


def extract_hf_id(links: list[str]) -> str | None:
    """Return the first Hugging Face dataset slug (owner/name) found in links."""
    for url in links:
        m = re.match(r"https://huggingface\.co/datasets/([^/?#]+/[^/?#]+)", url)
        if m:
            return m.group(1)
    return None


def extract_kaggle_id(links: list[str]) -> str | None:
    """Return the first Kaggle dataset slug (owner/name) found in links."""
    for url in links:
        m = re.match(r"https://(?:www\.)?kaggle\.com/datasets/([^/?#]+/[^/?#]+)", url)
        if m:
            return m.group(1)
    return None


def extract_zenodo_id(links: list[str], unique_identifier: str | None = None) -> str | None:
    """Return the first Zenodo numeric record ID found in links or DOI."""
    for url in links:
        m = re.search(r"zenodo\.org/(?:records?|deposit)/(\d+)", url)
        if m:
            return m.group(1)
    if unique_identifier:
        m = re.search(r"10\.5281/zenodo\.(\d+)", unique_identifier)
        if m:
            return m.group(1)
    return None


def fetch_hf_description(hf_id: str) -> str | None:
    """Fetch and clean the full description from a HuggingFace dataset README.

    Returns a single-line string capped at 1500 chars, or None if unavailable.
    """
    try:
        import requests as _requests
        url = f"https://huggingface.co/datasets/{hf_id}/resolve/main/README.md"
        r = _requests.get(url, allow_redirects=True, timeout=15,
                          headers={"User-Agent": "MaRDI-dataset-import/1.0"})
        if r.status_code != 200:
            return None
        text = r.text
    except Exception:
        return None

    # Strip YAML frontmatter (allow optional leading whitespace/BOM before ---)
    text = re.sub(r"^\s*---.*?---\s*", "", text, flags=re.DOTALL)
    # Strip fenced code blocks entirely (complete and unclosed)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*", "", text, flags=re.DOTALL)  # unclosed fence → strip to end
    text = re.sub(r"`[^`]+`", "", text)
    # Strip HTML tags (e.g. <br>, <table>)
    text = re.sub(r"<[^>]+>", " ", text)
    # Strip markdown table rows (lines containing |) and separator lines (---|)
    text = re.sub(r"^[ \t]*\|.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*[-:]+[ \t]*$", "", text, flags=re.MULTILINE)
    # Resolve markdown links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Strip heading markers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Strip emphasis markers (bold/italic only, preserve underscores in identifiers)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_{2}([^_]+)_{2}", r"\1", text)
    text = re.sub(r"_([^_\s][^_]*)_", r"\1", text)
    # Collapse all vertical whitespace to single space
    text = re.sub(r"[\r\n\t\v\f]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    text = html.unescape(text).strip()

    if len(text) > 1500:
        cut = text.rfind(". ", 0, 1500)
        text = text[:cut + 1] if cut > 0 else text[:1500]

    return text if len(text) > 50 else None


def map_item(d: dict) -> dict:
    claims: dict = {
        "P31": "Q56885",
        "P1460": "Q5984635",
    }

    authors = [a for a in (d.get("authors") or []) if a]
    if authors:
        claims["P43"] = authors

    desc = clean_description(d.get("description", ""))
    if desc:
        claims["P1459"] = desc

    links = [l for l in (d.get("links") or []) if l]
    if links:
        claims["P205"] = links
        hf_id = extract_hf_id(links)
        if hf_id:
            claims["P1991"] = hf_id
        kaggle_id = extract_kaggle_id(links)
        if kaggle_id:
            claims["P1992"] = kaggle_id
        zenodo_id = extract_zenodo_id(links, d.get("unique_identifier"))
        if zenodo_id:
            claims["P227"] = zenodo_id

    tldr = d.get("tldr", "").strip()
    if tldr:
        claims["P1963"] = tldr

    doi = extract_doi(d.get("unique_identifier"))
    if doi:
        claims["P27"] = doi

    date = d.get("updated_date")
    if date:
        try:
            claims["P1476"] = (
                datetime.fromisoformat(date.replace(" +00:00", "+00:00"))
                .strftime("+%Y-%m-%dT00:00:00Z")
            )
        except ValueError:
            claims["P1476"] = date

    licenses = [l for l in (d.get("licenses") or []) if l]
    license_qids = [LICENSE_QID_MAP[l] for l in licenses if LICENSE_QID_MAP.get(l)]
    unresolved_licenses = [
        l for l in licenses
        if l not in LICENSE_QID_MAP or LICENSE_QID_MAP[l] is None
    ]
    if license_qids:
        claims["P163"] = license_qids[0] if len(license_qids) == 1 else license_qids

    return {
        "label": d["title"],
        "claims": claims,
        "unresolved": {
            "licenses": unresolved_licenses,
        },
    }


# ---------------------------------------------------------------------------
# Subcommand: convert
# ---------------------------------------------------------------------------

def cmd_convert(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output) if args.output
        else input_path.with_name(input_path.stem + "_kg.json")
    )

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    seen_ids: set[str] = set()
    items = []
    filtered_count = 0
    enrich_hf = args.enrich_hf
    enriched_count = 0

    for d in data.get("datasets", []):
        sid = d.get("dataset_id")
        if sid and sid in seen_ids:
            continue
        if sid:
            seen_ids.add(sid)
        title = d.get("title", "")
        if _is_non_dataset(title):
            filtered_count += 1
            continue

        if enrich_hf:
            links = [l for l in (d.get("links") or []) if l]
            hf_id = extract_hf_id(links)
            if hf_id:
                desc = d.get("description", "")
                if not desc or desc.rstrip().endswith("…") or desc.rstrip().endswith("..."):
                    full = fetch_hf_description(hf_id)
                    if full:
                        d = dict(d, description=full)
                        enriched_count += 1
            time.sleep(0.3)

        items.append(map_item(d))

    output = {
        "theme_qid": args.theme_qid,
        "items": items,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    total = len(items)
    has_license = sum(1 for i in items if i["claims"].get("P163"))
    has_unresolved = sum(1 for i in items if i["unresolved"]["licenses"])
    print(f"Written {total} items to {output_path}")
    if filtered_count:
        print(f"  filtered (non-dataset) : {filtered_count}")
    if enrich_hf:
        print(f"  HF description enriched: {enriched_count}")
    print(f"  with description   : {sum(1 for i in items if i['claims'].get('P1459'))}")
    print(f"  with DOI           : {sum(1 for i in items if i['claims'].get('P27'))}")
    print(f"  with license (QID) : {has_license}")
    print(f"  unresolved license : {has_unresolved}")
    print(f"  no license         : {total - has_license - has_unresolved}")
    if has_unresolved:
        unresolved = {l for i in items for l in i["unresolved"]["licenses"]}
        print(f"  unresolved strings : {sorted(unresolved)}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: import_to_mardi
# ---------------------------------------------------------------------------

def _item_snak(prop: str, qid: str) -> dict:
    return {
        "mainsnak": {
            "snaktype": "value",
            "property": prop,
            "datavalue": {
                "value": {"entity-type": "item", "id": qid},
                "type": "wikibase-entityid",
            },
        },
        "type": "statement",
        "rank": "normal",
    }


def _string_snak(prop: str, value: str) -> dict:
    return {
        "mainsnak": {
            "snaktype": "value",
            "property": prop,
            "datavalue": {"value": value, "type": "string"},
        },
        "type": "statement",
        "rank": "normal",
    }


def _time_snak(prop: str, time_str: str) -> dict:
    return {
        "mainsnak": {
            "snaktype": "value",
            "property": prop,
            "datavalue": {
                "value": {
                    "time": time_str,
                    "timezone": 0,
                    "before": 0,
                    "after": 0,
                    "precision": 11,
                    "calendarmodel": "http://www.wikidata.org/entity/Q1985727",
                },
                "type": "time",
            },
        },
        "type": "statement",
        "rank": "normal",
    }


def _build_wikibase_data(item: dict, theme_qid: str) -> dict:
    """Convert an item dict to the wbeditentity data structure."""
    claims: dict[str, list] = {}

    all_claims = dict(item["claims"])

    for prop, value in all_claims.items():
        values = value if isinstance(value, list) else [value]
        snaks = []
        for v in values:
            if prop in _ITEM_PROPS:
                snaks.append(_item_snak(prop, v))
            elif prop in _TIME_PROPS:
                snaks.append(_time_snak(prop, v))
            else:
                snaks.append(_string_snak(prop, v))
        claims[prop] = snaks

    return {
        "labels": {"en": {"language": "en", "value": item["label"]}},
        "claims": claims,
    }


def _mw_login(session, api_url: str, username: str, password: str) -> str:
    """Log in and return a CSRF token."""
    r = session.get(api_url, params={
        "action": "query", "meta": "tokens", "type": "login", "format": "json",
    })
    r.raise_for_status()
    login_token = r.json()["query"]["tokens"]["logintoken"]

    r = session.post(api_url, data={
        "action": "login",
        "lgname": username,
        "lgpassword": password,
        "lgtoken": login_token,
        "format": "json",
    })
    r.raise_for_status()
    result = r.json()
    if result.get("login", {}).get("result") != "Success":
        raise RuntimeError(f"Login failed: {result}")

    r = session.get(api_url, params={"action": "query", "meta": "tokens", "format": "json"})
    r.raise_for_status()
    return r.json()["query"]["tokens"]["csrftoken"]


def _create_item(session, api_url: str, csrf_token: str, wikibase_data: dict) -> str:
    """Create a new Wikibase item; return its QID."""
    r = session.post(api_url, data={
        "action": "wbeditentity",
        "new": "item",
        "data": json.dumps(wikibase_data, ensure_ascii=False),
        "token": csrf_token,
        "format": "json",
    })
    r.raise_for_status()
    result = r.json()
    if "error" in result:
        raise RuntimeError(f"wbeditentity error: {result['error']}")
    return result["entity"]["id"]


def _add_part_to_theme(session, api_url: str, csrf_token: str, theme_qid: str, dataset_qid: str) -> None:
    """Add a P265 (has part) claim to the theme item pointing to the dataset."""
    r = session.post(api_url, data={
        "action": "wbcreateclaim",
        "entity": theme_qid,
        "snaktype": "value",
        "property": "P265",
        "value": json.dumps({"entity-type": "item", "id": dataset_qid}),
        "token": csrf_token,
        "format": "json",
    })
    r.raise_for_status()
    result = r.json()
    if "error" in result:
        raise RuntimeError(f"wbcreateclaim P265 error: {result['error']}")


def _set_theme_last_update(session, api_url: str, csrf_token: str, theme_qid: str) -> None:
    """Set P170 (last update) on the theme item to today's date.

    Updates the existing claim if one is present; creates a new one otherwise.
    """
    today = datetime.utcnow().strftime("+%Y-%m-%dT00:00:00Z")
    time_value = json.dumps({
        "time": today,
        "timezone": 0,
        "before": 0,
        "after": 0,
        "precision": 11,
        "calendarmodel": "http://www.wikidata.org/entity/Q1985727",
    })

    r = session.get(api_url, params={
        "action": "wbgetentities", "ids": theme_qid, "props": "claims", "format": "json",
    })
    r.raise_for_status()
    existing = r.json()["entities"][theme_qid].get("claims", {}).get("P170", [])

    if existing:
        guid = existing[0]["id"]
        r = session.post(api_url, data={
            "action": "wbsetclaimvalue",
            "claim": guid,
            "snaktype": "value",
            "value": time_value,
            "token": csrf_token,
            "format": "json",
        })
    else:
        r = session.post(api_url, data={
            "action": "wbcreateclaim",
            "entity": theme_qid,
            "snaktype": "value",
            "property": "P170",
            "value": time_value,
            "token": csrf_token,
            "format": "json",
        })
    r.raise_for_status()
    result = r.json()
    if "error" in result:
        raise RuntimeError(f"P170 update error: {result['error']}")


def _find_existing_item(session, api_url: str, label: str) -> str | None:
    """Search the KG for an item whose English label exactly matches `label`.

    Returns the QID of the first exact match, or None if not found.
    ``wbsearchentities`` does prefix/fuzzy matching, so we verify the label
    of each returned hit before accepting it.
    """
    r = session.get(api_url, params={
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "type": "item",
        "limit": 10,
        "format": "json",
    })
    r.raise_for_status()
    for hit in r.json().get("search", []):
        if hit.get("label", "").lower() == label.lower():
            return hit["id"]
    return None


def _load_sidecar(sidecar_path: Path) -> dict:
    if sidecar_path.exists():
        with open(sidecar_path, encoding="utf-8") as f:
            return json.load(f)
    return {"items": {}}


def _save_sidecar(sidecar_path: Path, sidecar: dict) -> None:
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)


def cmd_import_to_mardi(args: argparse.Namespace) -> int:
    try:
        import requests
    except ImportError:
        print("Error: 'requests' is required. Install it with: pip install requests", file=sys.stderr)
        return 1

    kg_path = Path(args.kg_json)
    if not kg_path.exists():
        print(f"Error: {kg_path} not found", file=sys.stderr)
        return 1

    username = args.username or os.environ.get("DOIP_USERNAME", "")
    password = args.password or os.environ.get("DOIP_PASSWORD", "")
    if not username or not password:
        print(
            "Error: credentials required. Pass --username/--password or set "
            "DOIP_USERNAME/DOIP_PASSWORD environment variables.",
            file=sys.stderr,
        )
        return 1

    with open(kg_path, encoding="utf-8") as f:
        kg_data = json.load(f)

    items = kg_data.get("items", [])
    theme_qid = args.theme_qid or kg_data.get("theme_qid", "")
    limit = args.limit if args.limit and args.limit > 0 else len(items)

    sidecar_path = kg_path.with_name(kg_path.stem + "_imported.json")
    sidecar = _load_sidecar(sidecar_path)
    already_imported = sidecar.get("items", {})

    api_url = args.mediawiki_url

    session = requests.Session()
    session.headers.update({"User-Agent": "MaRDI-dataset-import/1.0"})

    print(f"Logging in as {username} ...")
    try:
        csrf_token = _mw_login(session, api_url, username, password)
    except Exception as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    print("Login successful.")

    imported = 0
    skipped = 0
    failed = 0

    for item in items:
        if imported >= limit:
            break

        label = item.get("label", "(no label)")

        # 1. Check local sidecar (fast, no network round-trip).
        if label in already_imported:
            skipped += 1
            continue

        # 2. Check KG for an existing item with the same label.
        try:
            existing_qid = _find_existing_item(session, api_url, label)
        except Exception as exc:
            print(f"  WARN: KG search failed for {label!r}: {exc} — skipping", file=sys.stderr)
            failed += 1
            continue

        if existing_qid:
            print(f"  EXISTS {existing_qid}: {label!r} — recording in sidecar, skipping")
            already_imported[label] = existing_qid
            sidecar["items"] = already_imported
            _save_sidecar(sidecar_path, sidecar)
            skipped += 1
            continue

        wikibase_data = _build_wikibase_data(item, theme_qid)

        try:
            new_qid = _create_item(session, api_url, csrf_token, wikibase_data)
        except Exception as exc:
            print(f"  FAILED [{label!r}]: {exc}", file=sys.stderr)
            failed += 1
            continue

        already_imported[label] = new_qid
        sidecar["items"] = already_imported
        _save_sidecar(sidecar_path, sidecar)

        if theme_qid:
            try:
                _add_part_to_theme(session, api_url, csrf_token, theme_qid, new_qid)
            except Exception as exc:
                print(f"  WARN: could not link {new_qid} to theme {theme_qid}: {exc}", file=sys.stderr)

        print(f"  Created {new_qid}: {label!r}")
        imported += 1

        time.sleep(0.5)

    print(f"\nDone: {imported} imported, {skipped} skipped (exists or already imported), {failed} failed.")
    if imported:
        print(f"Sidecar written to {sidecar_path}")

    if imported and theme_qid:
        try:
            _set_theme_last_update(session, api_url, csrf_token, theme_qid)
            print(f"P170 (last update) set to today on {theme_qid}.")
        except Exception as exc:
            print(f"WARN: could not set P170 on {theme_qid}: {exc}", file=sys.stderr)

    if failed:
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- convert ---
    p_convert = sub.add_parser(
        "convert",
        help="Map a dataset search results JSON to a KG-ready JSON file.",
    )
    p_convert.add_argument("input", help="Path to the dataset search results JSON file")
    p_convert.add_argument(
        "--theme-qid", default="",
        help="QID of the research theme to associate with these datasets (stored in output)",
    )
    p_convert.add_argument(
        "--output", default="",
        help="Output path (default: <input stem>_kg.json alongside input)",
    )
    p_convert.add_argument(
        "--no-enrich-hf", dest="enrich_hf", action="store_false",
        help="Skip fetching full descriptions from HuggingFace READMEs (enrichment is on by default)",
    )
    p_convert.set_defaults(enrich_hf=True)

    # --- import_to_mardi ---
    p_import = sub.add_parser(
        "import_to_mardi",
        help="Import items from a KG-ready JSON file into the MaRDI Knowledge Graph.",
    )
    p_import.add_argument("kg_json", help="Path to the KG-ready JSON file (output of 'convert')")
    p_import.add_argument(
        "--limit", type=int, default=0,
        help="Maximum number of items to import (0 = all, useful for testing)",
    )
    p_import.add_argument(
        "--theme-qid", default="",
        help="Override the theme QID from the KG JSON. Each created dataset is linked to this theme via P265 (has part) on the theme item.",
    )
    p_import.add_argument(
        "--mediawiki-url", default=_MEDIAWIKI_API_URL,
        help=f"MediaWiki API URL (default: {_MEDIAWIKI_API_URL})",
    )
    p_import.add_argument(
        "--username", default="",
        help="Bot username (or set DOIP_USERNAME env var)",
    )
    p_import.add_argument(
        "--password", default="",
        help="Bot password (or set DOIP_PASSWORD env var)",
    )

    args = parser.parse_args(argv)

    if args.command == "convert":
        return cmd_convert(args)
    if args.command == "import_to_mardi":
        return cmd_import_to_mardi(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
