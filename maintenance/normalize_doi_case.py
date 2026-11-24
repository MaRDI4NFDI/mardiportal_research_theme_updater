#!/usr/bin/env python3
"""Uppercase DOIs (P27) that the Research Theme Updater wrote in lower/mixed case.

The portal stores DOIs uppercased. Until cf97896 the updater wrote them in the
source's case, which left lowercase DOIs on items it created and, on a few
older items, a second P27 claim that differs from the existing one only in case.

Per item (current state read from the wiki API, not from SPARQL, which may lag):
  - a non-uppercase P27 whose uppercase form is already on the item is removed
    (skipped for manual review if that claim carries references or qualifiers);
  - any other non-uppercase P27 is changed in place via wbsetclaimvalue, which
    keeps the claim's references and qualifiers.

Candidates: items >= Q7266000 (created by the updater) with a non-uppercase
DOI, plus any item that has two DOIs differing only in case.

Usage:
    python maintenance/normalize_doi_case.py                      # dry run, all candidates
    python maintenance/normalize_doi_case.py --qids Q1 Q2         # dry run, given items
    python maintenance/normalize_doi_case.py --qids Q1 --apply    # edit given items
    python maintenance/normalize_doi_case.py --apply              # edit all candidates

Writes a CSV of the plan (and of what was done with --apply) to --csv.

Credentials are read from environment variables:
    MEDIAWIKI_API_URL
    MEDIAWIKI_BOT_USER
    MEDIAWIKI_BOT_PASSWORD
    SPARQL_ENDPOINT_URL   (default: QLever)
"""
import argparse
import csv
import json
import logging
import os
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPARQL_ENDPOINT_DEFAULT = "https://qlever.portal.mardi4nfdi.de/sparql"
P_DOI = "P27"
MIN_UPDATER_QID = 7266000
SUMMARY = "Normalize DOI case (MaRDI4NFDI/docker-importer#294)"

_CANDIDATES_QUERY = f"""
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT DISTINCT ?item WHERE {{
  {{
    ?item wdt:{P_DOI} ?d .
    FILTER(?d != UCASE(?d))
    FILTER(xsd:integer(STRAFTER(STR(?item), "/entity/Q")) >= {MIN_UPDATER_QID})
  }} UNION {{
    ?item wdt:{P_DOI} ?a, ?b .
    FILTER(?a != ?b && UCASE(?a) = UCASE(?b))
  }}
}}
"""


def find_candidates(endpoint: str) -> list[str]:
    query = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" + _CANDIDATES_QUERY
    r = requests.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=300,
    )
    r.raise_for_status()
    qids = [b["item"]["value"].rsplit("/", 1)[-1] for b in r.json()["results"]["bindings"]]
    return sorted(set(qids), key=lambda q: int(q[1:]))


def get_doi_claims(s: requests.Session, api: str, qids: list[str]) -> dict[str, list[dict]]:
    """Return {qid: [P27 claim, ...]} read live from the wiki (50 items per request)."""
    out: dict[str, list[dict]] = {}
    for i in range(0, len(qids), 50):
        r = s.get(api, params={
            "action": "wbgetentities", "ids": "|".join(qids[i:i + 50]),
            "props": "claims", "format": "json",
        }, timeout=60)
        r.raise_for_status()
        for qid, ent in r.json().get("entities", {}).items():
            out[qid] = ent.get("claims", {}).get(P_DOI, [])
    return out


def _value(claim: dict) -> str:
    return claim.get("mainsnak", {}).get("datavalue", {}).get("value", "") or ""


def plan_item(qid: str, claims: list[dict]) -> list[dict]:
    """Return the planned actions for one item's P27 claims."""
    values = {_value(c) for c in claims}
    actions = []
    for c in claims:
        v = _value(c)
        if not v or v == v.upper():
            continue
        row = {"qid": qid, "claim_id": c["id"], "old": v, "new": v.upper()}
        if v.upper() in values:
            if c.get("references") or c.get("qualifiers"):
                row["action"] = "skip (duplicate has references/qualifiers)"
            else:
                row["action"] = "remove duplicate"
        else:
            row["action"] = "uppercase"
        actions.append(row)
    return actions


def login(api: str, user: str, password: str) -> requests.Session:
    s = requests.Session()
    lt = s.get(api, params={"action": "query", "meta": "tokens", "type": "login",
                            "format": "json"}, timeout=30).json()["query"]["tokens"]["logintoken"]
    r = s.post(api, data={"action": "login", "lgname": user, "lgpassword": password,
                          "lgtoken": lt, "format": "json"}, timeout=30).json()
    if r.get("login", {}).get("result") != "Success":
        raise RuntimeError(f"Login failed: {r}")
    return s


def csrf(s: requests.Session, api: str) -> str:
    return s.get(api, params={"action": "query", "meta": "tokens", "format": "json"},
                 timeout=30).json()["query"]["tokens"]["csrftoken"]


def apply_action(s: requests.Session, api: str, row: dict) -> str:
    if row["action"] == "uppercase":
        data = {"action": "wbsetclaimvalue", "claim": row["claim_id"], "snaktype": "value",
                "value": json.dumps(row["new"])}
    elif row["action"] == "remove duplicate":
        data = {"action": "wbremoveclaims", "claim": row["claim_id"]}
    else:
        return "skipped"
    data.update({"token": csrf(s, api), "summary": SUMMARY, "bot": "1", "format": "json"})
    r = s.post(api, data=data, timeout=60).json()
    if "error" in r:
        return f"error: {r['error'].get('code')}: {r['error'].get('info')}"
    return "done"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qids", nargs="*", help="only these items (default: all candidates)")
    ap.add_argument("--apply", action="store_true", help="make the edits (default: dry run)")
    ap.add_argument("--csv", default="normalize_doi_case.csv", help="plan/result CSV path")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between edits")
    args = ap.parse_args()

    api = os.environ.get("MEDIAWIKI_API_URL", "").strip()
    endpoint = os.environ.get("SPARQL_ENDPOINT_URL", "").strip() or SPARQL_ENDPOINT_DEFAULT
    if not api:
        log.error("MEDIAWIKI_API_URL is not set")
        return 1

    qids = args.qids or find_candidates(endpoint)
    log.info("%d candidate item(s)", len(qids))

    s = requests.Session()
    if args.apply:
        user = os.environ.get("MEDIAWIKI_BOT_USER", "").strip()
        password = os.environ.get("MEDIAWIKI_BOT_PASSWORD", "").strip()
        if not user or not password:
            log.error("MEDIAWIKI_BOT_USER / MEDIAWIKI_BOT_PASSWORD not set")
            return 1
        s = login(api, user, password)

    plan = []
    for qid, claims in get_doi_claims(s, api, qids).items():
        plan.extend(plan_item(qid, claims))
    log.info("%d action(s) planned on %d item(s)", len(plan), len({r["qid"] for r in plan}))

    for row in plan:
        if args.apply:
            row["result"] = apply_action(s, api, row)
            log.info("%s %s %r -> %r: %s", row["qid"], row["action"], row["old"], row["new"], row["result"])
            time.sleep(args.sleep)
        else:
            row["result"] = "dry run"
            log.info("%s %s %r -> %r", row["qid"], row["action"], row["old"], row["new"])

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "claim_id", "action", "old", "new", "result"])
        w.writeheader()
        w.writerows(plan)
    log.info("Wrote %s", args.csv)
    return 0 if all(not r["result"].startswith("error") for r in plan) else 2


if __name__ == "__main__":
    sys.exit(main())
