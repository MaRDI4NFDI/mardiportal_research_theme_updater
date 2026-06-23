#!/usr/bin/env python3
"""Merge two MaRDI Wikibase items.

Usage:
    python merge_items.py <TARGET_QID> <DUPLICATE_QID>

Merges DUPLICATE_QID into TARGET_QID: all statements are moved to TARGET_QID
and DUPLICATE_QID becomes a redirect. The TARGET_QID (first argument) is kept.

After the merge, duplicate claims (same property + same value) on the target
item are removed — this catches cases like repeated author name strings or
author links that both items carried.

Credentials are read from environment variables:
    MEDIAWIKI_API_URL      e.g. https://portal.mardi4nfdi.de/w/api.php
    MEDIAWIKI_BOT_USER
    MEDIAWIKI_BOT_PASSWORD
"""
import json
import os
import sys

import requests


def usage():
    print(__doc__.strip())


def _csrf(s: requests.Session, api: str) -> str:
    r = s.get(api, params={"action": "query", "meta": "tokens", "format": "json"}, timeout=30)
    r.raise_for_status()
    return r.json()["query"]["tokens"]["csrftoken"]


def _login(s: requests.Session, api: str, user: str, password: str) -> None:
    r = s.get(api, params={"action": "query", "meta": "tokens", "type": "login", "format": "json"}, timeout=30)
    r.raise_for_status()
    login_token = r.json()["query"]["tokens"]["logintoken"]

    r = s.post(api, data={
        "action": "login", "lgname": user, "lgpassword": password,
        "lgtoken": login_token, "format": "json",
    }, timeout=30)
    r.raise_for_status()
    result = r.json().get("login", {})
    if result.get("result") != "Success":
        print(f"Login failed: {result}", file=sys.stderr)
        sys.exit(1)


def _deduplicate(s: requests.Session, api: str, item_qid: str, csrf_token: str) -> int:
    """Remove claims on item_qid where the same (property, datavalue) appears more than once.

    Returns the number of duplicate claim GUIDs removed.
    """
    r = s.get(api, params={
        "action": "wbgetentities", "ids": item_qid,
        "props": "claims", "format": "json",
    }, timeout=30)
    r.raise_for_status()
    claims = r.json()["entities"][item_qid].get("claims", {})

    to_remove: list[str] = []
    for claim_list in claims.values():
        seen: dict[str, str] = {}  # serialised datavalue -> first guid
        for claim in claim_list:
            datavalue = claim.get("mainsnak", {}).get("datavalue", {})
            key = json.dumps(datavalue, sort_keys=True)
            guid = claim["id"]
            if key in seen:
                to_remove.append(guid)
            else:
                seen[key] = guid

    if not to_remove:
        return 0

    removed = 0
    for i in range(0, len(to_remove), 50):
        batch = to_remove[i : i + 50]
        r = s.post(api, data={
            "action": "wbremoveclaims",
            "claim": "|".join(batch),
            "token": csrf_token,
            "format": "json",
            "bot": "1",
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        if data.get("success"):
            removed += len(batch)
        else:
            print(f"Warning: wbremoveclaims failed for batch: {data.get('error')}", file=sys.stderr)

    return removed


def merge(api: str, user: str, password: str, target: str, duplicate: str) -> None:
    s = requests.Session()
    _login(s, api, user, password)

    token = _csrf(s, api)

    r = s.post(api, data={
        "action": "wbmergeitems",
        "fromid": duplicate,
        "toid": target,
        "ignoreconflicts": "description|sitelink",
        "summary": f"Merge duplicate {duplicate} into {target}",
        "token": token,
        "format": "json",
        "bot": "1",
    }, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        print(f"Merge failed: {data['error']['info']}", file=sys.stderr)
        sys.exit(1)

    print(f"Merged: {duplicate} → {target} (redirected).")

    token = _csrf(s, api)
    removed = _deduplicate(s, api, target, token)
    if removed:
        print(f"Removed {removed} duplicate claim(s) from {target}.")
    else:
        print("No duplicate claims found.")


def main():
    args = sys.argv[1:]
    if len(args) != 2:
        usage()
        sys.exit(0 if not args else 1)

    target, duplicate = args

    api = os.environ.get("MEDIAWIKI_API_URL", "").strip()
    user = os.environ.get("MEDIAWIKI_BOT_USER", "").strip()
    password = os.environ.get("MEDIAWIKI_BOT_PASSWORD", "").strip()

    missing = [name for name, val in [
        ("MEDIAWIKI_API_URL", api),
        ("MEDIAWIKI_BOT_USER", user),
        ("MEDIAWIKI_BOT_PASSWORD", password),
    ] if not val]
    if missing:
        print(f"Missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    merge(api, user, password, target, duplicate)


if __name__ == "__main__":
    main()
