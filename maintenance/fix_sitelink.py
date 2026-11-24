#!/usr/bin/env python3
"""Fix a broken sitelink between a MaRDI Wikibase item and its wiki page.

Symptom: the person/theme profile page renders [[Item:|]] instead of the QID
badge, and the publications list shows "No records found".  This happens when
the sitelink is stored in the item JSON but is missing from Wikibase's internal
sitelink index (wb_items_per_site), so mw.wikibase.getEntityIdForCurrentPage()
returns nil on the wiki page.

Fix: remove the sitelink and immediately re-add it.  This forces Wikibase to
write a fresh entry to the sitelink index.

Usage:
    python maintenance/fix_sitelink.py <QID> <PAGE_TITLE>

Example:
    python maintenance/fix_sitelink.py Q354180 "Marcus Weber"
    python maintenance/fix_sitelink.py Q482723 "Tim O. F. Conrad"

Credentials are read from environment variables:
    MEDIAWIKI_API_URL      e.g. https://portal.mardi4nfdi.de/w/api.php
    MEDIAWIKI_BOT_USER
    MEDIAWIKI_BOT_PASSWORD

Manual fix via the web UI (same effect):
    1. Go to the item page, e.g. https://portal.mardi4nfdi.de/wiki/Item:Q354180
    2. Scroll to the Sitelinks section at the bottom.
    3. Click the edit (pencil) icon next to the "mardi" entry.
    4. Clear the title field and save — this removes the sitelink.
    5. Click add (or the pencil again), type the page title back in, and save.
"""
import os
import sys
import time

import requests


def _login(s: requests.Session, api: str, user: str, password: str) -> None:
    r = s.get(api, params={"action": "query", "meta": "tokens",
                            "type": "login", "format": "json"}, timeout=30)
    r.raise_for_status()
    token = r.json()["query"]["tokens"]["logintoken"]
    r = s.post(api, data={"action": "login", "lgname": user,
                           "lgpassword": password, "lgtoken": token,
                           "format": "json"}, timeout=30)
    r.raise_for_status()
    if r.json().get("login", {}).get("result") != "Success":
        print(f"Login failed: {r.json()}", file=sys.stderr)
        sys.exit(1)


def _csrf(s: requests.Session, api: str) -> str:
    r = s.get(api, params={"action": "query", "meta": "tokens",
                            "format": "json"}, timeout=30)
    r.raise_for_status()
    return r.json()["query"]["tokens"]["csrftoken"]


def _sitelink_indexed(api: str, title: str) -> str | None:
    """Return the QID if the sitelink is indexed, else None."""
    r = requests.get(api, params={"action": "wbgetentities", "sites": "mardi",
                                   "titles": title, "props": "info",
                                   "format": "json"}, timeout=30)
    r.raise_for_status()
    entities = r.json()["entities"]
    qid = list(entities.keys())[0]
    return None if qid == "-1" or "missing" in entities[qid] else qid


def fix_sitelink(api: str, user: str, password: str, qid: str, title: str) -> None:
    # Diagnose first
    found = _sitelink_indexed(api, title)
    if found:
        print(f"Sitelink already indexed correctly: {title} → {found}")
        print("Nothing to fix.")
        return

    print(f"Sitelink not indexed for {qid} / '{title}' — fixing…")

    s = requests.Session()
    _login(s, api, user, password)

    # Remove
    token = _csrf(s, api)
    r = s.post(api, data={"action": "wbsetsitelink", "id": qid,
                           "linksite": "mardi", "linktitle": "",
                           "token": token, "format": "json", "bot": "1"},
               timeout=30)
    r.raise_for_status()
    if "error" in r.json():
        print(f"Failed to remove sitelink: {r.json()['error']['info']}", file=sys.stderr)
        sys.exit(1)
    print("  Removed sitelink.")
    time.sleep(1)

    # Re-add
    token = _csrf(s, api)
    r = s.post(api, data={"action": "wbsetsitelink", "id": qid,
                           "linksite": "mardi", "linktitle": title,
                           "token": token, "format": "json", "bot": "1"},
               timeout=30)
    r.raise_for_status()
    if "error" in r.json():
        print(f"Failed to re-add sitelink: {r.json()['error']['info']}", file=sys.stderr)
        sys.exit(1)
    print(f"  Re-added sitelink: {qid} → '{title}'.")
    time.sleep(1)

    # Verify
    found = _sitelink_indexed(api, title)
    if found:
        print(f"Fixed. Reverse lookup confirmed: '{title}' → {found}")
    else:
        print("ERROR: sitelink still not indexed after fix.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = sys.argv[1:]
    if len(args) != 2:
        print(__doc__.strip())
        sys.exit(0 if not args else 1)

    qid, title = args

    api      = os.environ.get("MEDIAWIKI_API_URL", "").strip()
    user     = os.environ.get("MEDIAWIKI_BOT_USER", "").strip()
    password = os.environ.get("MEDIAWIKI_BOT_PASSWORD", "").strip()

    missing = [n for n, v in [("MEDIAWIKI_API_URL", api),
                                ("MEDIAWIKI_BOT_USER", user),
                                ("MEDIAWIKI_BOT_PASSWORD", password)] if not v]
    if missing:
        print(f"Missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    fix_sitelink(api, user, password, qid, title)


if __name__ == "__main__":
    main()
