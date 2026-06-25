"""Resolve author name strings to MaRDI KG item QIDs.

Strategy (in order):
1. Search the MaRDI KG by label (wbsearchentities).
   - Single hit → return it directly.
   - Multiple hits → disambiguate via ORCID (step 2).
2. Fetch the author's ORCID from OpenAlex by name search.
   Pick the top result by works_count that carries an ORCID.
3. Check each KG candidate for P20 (ORCID); return the one that matches.
4. Fallback → None (caller keeps the bare P43 string).

Results are cached per name for the lifetime of the resolver instance.
"""
from __future__ import annotations

import logging

import requests

from ..http_utils import http_get

log = logging.getLogger(__name__)

_OPENALEX_USER_AGENT = "mardi-topic-overviews/1.0 (tofconrad@googlemail.com)"
_P_ORCID = "P20"


class AuthorResolver:
    def __init__(self, mediawiki_api_url: str, session=None):
        self.api_url = mediawiki_api_url
        self._session = session or requests.Session()
        self._cache: dict[str, str | None] = {}

    def resolve(self, name: str) -> str | None:
        """Return a MaRDI KG QID for *name*, or None if unresolvable."""
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    @staticmethod
    def _normalize(name: str) -> str:
        """Convert 'Last, First [Middle]' → 'First [Middle] Last'; pass through otherwise."""
        if "," in name:
            last, _, rest = name.partition(",")
            return f"{rest.strip()} {last.strip()}"
        return name

    def _resolve(self, name: str) -> str | None:
        normalized = self._normalize(name)
        hits = self._kg_search(normalized)
        if not hits:
            log.debug("Author %r: no KG match", normalized)
            return None
        if len(hits) == 1:
            log.info("Author %r resolved to %s (exact KG match)", normalized, hits[0])
            return hits[0]
        # Ambiguous — try ORCID disambiguation.
        orcid = self._openalex_orcid(normalized)
        if orcid:
            for qid in hits:
                if self._kg_orcid(qid) == orcid:
                    log.info(
                        "Author %r resolved to %s via ORCID %s", normalized, qid, orcid
                    )
                    return qid
        log.debug(
            "Author %r: %d KG hits but could not disambiguate (ORCID=%s)",
            normalized, len(hits), orcid,
        )
        return None

    def _kg_search(self, name: str) -> list[str]:
        try:
            r = http_get(
                self._session,
                self.api_url,
                params={
                    "action": "wbsearchentities",
                    "search": name,
                    "language": "en",
                    "type": "item",
                    "limit": 5,
                    "format": "json",
                },
                timeout=30,
            )
            return [h["id"] for h in r.json().get("search", [])]
        except Exception as exc:
            log.warning("KG search failed for %r: %s", name, exc)
            return []

    def _openalex_orcid(self, name: str) -> str | None:
        try:
            r = http_get(
                requests,
                "https://api.openalex.org/authors",
                params={"search": name, "per-page": 5},
                headers={"User-Agent": _OPENALEX_USER_AGENT},
                timeout=15,
            )
            results = r.json().get("results", [])
            best = max(
                (a for a in results if a.get("orcid")),
                key=lambda a: a.get("works_count", 0),
                default=None,
            )
            if best:
                return best["orcid"].rsplit("/", 1)[-1]
        except Exception as exc:
            log.warning("OpenAlex lookup failed for %r: %s", name, exc)
        return None

    def _kg_orcid(self, qid: str) -> str | None:
        try:
            r = http_get(
                self._session,
                self.api_url,
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "claims",
                    "format": "json",
                },
                timeout=30,
            )
            claims = r.json()["entities"][qid].get("claims", {})
            for claim in claims.get(_P_ORCID, []):
                dv = claim.get("mainsnak", {}).get("datavalue", {})
                if dv.get("type") == "string" and dv.get("value"):
                    return dv["value"]
        except Exception as exc:
            log.warning("ORCID lookup failed for %s: %s", qid, exc)
        return None
