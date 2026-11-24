"""Resolve author name strings to MaRDI KG item QIDs.

Strategy (in order):
1. Search the MaRDI KG by label (wbsearchentities). Only hits that are human
   items (instance of human) whose label equals the name count; wbsearchentities
   alone also returns prefix and alias matches of any kind of item.
   - Single candidate → return it directly.
   - Multiple candidates → disambiguate via ORCID (step 2).
2. Fetch the author's ORCID from OpenAlex by name search.
   Pick the top result by works_count that carries an ORCID.
3. Check each KG candidate for P20 (ORCID); return the one that matches.
4. Fallback → None (caller keeps the bare P43 string).

Results are cached per name for the lifetime of the resolver instance.
"""
from __future__ import annotations

import logging

import requests

from ..http_utils import LookupUnavailableError, http_get
from . import model as M

log = logging.getLogger(__name__)

_OPENALEX_USER_AGENT = "mardi-topic-overviews/1.0 (tofconrad@googlemail.com)"
_P_ORCID = "P20"


def normalize_author_name(name: str) -> str:
    """Convert 'Last, First [Middle]' → 'First [Middle] Last'; pass through otherwise.

    "First Last" is how the portal stores person labels and author name strings;
    zbMATH delivers "Last, First".
    """
    if "," in name:
        last, _, rest = name.partition(",")
        return f"{rest.strip()} {last.strip()}".strip()
    return name


def _same_name(label: str | None, name: str) -> bool:
    return " ".join((label or "").split()).casefold() == " ".join(name.split()).casefold()


class AuthorResolver:
    def __init__(self, mediawiki_api_url: str, session=None):
        self.api_url = mediawiki_api_url
        self._session = session or requests.Session()
        self._cache: dict[str, str | None] = {}
        self._claims_cache: dict[str, dict] = {}

    def resolve(self, name: str) -> str | None:
        """Return a MaRDI KG QID for *name*, or None if unresolvable.

        Raises LookupUnavailableError when the portal could not be asked; that
        outcome is not cached, so a later call asks again.
        """
        if name not in self._cache:
            self._cache[name] = self._resolve(name)
        return self._cache[name]

    def _resolve(self, name: str) -> str | None:
        normalized = normalize_author_name(name)
        # wbsearchentities also returns prefix and alias matches and items of any
        # kind. Only a human item whose label is exactly the name is a candidate.
        hits = [h for h in self._kg_search(normalized) if _same_name(h.get("label"), normalized)]
        candidates = [h["id"] for h in hits if self._is_human(h["id"])]
        if not candidates:
            log.debug("Author %r: no KG match (exact label, human)", normalized)
            return None
        if len(candidates) == 1:
            log.info("Author %r resolved to %s (exact label, human)", normalized, candidates[0])
            return candidates[0]
        # Ambiguous — try ORCID disambiguation.
        orcid = self._openalex_orcid(normalized)
        if orcid:
            for qid in candidates:
                if self._kg_orcid(qid) == orcid:
                    log.info(
                        "Author %r resolved to %s via ORCID %s", normalized, qid, orcid
                    )
                    return qid
        log.debug(
            "Author %r: %d KG candidates but could not disambiguate (ORCID=%s)",
            normalized, len(candidates), orcid,
        )
        return None

    def _kg_claims(self, qid: str) -> dict:
        """Claims of a KG item (cached). Raises LookupUnavailableError if the portal fails."""
        if qid not in self._claims_cache:
            claims: dict = {}
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
            except Exception as exc:
                raise LookupUnavailableError(f"claims lookup for {qid} failed: {exc}") from exc
            self._claims_cache[qid] = claims
        return self._claims_cache[qid]

    def _is_human(self, qid: str) -> bool:
        for claim in self._kg_claims(qid).get(M.P_INSTANCE_OF, []):
            value = claim.get("mainsnak", {}).get("datavalue", {}).get("value") or {}
            if value.get("id") == M.Q_HUMAN:
                return True
        return False

    def _kg_search(self, name: str) -> list[dict]:
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
            return [
                {"id": h["id"], "label": h.get("label", "")}
                for h in r.json().get("search", [])
            ]
        except Exception as exc:
            # Not "no match": the caller must not treat an outage as an unknown person.
            raise LookupUnavailableError(f"KG search for {name!r} failed: {exc}") from exc

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
        for claim in self._kg_claims(qid).get(_P_ORCID, []):
            dv = claim.get("mainsnak", {}).get("datavalue", {})
            if dv.get("type") == "string" and dv.get("value"):
                return dv["value"]
        return None
