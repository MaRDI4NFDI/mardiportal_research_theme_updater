"""Write papers (canonical) and topic membership into the MaRDI Wikibase.

Membership lives on the TOPIC item: ``topic --has part(s) (P265)--> paper``.
Paper items stay topic-agnostic — they carry no statement about belonging to a
topic, so the canonical paper entity is never polluted by this feature.
"""
from __future__ import annotations

import json
import logging
import requests

from wikibaseintegrator.datatypes import Item as WBItem
from wikibaseintegrator.models import Qualifiers

from ..harvest.arxiv_oai import PaperRecord
from . import model as M
from .author_resolver import AuthorResolver

log = logging.getLogger(__name__)


def to_wbi_time(date: str) -> str:
    return f"+{date}T00:00:00Z"


class KGClient:
    def __init__(
        self,
        mc,
        author_resolver: AuthorResolver | None = None,
        api_url: str = "",
        bot_user: str = "",
        bot_password: str = "",
        sparql_endpoint: str = "",
    ):
        self.mc = mc
        self.author_resolver = author_resolver
        self._api_url = api_url
        self._bot_user = bot_user
        self._bot_password = bot_password
        self._sparql_endpoint = sparql_endpoint or "https://query.portal.mardi4nfdi.de/sparql"
        self._session: requests.Session | None = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            s = requests.Session()
            lt = s.get(
                self._api_url,
                params={"action": "query", "meta": "tokens", "type": "login", "format": "json"},
                timeout=30,
            ).json()["query"]["tokens"]["logintoken"]
            s.post(
                self._api_url,
                data={"action": "login", "lgname": self._bot_user, "lgpassword": self._bot_password,
                      "lgtoken": lt, "format": "json"},
                timeout=30,
            )
            self._session = s
        return self._session

    def _csrf(self) -> str:
        return self._get_session().get(
            self._api_url,
            params={"action": "query", "meta": "tokens", "format": "json"},
            timeout=30,
        ).json()["query"]["tokens"]["csrftoken"]

    def get_paper_qid(self, arxiv_id: str) -> str | None:
        """Return the QID of the canonical paper item for ``arxiv_id`` if it exists."""
        if not arxiv_id:
            return None
        existing = self.mc.search_entity_by_value(M.P_ARXIV_ID, arxiv_id)
        return existing[0] if existing else None

    def find_existing_paper(self, record: PaperRecord) -> str | None:
        """Return QID of an existing paper item matching any known identifier, or None.

        Tries arXiv ID → DOI → OpenAlex ID → zbMATH ID in order and returns on
        the first hit, so the same paper is never imported twice regardless of
        which source provided it.

        For Zenodo DOIs (10.5281/zenodo.*) an additional label-based SPARQL
        lookup is performed when all identifier checks fail.  Zenodo assigns a
        new DOI per version upload, so two version deposits of the same paper
        share no identifier yet have identical titles — the label lookup catches
        this case without risking false positives for other paper types.

        Uses direct SPARQL with explicit MaRDI prefixes rather than
        mardiclient's search_entity_by_value, which silently hits the wrong
        endpoint (wikibaseintegrator defaults to Wikidata when its global config
        is not initialised in the current process).
        """
        checks = [
            (M.P_ARXIV_ID, record.arxiv_id),
            (M.P_DOI, record.doi or ""),
            (M.P_OPENALEX_ID, getattr(record, "openalex_id", "")),
            (M.P_ZBMATH_ID, getattr(record, "zbmath_id", "")),
        ]
        for prop, value in checks:
            if not value:
                continue
            qid = self._sparql_find_by_value(prop, value)
            if qid:
                return qid

        doi = record.doi or ""
        if doi.startswith("10.5281/zenodo.") and record.title:
            qid = self._sparql_find_by_label(record.title)
            if qid:
                log.info(
                    "Zenodo label fallback matched %r → %s (DOI %s)",
                    record.title, qid, doi,
                )
                return qid

        return None

    def _sparql_find_by_value(self, prop: str, value: str) -> str | None:
        """Return the first QID whose ``prop`` claim matches ``value``, or None.

        Tries CirrusSearch (haswbstatement) first — it is updated in near
        real-time by ElasticSearch and avoids the Blazegraph SPARQL lag.
        Falls back to a direct SPARQL query if the search returns nothing.

        Identifier values (DOI, arXiv ID, etc.) never contain quotes, so plain
        string interpolation into the SPARQL query is safe here.
        """
        # --- CirrusSearch (fast, near real-time) ---
        try:
            resp = requests.get(
                self._api_url,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": f"haswbstatement:{prop}={value}",
                    "srnamespace": "120",
                    "srlimit": "1",
                    "format": "json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            hits = resp.json().get("query", {}).get("search", [])
            if hits:
                # title is "Item:Q12345" — strip the namespace prefix
                title = hits[0]["title"]
                return title.split(":", 1)[-1]
        except Exception as exc:
            log.warning("CirrusSearch lookup for %s=%r failed: %s", prop, value, exc)

        # --- SPARQL fallback (may lag by minutes after a write) ---
        query = f"""
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT ?item WHERE {{
  ?item wdt:{prop} "{value}" .
}}
LIMIT 1
"""
        try:
            resp = requests.get(
                self._sparql_endpoint,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
            )
            resp.raise_for_status()
            bindings = resp.json().get("results", {}).get("bindings", [])
            if bindings:
                uri = bindings[0]["item"]["value"]
                return uri.rstrip("/").rsplit("/", 1)[-1]
        except Exception as exc:
            log.warning("SPARQL lookup for %s=%r failed: %s", prop, value, exc)

        return None

    def _sparql_find_by_label(self, title: str) -> str | None:
        """Return the QID of a scholarly article whose English label exactly matches
        *title*, or None.  Only used as a Zenodo version-dedup fallback."""
        # Escape backslashes and double-quotes so the title is safe inside a
        # SPARQL string literal.
        escaped = title.replace("\\", "\\\\").replace('"', '\\"')
        query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
SELECT ?item WHERE {{
  ?item rdfs:label "{escaped}"@en .
  ?item wdt:{M.P_INSTANCE_OF} wd:{M.Q_SCHOLARLY_ARTICLE} .
}}
LIMIT 1
"""
        try:
            resp = requests.get(
                self._sparql_endpoint,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
            )
            resp.raise_for_status()
            bindings = resp.json().get("results", {}).get("bindings", [])
            if bindings:
                uri = bindings[0]["item"]["value"]
                return uri.rstrip("/").rsplit("/", 1)[-1]
        except Exception as exc:
            log.warning("Label SPARQL lookup for %r failed: %s", title, exc)
        return None

    def paper_has_tldr(self, paper_qid: str) -> bool:
        """Return whether the paper item already has a TL;DR claim."""
        item = self.mc.item.get(entity_id=paper_qid)
        return bool(item.get_value(M.P_TLDR))

    def import_paper(
        self,
        record: PaperRecord,
        tldr: str | None = None,
        keywords: list[str] | None = None,
        generated_by: str | None = None,
    ) -> str:
        """Upsert the canonical paper item (idempotent by arXiv ID). Returns its QID.

        Writes only paper-intrinsic statements; nothing about topics. ``tldr`` is
        an optional AI summary; ``keywords`` an optional list of topical tags
        (one statement each); ``generated_by`` an optional QID of the LLM that
        produced the item (sets ``generated by`` P1642). All stored on the paper.
        """
        # NOTE: use BARE local MaRDI PIDs/QIDs. A "wdt:"/"wd:" prefix makes
        # mardiclient interpret the id as a *Wikidata* one and remote-map it,
        # which both picks the wrong property and 404s. Our model.py ids are local.
        existing_qid = self.find_existing_paper(record)
        if existing_qid:
            item = self.mc.item.get(entity_id=existing_qid)
        else:
            item = self.mc.item.new()
            item.labels.set("en", record.title[:250])

        # Collect already-set single-value properties to avoid duplicates when
        # re-processing a paper that was partially imported in an earlier run.
        existing_author_qids = set(item.get_value(M.P_AUTHOR) or [])
        has_date = bool(item.get_value(M.P_PUBLICATION_DATE))

        item.add_claim(M.P_INSTANCE_OF, value=M.Q_SCHOLARLY_ARTICLE)
        item.add_claim(M.P_PROFILE_TYPE, value=M.Q_PUBLICATION_PROFILE)
        if record.arxiv_id:
            item.add_claim(M.P_ARXIV_ID, value=record.arxiv_id)
        if getattr(record, "openalex_id", ""):
            item.add_claim(M.P_OPENALEX_ID, value=record.openalex_id)
        if getattr(record, "zbmath_id", ""):
            item.add_claim(M.P_ZBMATH_ID, value=record.zbmath_id)
        if record.doi:
            item.add_claim(M.P_DOI, value=record.doi)
        item.add_claim(M.P_TITLE, value=record.title)
        if record.published and not has_date:
            item.add_claim(M.P_PUBLICATION_DATE, value=to_wbi_time(record.published))
        for cat in record.categories:
            item.add_claim(M.P_ARXIV_CLASSIFICATION, value=cat)
        for name in record.authors:
            item.add_claim(M.P_AUTHOR_NAME_STRING, value=name)
            author_qid = self.author_resolver.resolve(name) if self.author_resolver else None
            if author_qid and author_qid not in existing_author_qids:
                qual = Qualifiers()
                qual.add(WBItem(prop_nr=M.P_GENERATED_BY, value=M.Q_LLM_AUTHOR_RESOLVER))
                item.add_claim(M.P_AUTHOR, value=author_qid, qualifiers=qual)
                existing_author_qids.add(author_qid)
        if tldr:
            item.add_claim(M.P_TLDR, value=tldr)
        for kw in keywords or []:
            item.add_claim(M.P_KEYWORDS, value=kw)
        if generated_by:
            item.add_claim(M.P_GENERATED_BY, value=generated_by)  # bare local QID

        return item.write().id

    def write_missing_identifiers(self, paper_qid: str, record: "PaperRecord") -> None:
        """Write identifier claims that are present on the record but absent from the KG item.

        Called when a paper is skipped (already has P1963) so that newly available
        identifiers (DOI, arXiv ID from locations, zbMATH ID, OpenAlex ID) are
        never permanently lost just because the paper was imported before they existed.
        """
        s = self._get_session()
        candidate_ids = [
            (M.P_ARXIV_ID, record.arxiv_id or ""),
            (M.P_DOI, record.doi or ""),
            (M.P_OPENALEX_ID, getattr(record, "openalex_id", "") or ""),
            (M.P_ZBMATH_ID, getattr(record, "zbmath_id", "") or ""),
        ]
        for prop, value in candidate_ids:
            if not value:
                continue
            r = s.get(
                self._api_url,
                params={"action": "wbgetclaims", "entity": paper_qid,
                        "property": prop, "format": "json"},
                timeout=30,
            )
            r.raise_for_status()
            if r.json().get("claims", {}).get(prop):
                continue  # already set
            import json as _json
            s.post(
                self._api_url,
                data={
                    "action": "wbcreateclaim", "entity": paper_qid,
                    "snaktype": "value", "property": prop,
                    "value": _json.dumps(value),
                    "token": self._csrf(), "format": "json", "bot": "1",
                },
                timeout=30,
            ).raise_for_status()
            log.info("write_missing_identifiers: %s added %s=%s", paper_qid, prop, value)

    def add_zbmath_enrichment(
        self,
        paper_qid: str,
        zbmath_id: str,
        zbmath_author_ids: list[tuple[str, str]],
    ) -> None:
        """Backfill zbMATH data onto an already-imported paper item.

        Writes P225 (zbMATH document ID) via ``wbcreateclaim`` if not already
        set, then for each author with a zbMATH Autorenkennung (P676) that
        resolves to a KG person item, adds a P16 claim with a P1642 qualifier.
        Uses the claims API throughout — safe on items with pre-existing claims.
        """
        s = self._get_session()

        if zbmath_id:
            r = s.get(
                self._api_url,
                params={
                    "action": "wbgetclaims", "entity": paper_qid,
                    "property": M.P_ZBMATH_ID, "format": "json",
                },
                timeout=30,
            )
            r.raise_for_status()
            if not r.json().get("claims", {}).get(M.P_ZBMATH_ID):
                s.post(
                    self._api_url,
                    data={
                        "action": "wbcreateclaim", "entity": paper_qid,
                        "snaktype": "value", "property": M.P_ZBMATH_ID,
                        "value": json.dumps(zbmath_id),
                        "token": self._csrf(), "format": "json", "bot": "1",
                    },
                    timeout=30,
                ).raise_for_status()
                log.info("Enriched %s: added P225=%s", paper_qid, zbmath_id)

        # Fetch existing P16 claims once to avoid adding duplicate authors.
        existing_p16 = set()
        r_p16 = s.get(
            self._api_url,
            params={"action": "wbgetclaims", "entity": paper_qid,
                    "property": M.P_AUTHOR, "format": "json"},
            timeout=30,
        )
        r_p16.raise_for_status()
        for c in r_p16.json().get("claims", {}).get(M.P_AUTHOR, []):
            qid = (c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id")
            if qid:
                existing_p16.add(qid)

        for name, zbmath_author_id in zbmath_author_ids:
            if not zbmath_author_id:
                continue
            hits = self.mc.search_entity_by_value(M.P_ZBMATH_AUTHOR_ID, zbmath_author_id)
            if not hits:
                log.debug("P676=%s not found in KG (author %r)", zbmath_author_id, name)
                continue
            author_qid = hits[0]
            if author_qid in existing_p16:
                log.debug("P16=%s already set on %s (author %r), skipping", author_qid, paper_qid, name)
                continue
            existing_p16.add(author_qid)
            r = s.post(
                self._api_url,
                data={
                    "action": "wbcreateclaim", "entity": paper_qid,
                    "snaktype": "value", "property": M.P_AUTHOR,
                    "value": json.dumps({"entity-type": "item", "id": author_qid}),
                    "token": self._csrf(), "format": "json", "bot": "1",
                },
                timeout=30,
            )
            r.raise_for_status()
            claim_guid = (r.json().get("claim") or {}).get("id")
            if claim_guid:
                s.post(
                    self._api_url,
                    data={
                        "action": "wbsetqualifier", "claim": claim_guid,
                        "snaktype": "value", "property": M.P_GENERATED_BY,
                        "value": json.dumps(
                            {"entity-type": "item", "id": M.Q_LLM_AUTHOR_RESOLVER}
                        ),
                        "token": self._csrf(), "format": "json", "bot": "1",
                    },
                    timeout=30,
                ).raise_for_status()
            log.info(
                "Enriched %s: added P16=%s via P676=%s (author %r)",
                paper_qid, author_qid, zbmath_author_id, name,
            )

    def link_citations(self, paper_qid: str, cited_qids: list[str]) -> None:
        """Write P223 (cites work) claims from paper_qid to each cited QID, idempotently.

        Reads existing P223 claims first to avoid duplicates on re-runs.
        """
        if not cited_qids:
            return
        s = self._get_session()
        r = s.get(
            self._api_url,
            params={"action": "wbgetclaims", "entity": paper_qid,
                    "property": M.P_CITES_WORK, "format": "json"},
            timeout=30,
        )
        r.raise_for_status()
        existing: set[str] = set()
        for c in r.json().get("claims", {}).get(M.P_CITES_WORK, []):
            qid = (c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id")
            if qid:
                existing.add(qid)
        added = 0
        for cited_qid in cited_qids:
            if cited_qid in existing:
                continue
            existing.add(cited_qid)
            s.post(
                self._api_url,
                data={
                    "action": "wbcreateclaim", "entity": paper_qid,
                    "snaktype": "value", "property": M.P_CITES_WORK,
                    "value": json.dumps({"entity-type": "item", "id": cited_qid}),
                    "token": self._csrf(), "format": "json", "bot": "1",
                },
                timeout=30,
            ).raise_for_status()
            added += 1
        if added:
            log.info("link_citations: %s → %d new P223 claim(s)", paper_qid, added)

    def link_topic(self, topic_qid: str, paper_qid: str) -> None:
        """Add the paper to the topic's ``has part(s)`` (P265) list, idempotently.

        Reads the topic's current membership first and only writes when the paper
        is not already listed, so re-runs never duplicate entries.
        """
        topic_item = self.mc.item.get(entity_id=topic_qid)
        current = topic_item.get_value(M.P_HAS_PART) or []
        if paper_qid in current:
            return
        topic_item.add_claim(M.P_HAS_PART, value=paper_qid)
        topic_item.write()

    def enforce_theme_limit(self, topic_qid: str, max_papers: int) -> int:
        """Unlink the oldest papers from a theme when it exceeds *max_papers*.

        Fetches all P265 claims on the theme, resolves publication dates (P28)
        for each linked paper, sorts oldest-first, and removes via wbremoveclaims
        until the count is at or below the limit. Returns the number removed.
        """
        if max_papers <= 0 or not self._api_url:
            return 0
        s = self._get_session()
        resp = s.get(
            self._api_url,
            params={"action": "wbgetentities", "ids": topic_qid, "props": "claims", "format": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        p265 = resp.json()["entities"][topic_qid].get("claims", {}).get(M.P_HAS_PART, [])
        if len(p265) <= max_papers:
            return 0

        # Map paper QID -> claim GUID
        paper_guids: dict[str, str] = {}
        for c in p265:
            qid = (c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id")
            if qid:
                paper_guids[qid] = c["id"]

        # Batch-fetch publication dates
        paper_qids = list(paper_guids)
        dates: dict[str, str] = {}
        for i in range(0, len(paper_qids), 50):
            batch = paper_qids[i : i + 50]
            r = s.get(
                self._api_url,
                params={"action": "wbgetentities", "ids": "|".join(batch), "props": "claims", "format": "json"},
                timeout=60,
            )
            for pqid, entity in r.json()["entities"].items():
                p28 = entity.get("claims", {}).get(M.P_PUBLICATION_DATE, [])
                dates[pqid] = p28[0]["mainsnak"]["datavalue"]["value"]["time"] if p28 else ""

        # Oldest first
        to_remove = sorted(paper_qids, key=lambda q: dates.get(q, ""))[: len(paper_qids) - max_papers]
        guids = [paper_guids[q] for q in to_remove]
        r = s.post(
            self._api_url,
            data={"action": "wbremoveclaims", "claim": "|".join(guids),
                  "token": self._csrf(), "format": "json", "bot": "1"},
            timeout=60,
        )
        if r.json().get("success"):
            log.info(
                "Theme %s: unlinked %d oldest paper(s) to enforce limit of %d",
                topic_qid, len(guids), max_papers,
            )
            return len(guids)
        log.warning("Theme %s: wbremoveclaims failed: %s", topic_qid, r.json().get("error"))
        return 0

    def get_theme_sitelink(self, theme_qid: str) -> str | None:
        """Return the wiki page title the theme item is connected to (``mardi``
        sitelink), or None if the item has no connected page yet."""
        item = self.mc.item.get(entity_id=theme_qid)
        sitelink = item.sitelinks.get(M.SITE_ID)
        return sitelink.title if sitelink else None

    def set_theme_sitelink(self, theme_qid: str, page_title: str) -> None:
        """Connect the theme item to its wiki page via the ``mardi`` sitelink."""
        item = self.mc.item.get(entity_id=theme_qid)
        item.sitelinks.set(site=M.SITE_ID, title=page_title)
        item.write()


def make_kg_client(config) -> KGClient:
    from mardiclient import MardiClient

    mc = MardiClient(
        user=config.mediawiki_bot_user,
        password=config.mediawiki_bot_password,
        login_with_bot=True,
        mediawiki_api_url=config.mediawiki_api_url,
        sparql_endpoint_url=config.sparql_endpoint_url,
        wikibase_url=config.wikibase_url,
    )
    resolver = AuthorResolver(config.mediawiki_api_url)
    return KGClient(
        mc,
        author_resolver=resolver,
        api_url=config.mediawiki_api_url,
        bot_user=config.mediawiki_bot_user,
        bot_password=config.mediawiki_bot_password,
        sparql_endpoint=config.sparql_endpoint_url,
    )
