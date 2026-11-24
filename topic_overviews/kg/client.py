"""Write papers (canonical) and topic membership into the MaRDI Wikibase.

Membership lives on the TOPIC item: ``topic --has part(s) (P265)--> paper``.
Paper items stay topic-agnostic — they carry no statement about belonging to a
topic, so the canonical paper entity is never polluted by this feature.
"""
from __future__ import annotations

import json
import logging
import requests

from ..http_utils import LookupUnavailableError, http_get, http_post

from wikibaseintegrator.datatypes import ExternalID as WBExternalID
from wikibaseintegrator.datatypes import Item as WBItem
from wikibaseintegrator.models import Qualifiers

from ..harvest.arxiv_oai import PaperRecord
from . import model as M
from .names import normalize_author_name

log = logging.getLogger(__name__)


def to_wbi_time(date: str) -> str:
    """Wikibase time string for ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD``.

    A missing month or day is written as 00, which WikibaseIntegrator stores
    with year or month precision instead of a made-up day.
    """
    year, month, day = (date.split("-") + ["00", "00"])[:3]
    return f"+{year}-{month}-{day}T00:00:00Z"


def _api_error(result: dict) -> str:
    """Readable text for a MediaWiki API error response."""
    error = result.get("error") or {}
    return error.get("info") or error.get("code") or str(error)


def normalize_doi(doi: str | None) -> str:
    """Return ``doi`` in the portal's canonical form (uppercase), or "".

    DOIs are case-insensitive and the portal stores them uppercased; writing
    another case variant would add a second P27 claim for the same DOI.
    """
    return (doi or "").strip().upper()


def _license_item(license_url: str) -> str:
    """License item for a license URL; "" (and a warning) if the URL is not mapped.

    P163 holds license items, not URLs. An unmapped license is left out rather
    than written in the wrong form; the warning shows which URLs the table lacks.
    """
    qid = M.license_qid(license_url)
    if not qid:
        log.warning("No license item for %r — P163 not written (extend model.LICENSE_QIDS)", license_url)
    return qid


def _description(record) -> str:
    """Description for a new item, as the other importers write it ("" if none)."""
    if getattr(record, "arxiv_id", ""):
        return "scientific article from arXiv"
    de_number = getattr(record, "zbmath_de_number", "")
    if de_number:
        return f"scientific article; zbMATH DE number {de_number}"
    return ""


def _is_description_clash(exc: Exception) -> bool:
    text = str(exc).lower()
    return "already has label" in text and "same description" in text


def _distinct_description(record, description: str) -> str:
    identifier = getattr(record, "arxiv_id", "") or getattr(record, "zbmath_id", "")
    return f"{description} ({identifier})" if identifier else description


def paper_doi(record) -> str:
    """DOI that belongs on the item we write for ``record``, in canonical form.

    An arXiv preprint is its own item with the arXiv DOI
    (``10.48550/ARXIV.<id>``), never the publisher DOI of the article published
    from it: that DOI identifies the published item, and importers that look
    items up by DOI would otherwise take the preprint for the publication.
    """
    if getattr(record, "arxiv_id", ""):
        return normalize_doi(f"10.48550/arXiv.{record.arxiv_id}")
    return normalize_doi(record.doi)


class KGClient:
    def __init__(
        self,
        mc,
        api_url: str = "",
        bot_user: str = "",
        bot_password: str = "",
        sparql_endpoint: str = "",
    ):
        self.mc = mc
        self._api_url = api_url
        self._bot_user = bot_user
        self._bot_password = bot_password
        self._sparql_endpoint = sparql_endpoint or "https://qlever.portal.mardi4nfdi.de/sparql"
        self._session: requests.Session | None = None
        # Set by import_paper: True when a lookup failed during the import (e.g. an
        # author search hit a portal outage), so the item is not yet complete.
        self.last_import_incomplete = False
        # Filled by prefetch_known: positive lookup results only, so a paper that
        # is not in it is still looked up live (the SPARQL index lags behind writes).
        self._lookup_cache: dict[tuple[str, str, tuple[str, ...]], str] = {}
        self._tldr_known: set[str] = set()

    def _get_session(self) -> requests.Session:
        if self._session is None:
            s = requests.Session()
            lt = http_get(
                s,
                self._api_url,
                params={"action": "query", "meta": "tokens", "type": "login", "format": "json"},
                timeout=30,
            ).json()["query"]["tokens"]["logintoken"]
            http_post(
                s,
                self._api_url,
                data={"action": "login", "lgname": self._bot_user, "lgpassword": self._bot_password,
                      "lgtoken": lt, "format": "json"},
                timeout=30,
            )
            self._session = s
        return self._session

    def _csrf(self) -> str:
        return http_get(
            self._get_session(),
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

    @staticmethod
    def _lookup_specs(record: PaperRecord) -> list[tuple[str, str, tuple[str, ...]]]:
        """(property, value, excluded properties) lookups that identify ``record``, in order.

        arXiv items and published (zbMATH) items are separate on the portal and
        must stay separate. An arXiv record is matched only by its arXiv ID or
        its arXiv DOI and ignores items with P225/P1451; every other record
        ignores items with P21 on the shared lookups (publisher DOI, OpenAlex ID).
        Empty values are left out.
        """
        if record.arxiv_id:
            # A preprint is matched only on identifiers that belong to the
            # preprint: its arXiv ID and its arXiv DOI. The publisher DOI and
            # the OpenAlex ID (OpenAlex often describes the published article)
            # identify the published item, so they are never used here.
            own_ids = [(M.P_ARXIV_ID, record.arxiv_id)]
            exclude = (M.P_ZBMATH_ID, M.P_ZBMATH_DE_NUMBER)
            shared_ids = [(M.P_DOI, paper_doi(record))]
            own_exclude: tuple[str, ...] = ()
        else:
            own_ids = [
                (M.P_ZBMATH_ID, getattr(record, "zbmath_id", "")),
                (M.P_ZBMATH_DE_NUMBER, getattr(record, "zbmath_de_number", "")),
            ]
            exclude = (M.P_ARXIV_ID,)
            shared_ids = [
                (M.P_DOI, paper_doi(record)),
                (M.P_OPENALEX_ID, getattr(record, "openalex_id", "")),
            ]
            own_exclude = ()
        return (
            [(p, v, own_exclude) for p, v in own_ids if v]
            + [(p, v, exclude) for p, v in shared_ids if v]
        )

    def prefetch_known(self, records, batch_size: int = 100) -> int:
        """Look up many records at once; later lookups for them are answered from memory.

        One SPARQL query per property and batch of ``batch_size`` identifiers
        replaces the per-paper search and item load for papers the KG already
        has. Only found items are remembered: a paper that is missing here is
        still looked up live, because the SPARQL index can lag behind recent
        writes. Any failure leaves the cache as it is and the run unchanged.
        Returns the number of identifiers found.
        """
        groups: dict[tuple[str, tuple[str, ...]], set[str]] = {}
        for record in records:
            for prop, value, exclude in self._lookup_specs(record):
                key = (prop, value, exclude)
                if key in self._lookup_cache:
                    continue
                # The value goes into a SPARQL string literal; real identifiers
                # never contain these characters.
                if any(c in value for c in '"\\\n\r\t '):
                    continue
                groups.setdefault((prop, exclude), set()).add(value)

        found = 0
        for (prop, exclude), values in groups.items():
            ordered = sorted(values)
            for i in range(0, len(ordered), batch_size):
                chunk = ordered[i:i + batch_size]
                try:
                    found += self._prefetch_chunk(prop, exclude, chunk)
                except Exception as exc:
                    log.warning(
                        "Batch lookup of %d %s value(s) failed (%s) — looking them up one by one",
                        len(chunk), prop, exc,
                    )
        return found

    def _prefetch_chunk(self, prop: str, exclude: tuple[str, ...], values: list[str]) -> int:
        exclusions = "".join(
            f"\n  FILTER NOT EXISTS {{ ?item wdt:{p} [] }}" for p in exclude
        )
        listed = " ".join(f'"{v}"' for v in values)
        query = f"""
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT ?id ?item ?tldr WHERE {{
  VALUES ?id {{ {listed} }}
  ?item wdt:{prop} ?id .{exclusions}
  OPTIONAL {{ ?item wdt:{M.P_TLDR} ?tldr }}
}}
"""
        resp = http_get(
            requests,
            self._sparql_endpoint,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=60,
        )
        found = 0
        for row in resp.json().get("results", {}).get("bindings", []):
            value = row["id"]["value"]
            qid = row["item"]["value"].rstrip("/").rsplit("/", 1)[-1]
            if (prop, value, exclude) not in self._lookup_cache:
                self._lookup_cache[(prop, value, exclude)] = qid
                found += 1
            if "tldr" in row:
                self._tldr_known.add(qid)
        return found

    def find_existing_paper(self, record: PaperRecord) -> str | None:
        """Return QID of an existing paper item matching any known identifier, or None.

        Tries the record's own identifiers first (arXiv ID, or zbMATH ID → DE
        Number), then DOI → OpenAlex ID, and returns on the first hit, so the
        same paper is never imported twice regardless of which source provided it.

        arXiv items and published (zbMATH) items are separate on the portal and
        must stay separate. An arXiv record is matched only by its arXiv ID or
        its arXiv DOI and ignores items with P225/P1451; every other record
        ignores items with P21 on the shared lookups (publisher DOI, OpenAlex ID).

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
        for prop, value, exclude in self._lookup_specs(record):
            qid = self._sparql_find_by_value(prop, value, exclude_props=exclude)
            if qid:
                return qid

        doi = record.doi or ""
        if doi.lower().startswith("10.5281/zenodo.") and record.title:
            qid = self._sparql_find_by_label(record.title)
            if qid:
                log.info(
                    "Zenodo label fallback matched %r → %s (DOI %s)",
                    record.title, qid, doi,
                )
                return qid

        return None

    def _sparql_find_by_value(
        self, prop: str, value: str, exclude_props: tuple[str, ...] = (),
    ) -> str | None:
        """Return the first QID whose ``prop`` claim matches ``value``, or None.

        Items that have any claim for one of ``exclude_props`` are skipped.

        Tries CirrusSearch (haswbstatement) first — it is updated in near
        real-time by ElasticSearch and avoids the SPARQL index lag.
        Falls back to a direct SPARQL query if the search returns nothing.

        Identifier values (DOI, arXiv ID, etc.) never contain quotes, so plain
        string interpolation into the SPARQL query is safe here.
        """
        cached = self._lookup_cache.get((prop, value, tuple(exclude_props)))
        if cached:
            return cached
        failures: list[Exception] = []
        # --- CirrusSearch (fast, near real-time) ---
        try:
            resp = http_get(
                requests,
                self._api_url,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": " ".join(
                        [f"haswbstatement:{prop}={value}"]
                        + [f"-haswbstatement:{p}" for p in exclude_props]
                    ),
                    "srnamespace": "120",
                    "srlimit": "1",
                    "format": "json",
                },
                timeout=15,
            )
            hits = resp.json().get("query", {}).get("search", [])
            if hits:
                # title is "Item:Q12345" — strip the namespace prefix
                title = hits[0]["title"]
                return title.split(":", 1)[-1]
        except Exception as exc:
            failures.append(exc)
            log.warning("CirrusSearch lookup for %s=%r failed: %s", prop, value, exc)

        # --- SPARQL fallback (may lag by minutes after a write) ---
        exclusions = "".join(
            f"\n  FILTER NOT EXISTS {{ ?item wdt:{p} [] }}" for p in exclude_props
        )
        query = f"""
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
SELECT ?item WHERE {{
  ?item wdt:{prop} "{value}" .{exclusions}
}}
LIMIT 1
"""
        try:
            resp = http_get(
                requests,
                self._sparql_endpoint,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
            )
            bindings = resp.json().get("results", {}).get("bindings", [])
            if bindings:
                uri = bindings[0]["item"]["value"]
                return uri.rstrip("/").rsplit("/", 1)[-1]
        except Exception as exc:
            failures.append(exc)
            log.warning("SPARQL lookup for %s=%r failed: %s", prop, value, exc)

        if len(failures) == 2:
            # Neither lookup could be asked. "Not found" would make the caller
            # create a duplicate item, so the paper is retried on the next run.
            raise LookupUnavailableError(
                f"could not look up {prop}={value!r}: {failures[-1]}"
            ) from failures[-1]
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
            resp = http_get(
                requests,
                self._sparql_endpoint,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
            )
            bindings = resp.json().get("results", {}).get("bindings", [])
            if bindings:
                uri = bindings[0]["item"]["value"]
                return uri.rstrip("/").rsplit("/", 1)[-1]
        except Exception as exc:
            log.warning("Label SPARQL lookup for %r failed: %s", title, exc)
        return None

    def _resolve_journal_qid(self, journal_title: str) -> str:
        """Return KG QID for the journal whose English label matches *journal_title*, or ''.

        Results are cached in-process so repeated calls for the same journal (common
        when a batch of papers all come from the same journal) only hit SPARQL once.
        """
        if not journal_title:
            return ""
        if not hasattr(self, "_journal_cache"):
            self._journal_cache: dict[str, str] = {}
        if journal_title in self._journal_cache:
            return self._journal_cache[journal_title]
        escaped = journal_title.replace("\\", "\\\\").replace('"', '\\"')
        query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <https://portal.mardi4nfdi.de/prop/direct/>
PREFIX wd: <https://portal.mardi4nfdi.de/entity/>
SELECT ?item WHERE {{
  ?item rdfs:label "{escaped}"@en .
  ?item wdt:{M.P_INSTANCE_OF} wd:{M.Q_SCIENTIFIC_JOURNAL} .
}}
LIMIT 1
"""
        qid = ""
        try:
            resp = http_get(
                requests,
                self._sparql_endpoint,
                params={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
            )
            bindings = resp.json().get("results", {}).get("bindings", [])
            if bindings:
                qid = bindings[0]["item"]["value"].rstrip("/").rsplit("/", 1)[-1]
        except Exception as exc:
            log.warning("Journal SPARQL lookup for %r failed: %s", journal_title, exc)
        self._journal_cache[journal_title] = qid
        if qid:
            log.debug("Journal %r → %s", journal_title, qid)
        else:
            log.debug("Journal %r not found in KG", journal_title)
        return qid

    def paper_has_tldr(self, paper_qid: str) -> bool:
        """Return whether the paper item already has a TL;DR claim."""
        if paper_qid in self._tldr_known:
            return True
        item = self.mc.item.get(entity_id=paper_qid)
        return bool(item.get_value(M.P_TLDR))

    def import_paper(
        self,
        record: PaperRecord,
        tldr: str | None = None,
        keywords: list[str] | None = None,
        tldr_model_qid: str | None = None,
    ) -> str:
        """Upsert the canonical paper item (idempotent by arXiv ID). Returns its QID.

        Writes only paper-intrinsic statements; nothing about topics. ``tldr`` is
        an optional AI summary; ``keywords`` an optional list of topical tags
        (one statement each); ``tldr_model_qid`` an optional QID of the LLM that generated the TL;DR
        (written as a P1642 qualifier on the P1963 claim). All stored on the paper.
        """
        # NOTE: use BARE local MaRDI PIDs/QIDs. A "wdt:"/"wd:" prefix makes
        # mardiclient interpret the id as a *Wikidata* one and remote-map it,
        # which both picks the wrong property and 404s. Our model.py ids are local.
        self.last_import_incomplete = False
        existing_qid = self.find_existing_paper(record)
        description = ""
        if existing_qid:
            item = self.mc.item.get(entity_id=existing_qid)
        else:
            item = self.mc.item.new()
            item.labels.set("en", record.title[:250])
            # Only new items get a description; an existing item keeps its own.
            description = _description(record)
            if description:
                item.descriptions.set("en", description)

        # Collect already-set single-value properties to avoid duplicates when
        # re-processing a paper that was partially imported in an earlier run.
        existing_author_qids = set(item.get_value(M.P_AUTHOR) or [])
        has_date = bool(item.get_value(M.P_PUBLICATION_DATE))
        # Another source's spelling of the title (hyphen, case, punctuation)
        # would otherwise be added as a second P159.
        has_title = bool(item.get_value(M.P_TITLE))
        # Never mix arXiv and zbMATH identifiers on one item (see find_existing_paper).
        is_arxiv_item = bool(item.get_value(M.P_ARXIV_ID))
        is_zbmath_item = bool(
            item.get_value(M.P_ZBMATH_ID) or item.get_value(M.P_ZBMATH_DE_NUMBER)
        )

        item.add_claim(M.P_INSTANCE_OF, value=M.Q_SCHOLARLY_ARTICLE)
        item.add_claim(M.P_PROFILE_TYPE, value=M.Q_PUBLICATION_PROFILE)
        if record.arxiv_id and not is_zbmath_item:
            item.add_claim(M.P_ARXIV_ID, value=record.arxiv_id)
        if getattr(record, "openalex_id", ""):
            item.add_claim(M.P_OPENALEX_ID, value=record.openalex_id)
        if getattr(record, "zbmath_id", "") and not is_arxiv_item:
            item.add_claim(M.P_ZBMATH_ID, value=record.zbmath_id)
        if getattr(record, "zbmath_de_number", "") and not is_arxiv_item:
            item.add_claim(M.P_ZBMATH_DE_NUMBER, value=record.zbmath_de_number)
        doi = paper_doi(record)
        existing_dois = {normalize_doi(d) for d in item.get_value(M.P_DOI) or []}
        if doi and doi not in existing_dois:
            item.add_claim(M.P_DOI, value=doi)
        if not has_title:
            item.add_claim(M.P_TITLE, value=record.title)
        if record.published and not has_date:
            item.add_claim(M.P_PUBLICATION_DATE, value=to_wbi_time(record.published))
        for cat in record.categories:
            item.add_claim(M.P_ARXIV_CLASSIFICATION, value=cat)
        for code in getattr(record, "msc_codes", None) or []:
            item.add_claim(M.P_MSC_ID, value=code)
        zbmath_codes = dict(getattr(record, "zbmath_author_ids", None) or [])
        orcids = dict(getattr(record, "author_orcids", None) or [])
        linked_names: set[str] = set()
        for name in record.authors:
            # A person is stored either as a link to an author item (found by
            # identifier) or, failing that, as a name string "First Last"
            # (zbMATH gives "Last, First") - never both.
            name_string = normalize_author_name(name)
            author_qid = None
            if not self.last_import_incomplete:  # after one outage, stop asking for this paper
                try:
                    author_qid = self._find_author(zbmath_codes.get(name), orcids.get(name))
                except LookupUnavailableError as exc:
                    # An outage is not "unknown person": write neither link nor name
                    # string and keep the item incomplete, so the next run asks again
                    # instead of leaving a name string next to the later link.
                    log.warning("Author %r not resolved, portal unavailable: %s", name, exc)
                    self.last_import_incomplete = True
            if self.last_import_incomplete:
                continue
            if author_qid:
                linked_names.add(name_string)
                if author_qid not in existing_author_qids:
                    qual = Qualifiers()
                    qual.add(WBItem(prop_nr=M.P_GENERATED_BY, value=M.Q_WORKFLOW))
                    item.add_claim(M.P_AUTHOR, value=author_qid, qualifiers=qual)
                    existing_author_qids.add(author_qid)
            else:
                item.add_claim(M.P_AUTHOR_NAME_STRING, value=name_string)
        self._remove_name_strings(item, linked_names)
        if tldr:
            self._add_tldr_claim(item, tldr, tldr_model_qid)
        for kw in keywords or []:
            item.add_claim(M.P_KEYWORDS, value=kw)
        for kw in getattr(record, "zbmath_keywords", None) or []:
            item.add_claim(M.P_ZBMATH_KEYWORDS, value=kw)
        for kw in getattr(record, "openalex_keywords", None) or []:
            item.add_claim(M.P_KEYWORDS, value=kw)
        for concept in getattr(record, "concepts", None) or []:
            display_name, wikidata_qid = concept if isinstance(concept, tuple) else (concept, "")
            if wikidata_qid:
                qual = Qualifiers()
                qual.add(WBExternalID(prop_nr=M.P_WIKIDATA_QID, value=wikidata_qid))
                item.add_claim(M.P_CONCEPTS, value=display_name, qualifiers=qual)
            else:
                item.add_claim(M.P_CONCEPTS, value=display_name)
        oa_status = getattr(record, "oa_status", "") or ""
        if oa_status and oa_status in M.OA_STATUS_MAP:
            item.add_claim(M.P_OA_STATUS, value=M.OA_STATUS_MAP[oa_status])
        license_url = getattr(record, "license_url", "") or ""
        if license_url:
            license_item = _license_item(license_url)
            if license_item:
                item.add_claim(M.P_LICENSE, value=license_item)
        journal_title = getattr(record, "journal_title", "") or ""
        if journal_title:
            journal_qid = self._resolve_journal_qid(journal_title)
            if journal_qid:
                item.add_claim(M.P_PUBLISHED_IN, value=journal_qid)

        try:
            return item.write().id
        except Exception as exc:
            # The portal allows one item per label+description pair. A different
            # paper with the same title and description is not this paper (it was
            # not found above), so make the description unique instead of losing it.
            if description and _is_description_clash(exc):
                unique = _distinct_description(record, description)
                log.info("Description %r already used with this label — using %r", description, unique)
                item.descriptions.set("en", unique)
                return item.write().id
            raise

    @staticmethod
    def _remove_name_strings(item, names: set[str]) -> None:
        """Remove P43 name strings equal to one of *names* (authors now linked as items)."""
        if not names:
            return
        get = getattr(getattr(item, "claims", None), "get", None)
        if not callable(get):
            return
        for claim in get(M.P_AUTHOR_NAME_STRING) or []:
            value = ((claim.mainsnak.datavalue or {}).get("value") or "").strip()
            if value in names:
                claim.remove()

    def _find_author(self, zbmath_author_id: str | None, orcid: str | None) -> str | None:
        """QID of the person item with this zbMATH author ID (P676) or ORCID (P20), or None.

        Only identifiers count. A name says nothing about whether two people are
        the same, so an author without an ID match gets just the author name
        string and no link to a person item.
        """
        for prop, value in ((M.P_ZBMATH_AUTHOR_ID, zbmath_author_id), (M.P_ORCID, orcid)):
            if value:
                qid = self._sparql_find_by_value(prop, value)
                if qid:
                    return qid
        return None

    @staticmethod
    def _add_tldr_claim(item, tldr: str, tldr_model_qid: str | None) -> None:
        if tldr_model_qid:
            qual = Qualifiers()
            qual.add(WBItem(prop_nr=M.P_GENERATED_BY, value=tldr_model_qid))
            item.add_claim(M.P_TLDR, value=tldr, qualifiers=qual)
        else:
            item.add_claim(M.P_TLDR, value=tldr)

    def add_tldr(self, paper_qid: str, tldr: str, tldr_model_qid: str | None = None) -> None:
        """Write the TL;DR (P1963) onto an existing paper item.

        The pipeline writes it as the last step of an import: a paper with a
        TL;DR counts as fully imported (theme links included) and is skipped
        on later runs.
        """
        item = self.mc.item.get(entity_id=paper_qid)
        self._add_tldr_claim(item, tldr, tldr_model_qid)
        item.write()

    def write_missing_identifiers(self, paper_qid: str, record: "PaperRecord") -> None:
        """Write identifier claims that are present on the record but absent from the KG item.

        Called when a paper is skipped (already has P1963) so that newly available
        identifiers (DOI, arXiv ID from locations, zbMATH ID, OpenAlex ID) are
        never permanently lost just because the paper was imported before they existed.
        """
        s = self._get_session()
        claims = http_get(
            s,
            self._api_url,
            params={"action": "wbgetclaims", "entity": paper_qid, "format": "json"},
            timeout=30,
        ).json().get("claims", {})
        # Never mix arXiv and zbMATH identifiers on one item (see find_existing_paper).
        is_arxiv_item = bool(claims.get(M.P_ARXIV_ID))
        is_zbmath_item = bool(claims.get(M.P_ZBMATH_ID) or claims.get(M.P_ZBMATH_DE_NUMBER))
        candidate_ids = [
            (M.P_ARXIV_ID, "" if is_zbmath_item else record.arxiv_id or ""),
            (M.P_DOI, paper_doi(record)),
            (M.P_OPENALEX_ID, getattr(record, "openalex_id", "") or ""),
            (M.P_ZBMATH_ID, "" if is_arxiv_item else getattr(record, "zbmath_id", "") or ""),
        ]
        for prop, value in candidate_ids:
            if not value or claims.get(prop):
                continue  # nothing to write, or already set
            import json as _json
            http_post(
                s,
                self._api_url,
                data={
                    "action": "wbcreateclaim", "entity": paper_qid,
                    "snaktype": "value", "property": prop,
                    "value": _json.dumps(value),
                    "token": self._csrf(), "format": "json", "bot": "1",
                },
                timeout=30,
            )
            log.info("write_missing_identifiers: %s added %s=%s", paper_qid, prop, value)

    def find_published_version(self, zb_record: PaperRecord) -> str | None:
        """Return the QID of the published (zbMATH) item for ``zb_record``, or None.

        Looks up the DE Number first: every existing P1676 target carries P1451,
        while only a small fraction carries P225.
        """
        for prop, value in (
            (M.P_ZBMATH_DE_NUMBER, zb_record.zbmath_de_number),
            (M.P_ZBMATH_ID, zb_record.zbmath_id),
        ):
            if value:
                qid = self._sparql_find_by_value(prop, value, exclude_props=(M.P_ARXIV_ID,))
                if qid:
                    return qid
        return None

    def find_arxiv_item(self, arxiv_id: str) -> str | None:
        """Return the QID of the arXiv item for ``arxiv_id``, or None."""
        if not arxiv_id:
            return None
        return self._sparql_find_by_value(
            M.P_ARXIV_ID, arxiv_id,
            exclude_props=(M.P_ZBMATH_ID, M.P_ZBMATH_DE_NUMBER),
        )

    def link_preprint(self, preprint_qid: str, published_qid: str) -> bool:
        """Add ``preprint_qid --preprint of (P1676)--> published_qid`` if absent.

        Returns True if a claim was written.
        """
        if not preprint_qid or not published_qid or preprint_qid == published_qid:
            return False
        s = self._get_session()
        r = http_get(
            s,
            self._api_url,
            params={"action": "wbgetclaims", "entity": preprint_qid,
                    "property": M.P_PREPRINT_OF, "format": "json"},
            timeout=30,
        )
        for c in r.json().get("claims", {}).get(M.P_PREPRINT_OF, []):
            if (c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id") == published_qid:
                return False
        http_post(
            s,
            self._api_url,
            data={
                "action": "wbcreateclaim", "entity": preprint_qid,
                "snaktype": "value", "property": M.P_PREPRINT_OF,
                "value": json.dumps({"entity-type": "item", "id": published_qid}),
                "token": self._csrf(), "format": "json", "bot": "1",
            },
            timeout=30,
        )
        log.info("Linked %s --P1676--> %s", preprint_qid, published_qid)
        return True

    def add_zbmath_enrichment(
        self,
        paper_qid: str,
        zbmath_id: str,
        zbmath_author_ids: list[tuple[str, str]],
        zbmath_de_number: str = "",
        msc_codes: list[str] | None = None,
        zbmath_keywords: list[str] | None = None,
        journal_title: str = "",
        license_url: str = "",
    ) -> None:
        """Backfill zbMATH data onto an already-imported paper item.

        Writes P225 (zbMATH document ID string), P1451 (zbMATH DE Number),
        P226 (MSC codes), P1450 (zbMATH keywords), and P200 (published in —
        when the journal is found in the KG) if not already set, then for each
        author with a zbMATH Autorenkennung (P676) that resolves to a KG person
        item, adds a P16 claim with a P1642 qualifier.  Uses the claims API
        throughout — safe on items with pre-existing claims.
        """
        s = self._get_session()

        def _write_if_missing(prop: str, value: str, label: str, is_item: bool = False) -> None:
            r = http_get(
                s,
                self._api_url,
                params={"action": "wbgetclaims", "entity": paper_qid,
                        "property": prop, "format": "json"},
                timeout=30,
            )
            if not r.json().get("claims", {}).get(prop):
                http_post(
                    s,
                    self._api_url,
                    data={
                        "action": "wbcreateclaim", "entity": paper_qid,
                        "snaktype": "value", "property": prop,
                        "value": json.dumps(
                            {"entity-type": "item", "id": value} if is_item else value
                        ),
                        "token": self._csrf(), "format": "json", "bot": "1",
                    },
                    timeout=30,
                )
                log.info("Enriched %s: added %s=%s", paper_qid, label, value)

        if zbmath_id:
            _write_if_missing(M.P_ZBMATH_ID, zbmath_id, "P225")
        if zbmath_de_number:
            _write_if_missing(M.P_ZBMATH_DE_NUMBER, zbmath_de_number, "P1451")
        if license_url:
            license_item = _license_item(license_url)
            if license_item:
                _write_if_missing(M.P_LICENSE, license_item, "P163", is_item=True)
        if msc_codes:
            r = http_get(
                s,
                self._api_url,
                params={"action": "wbgetclaims", "entity": paper_qid,
                        "property": M.P_MSC_ID, "format": "json"},
                timeout=30,
            )
            existing_msc = {
                c["mainsnak"]["datavalue"]["value"]
                for c in r.json().get("claims", {}).get(M.P_MSC_ID, [])
                if c.get("mainsnak", {}).get("datavalue")
            }
            for code in msc_codes:
                if code not in existing_msc:
                    http_post(
                        s,
                        self._api_url,
                        data={
                            "action": "wbcreateclaim", "entity": paper_qid,
                            "snaktype": "value", "property": M.P_MSC_ID,
                            "value": json.dumps(code),
                            "token": self._csrf(), "format": "json", "bot": "1",
                        },
                        timeout=30,
                    )
                    log.info("Enriched %s: added P226=%s", paper_qid, code)

        # P1450 zbMATH keywords
        if zbmath_keywords:
            r = http_get(
                s,
                self._api_url,
                params={"action": "wbgetclaims", "entity": paper_qid,
                        "property": M.P_ZBMATH_KEYWORDS, "format": "json"},
                timeout=30,
            )
            existing_kw = {
                c["mainsnak"]["datavalue"]["value"]
                for c in r.json().get("claims", {}).get(M.P_ZBMATH_KEYWORDS, [])
                if c.get("mainsnak", {}).get("datavalue")
            }
            for kw in zbmath_keywords:
                if kw not in existing_kw:
                    http_post(
                        s,
                        self._api_url,
                        data={
                            "action": "wbcreateclaim", "entity": paper_qid,
                            "snaktype": "value", "property": M.P_ZBMATH_KEYWORDS,
                            "value": json.dumps(kw),
                            "token": self._csrf(), "format": "json", "bot": "1",
                        },
                        timeout=30,
                    )
                    log.info("Enriched %s: added P1450=%r", paper_qid, kw)

        # P200 published in (journal item)
        if journal_title:
            r = http_get(
                s,
                self._api_url,
                params={"action": "wbgetclaims", "entity": paper_qid,
                        "property": M.P_PUBLISHED_IN, "format": "json"},
                timeout=30,
            )
            if not r.json().get("claims", {}).get(M.P_PUBLISHED_IN):
                journal_qid = self._resolve_journal_qid(journal_title)
                if journal_qid:
                    http_post(
                        s,
                        self._api_url,
                        data={
                            "action": "wbcreateclaim", "entity": paper_qid,
                            "snaktype": "value", "property": M.P_PUBLISHED_IN,
                            "value": json.dumps({"entity-type": "item", "id": journal_qid}),
                            "token": self._csrf(), "format": "json", "bot": "1",
                        },
                        timeout=30,
                    )
                    log.info("Enriched %s: added P200=%s (%r)", paper_qid, journal_qid, journal_title)

        # Fetch existing P16 claims once to avoid adding duplicate authors.
        existing_p16 = set()
        r_p16 = http_get(
            s,
            self._api_url,
            params={"action": "wbgetclaims", "entity": paper_qid,
                    "property": M.P_AUTHOR, "format": "json"},
            timeout=30,
        )
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
            http_post(
                s,
                self._api_url,
                data={
                    "action": "wbcreateclaim", "entity": paper_qid,
                    "snaktype": "value", "property": M.P_AUTHOR,
                    "value": json.dumps({"entity-type": "item", "id": author_qid}),
                    "token": self._csrf(), "format": "json", "bot": "1",
                },
                timeout=30,
            )
            log.info(
                "Enriched %s: added P16=%s via P676=%s (author %r)",
                paper_qid, author_qid, zbmath_author_id, name,
            )

    def add_crossref_license(self, paper_qid: str, license_url: str) -> None:
        """Write P163 (license item) from a Crossref license URL if not already set."""
        license_item = _license_item(license_url) if license_url else ""
        if not license_item:
            return
        s = self._get_session()
        r = http_get(
            s,
            self._api_url,
            params={"action": "wbgetclaims", "entity": paper_qid,
                    "property": M.P_LICENSE, "format": "json"},
            timeout=30,
        )
        if not r.json().get("claims", {}).get(M.P_LICENSE):
            resp = http_post(
                s,
                self._api_url,
                data={
                    "action": "wbcreateclaim", "entity": paper_qid,
                    "snaktype": "value", "property": M.P_LICENSE,
                    "value": json.dumps({"entity-type": "item", "id": license_item}),
                    "token": self._csrf(), "format": "json", "bot": "1",
                },
                timeout=30,
            )
            result = resp.json()
            if "error" in result:  # the API reports rejected edits in a 200 response
                log.warning(
                    "add_crossref_license: P163 %s → %s rejected: %s",
                    paper_qid, license_item, _api_error(result),
                )
                return
            log.info("Added P163 (license) to %s: %s (%s)", paper_qid, license_item, license_url)

    def link_citations(
        self,
        paper_qid: str,
        cited_qids: list[str],
        reference_qid: str = "",
    ) -> int:
        """Write P223 (cites work) claims from paper_qid to each cited QID, idempotently.

        Reads existing P223 claims first to avoid duplicates on re-runs.
        If ``reference_qid`` is given, a ``P98=<reference_qid>`` ("stated in") reference
        block is added to each newly written claim.

        The API reports rejected edits in the JSON body of a 200 response; those
        are logged and not counted. Returns the number of claims written.
        """
        if not cited_qids:
            return 0
        s = self._get_session()
        r = http_get(
            s,
            self._api_url,
            params={"action": "wbgetclaims", "entity": paper_qid,
                    "property": M.P_CITES_WORK, "format": "json"},
            timeout=30,
        )
        existing: set[str] = set()
        for c in r.json().get("claims", {}).get(M.P_CITES_WORK, []):
            qid = (c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id")
            if qid:
                existing.add(qid)
        reference_snaks = json.dumps({
            M.P_STATED_IN: [{
                "snaktype": "value",
                "property": M.P_STATED_IN,
                "datavalue": {"type": "wikibase-entityid",
                              "value": {"entity-type": "item", "id": reference_qid}},
            }]
        }) if reference_qid else None
        added = 0
        for cited_qid in cited_qids:
            if cited_qid in existing:
                continue
            existing.add(cited_qid)
            resp = http_post(
                s,
                self._api_url,
                data={
                    "action": "wbcreateclaim", "entity": paper_qid,
                    "snaktype": "value", "property": M.P_CITES_WORK,
                    "value": json.dumps({"entity-type": "item", "id": cited_qid}),
                    "token": self._csrf(), "format": "json", "bot": "1",
                },
                timeout=30,
            )
            result = resp.json()
            if "error" in result:
                log.warning(
                    "link_citations: P223 %s → %s rejected: %s",
                    paper_qid, cited_qid, _api_error(result),
                )
                continue
            added += 1
            if reference_snaks:
                claim_guid = (result.get("claim") or {}).get("id")
                if claim_guid:
                    ref_result = http_post(
                        s,
                        self._api_url,
                        data={
                            "action": "wbsetreference", "statement": claim_guid,
                            "snaks": reference_snaks,
                            "token": self._csrf(), "format": "json", "bot": "1",
                        },
                        timeout=30,
                    ).json()
                    if "error" in ref_result:
                        log.warning(
                            "link_citations: reference %s on %s rejected: %s",
                            reference_qid, claim_guid, _api_error(ref_result),
                        )
        if added:
            log.info("link_citations: %s → %d new P223 claim(s)%s", paper_qid, added,
                     f" [stated in {reference_qid}]" if reference_qid else "")
        return added

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

        Fetches all P265 claims on the theme and the linked items' publication
        dates (P28), then removes claims via wbremoveclaims:

        1. Duplicate links first: several claims pointing to the same item, or
           to an item and a redirect to it (merged duplicates). The claim on
           the item's current QID is kept. Duplicates do not count as papers.
        2. Then the oldest *dated* items until the theme is at the limit.
           Undated entries (curated datasets, items without P28) are never
           removed — their age is unknown — even if the theme stays above it.

        Returns the number of claims removed.
        """
        if max_papers <= 0 or not self._api_url:
            return 0
        s = self._get_session()
        resp = http_get(
            s,
            self._api_url,
            params={"action": "wbgetentities", "ids": topic_qid, "props": "claims", "format": "json"},
            timeout=30,
        )
        p265 = resp.json()["entities"][topic_qid].get("claims", {}).get(M.P_HAS_PART, [])
        if len(p265) <= max_papers:
            return 0

        # (linked QID, claim GUID) in claim order
        links: list[tuple[str, str]] = []
        for c in p265:
            qid = (c.get("mainsnak", {}).get("datavalue", {}).get("value") or {}).get("id")
            if qid:
                links.append((qid, c["id"]))

        # Batch-fetch publication dates. The API follows redirects: the entity
        # is returned under the requested QID with "id" set to the target.
        linked_qids = list(dict.fromkeys(q for q, _ in links))
        target: dict[str, str] = {}
        dates: dict[str, str] = {}
        for i in range(0, len(linked_qids), 50):
            batch = linked_qids[i : i + 50]
            r = http_get(
                s,
                self._api_url,
                params={"action": "wbgetentities", "ids": "|".join(batch), "props": "claims", "format": "json"},
                timeout=60,
            )
            for pqid, entity in r.json()["entities"].items():
                tqid = entity.get("id") or pqid
                target[pqid] = tqid
                p28 = entity.get("claims", {}).get(M.P_PUBLICATION_DATE, [])
                if p28:
                    dates[tqid] = p28[0]["mainsnak"]["datavalue"]["value"]["time"]

        # One kept claim per item; prefer the claim on the item's current QID.
        kept: dict[str, str] = {}      # target QID -> GUID of the kept claim
        duplicates: list[str] = []
        for qid, guid in sorted(links, key=lambda l: target.get(l[0], l[0]) != l[0]):
            tqid = target.get(qid, qid)
            if tqid in kept:
                duplicates.append(guid)
            else:
                kept[tqid] = guid

        excess = max(0, len(kept) - max_papers)
        dated = sorted((q for q in kept if q in dates), key=lambda q: dates[q])
        oldest = dated[:excess]
        if excess > len(oldest):
            log.warning(
                "Theme %s: %d undated entr(y/ies) are never removed — theme stays at %d "
                "entries, above the limit of %d",
                topic_qid, len(kept) - len(dated), len(kept) - len(oldest), max_papers,
            )
        guids = duplicates + [kept[q] for q in oldest]
        if not guids:
            return 0
        if duplicates:
            log.info("Theme %s: removing %d duplicate link(s)", topic_qid, len(duplicates))
        r = http_post(
            s,
            self._api_url,
            data={"action": "wbremoveclaims", "claim": "|".join(guids),
                  "token": self._csrf(), "format": "json", "bot": "1"},
            timeout=60,
        )
        if r.json().get("success"):
            log.info(
                "Theme %s: unlinked %d oldest paper(s) and %d duplicate link(s) "
                "to enforce limit of %d",
                topic_qid, len(oldest), len(duplicates), max_papers,
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
    return KGClient(
        mc,
        api_url=config.mediawiki_api_url,
        bot_user=config.mediawiki_bot_user,
        bot_password=config.mediawiki_bot_password,
        sparql_endpoint=config.sparql_endpoint_url,
    )
