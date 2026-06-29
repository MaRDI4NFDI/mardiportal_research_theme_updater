"""Sequential pipeline steps. Plain functions with injected deps (Prefect-ready)."""
from __future__ import annotations

import dataclasses
import logging
import time

from .config import Config
from .state import State
from .harvest.arxiv_search import search_records
from .harvest.openalex import fetch_openalex_records, lookup_openalex_enrichment
from .harvest.zbmath import fetch_zbmath_records
from .kg import model as M
from .kg.topics import Topic
from .kg.citation_linker import (
    fetch_zbmath_references,
    fetch_openalex_referenced_works,
    fetch_crossref_data,
    resolve_qids_by_zbmath_doc_ids,
    resolve_qids_by_openalex_ids,
    resolve_qids_by_dois,
    fetch_s2_references,
    resolve_s2_references,
)
from .llm.topic_classifier import classify_paper
from .llm.summarizer import summarize_paper
from .llm.keyworder import keywords_paper
from .llm.client import make_llm_client
from .wiki.page_builder import RESEARCH_THEME_STUB
from .arxiv_to_md import fetch_and_convert
from .lakefs_upload import upload_markdown, commit_upload

log = logging.getLogger(__name__)


class PipelineError(Exception):
    """Fatal pipeline error — should terminate the run."""


def default_harvest(config: Config):
    """Default paper source: arXiv keyword search over a recent date window."""
    return search_records(config.arxiv_query, config.since_days)


def _effective_since_days(topic: Topic, config: Config) -> int:
    return topic.since_days if topic.since_days is not None else config.since_days


def _harvest_configs(config: Config, topics: list[Topic]) -> list[Config]:
    """Build one arXiv-search config per distinct (query, since_days) pair.

    Theme-owned queries and since_days are preferred. The global values remain
    fallbacks for themes that do not yet have the KG properties set.
    """
    seen: set[tuple[str, int]] = set()
    result: list[Config] = []
    for topic in topics:
        query = (topic.arxiv_query or config.arxiv_query).strip()
        days = _effective_since_days(topic, config)
        if query and (query, days) not in seen:
            seen.add((query, days))
            result.append(dataclasses.replace(config, arxiv_query=query, since_days=days))
    if not result and config.arxiv_query.strip():
        result.append(config)
    return result


def _openalex_query_configs(config: Config, topics: list[Topic]) -> list[tuple[str, int]]:
    """Collect unique (openalex_query, since_days) pairs from topics, preserving order."""
    seen: set[tuple[str, int]] = set()
    result: list[tuple[str, int]] = []
    for topic in topics:
        q = topic.openalex_query.strip()
        if not q:
            continue
        days = _effective_since_days(topic, config)
        if (q, days) not in seen:
            seen.add((q, days))
            result.append((q, days))
    return result


def _zbmath_query_configs(config: Config, topics: list[Topic]) -> list[tuple[str, int]]:
    """Collect unique (zbmath_query, since_days) pairs from topics, preserving order."""
    seen: set[tuple[str, int]] = set()
    result: list[tuple[str, int]] = []
    for topic in topics:
        q = (topic.zbmath_query or config.zbmath_query).strip()
        if not q:
            continue
        days = _effective_since_days(topic, config)
        if (q, days) not in seen:
            seen.add((q, days))
            result.append((q, days))
    return result


def _process_record(
    record,
    source_label: str,
    covering: list[Topic],
    config: Config,
    state,
    kg,
    classify,
    summarize,
    keyworder,
    llm,
    model: str | None,
    topic_label: dict,
    imported_titles: list,
    imported_qids: list,
    theme_added: dict,
    lookup_zb=None,
    link_cites_fn=None,
) -> bool:
    """Classify, optionally import, and link one record. Returns True if imported."""
    rid = record.record_id
    if rid in state.seen_ids:
        return False
    state.seen_ids.add(rid)

    existing_qid = None
    find_existing_paper = getattr(kg, "find_existing_paper", None)
    paper_has_tldr = getattr(kg, "paper_has_tldr", None)
    if callable(find_existing_paper) and callable(paper_has_tldr):
        existing_qid = find_existing_paper(record)
        if existing_qid and paper_has_tldr(existing_qid):
            # Paper already fully imported — but write any identifiers the record
            # now provides that are still missing (e.g. DOI or arXiv ID added later).
            write_missing_ids = getattr(kg, "write_missing_identifiers", None)
            if callable(write_missing_ids) and not config.dry_run:
                write_missing_ids(existing_qid, record)
            log.info(
                "Skipping %s paper %s (%s): KG item %s already has P1963",
                source_label, rid, record.title, existing_qid,
            )
            return False

    keyword_matched = [
        t for t in covering
        if t.matches_keywords(record.title, record.abstract)
    ]
    if keyword_matched:
        for t in keyword_matched:
            kw = t.matches_keywords(record.title, record.abstract)
            log.info(
                "Auto-classifying %s paper %s (%s) to %s (%s) via keyword %r",
                source_label, rid, record.title, t.label, t.qid, kw,
            )
        matched = [t.qid for t in keyword_matched]
    else:
        log.info("Classifying %s paper %s (%s) with model %s", source_label, rid, record.title, model)
        matched = classify(
            record,
            covering,
            model=model,
            api_key=config.anthropic_api_key,
            llm=llm,
        )
    matched_labels = [f"{topic_label.get(q, q)} ({q})" for q in matched] if matched else []
    log.info(
        "Classified %s paper %s (%s) into: %s",
        source_label, rid, record.title,
        ", ".join(matched_labels) if matched_labels else "no matching theme",
    )
    if not matched:
        return False

    if not config.dry_run:
        log.info("Generating TL;DR and keywords for %s (%s)", rid, record.title)
        tldr = summarize(record, model=model, api_key=config.anthropic_api_key, llm=llm)
        if not tldr:
            log.warning("Skipping %s (%s): LLM returned empty TL;DR", rid, record.title)
            return False
        keywords = keyworder(record, model=model, api_key=config.anthropic_api_key, llm=llm)
        paper_qid = kg.import_paper(
            record, tldr=tldr, keywords=keywords, generated_by=M.Q_WORKFLOW,
            tldr_model_qid=config.model_qid or None,
        )
        imported_qids.append(paper_qid)
        log.info("Inserted %s paper %s as KG item %s", source_label, rid, paper_qid)
        # Upload arXiv HTML5-derived Markdown to lakeFS (skip if not configured or no arXiv ID)
        if record.arxiv_id and config.lakefs_url and config.lakefs_user:
            try:
                markdown = fetch_and_convert(record.arxiv_id)
                lakefs_path = upload_markdown(
                    paper_qid,
                    markdown,
                    url=config.lakefs_url,
                    user=config.lakefs_user,
                    password=config.lakefs_password,
                    repo=config.lakefs_repo,
                    branch=config.lakefs_branch,
                )
                log.info("Uploaded arXiv HTML markdown for %s to lakeFS: %s", paper_qid, lakefs_path)
                commit_upload(
                    f"Add arXiv fulltext for {paper_qid}",
                    url=config.lakefs_url,
                    user=config.lakefs_user,
                    password=config.lakefs_password,
                    repo=config.lakefs_repo,
                    branch=config.lakefs_branch,
                    metadata={"qid": paper_qid, "arxiv_id": record.arxiv_id},
                )
            except Exception as exc:
                log.warning("lakeFS upload failed for %s (%s): %s", paper_qid, record.arxiv_id, exc)
        # Inline zbMATH enrichment for arXiv/OpenAlex records (adds P225 + P676 author resolution).
        zb = None
        if callable(lookup_zb) and record.arxiv_id and not record.zbmath_id:
            zb = lookup_zb(record.arxiv_id)
            if zb:
                log.info(
                    "zbMATH enrichment: found %s (DE:%s) for arXiv:%s — adding P225/P1451/P226 + P676 authors",
                    zb.zbmath_id, zb.zbmath_de_number, record.arxiv_id,
                )
                kg.add_zbmath_enrichment(
                    paper_qid, zb.zbmath_id, zb.zbmath_author_ids,
                    zbmath_de_number=zb.zbmath_de_number,
                    msc_codes=zb.msc_codes,
                    zbmath_keywords=zb.zbmath_keywords,
                    journal_title=zb.journal_title,
                    license_url=zb.license_url,
                )
        # Citation linking: resolve cited papers already in the KG and write P223 claims.
        if callable(link_cites_fn):
            try:
                link_cites_fn(paper_qid, record, zb)
            except Exception as exc:
                log.warning("Citation linking for %s failed: %s", paper_qid, exc)
        for topic_qid in matched:
            kg.link_topic(topic_qid, paper_qid)
            theme_added.setdefault(topic_qid, []).append((paper_qid, record.title))
            log.info(
                "Linked %s to theme %s (%s)", paper_qid,
                topic_label.get(topic_qid, topic_qid), topic_qid,
            )
            if config.theme_max_papers:
                kg.enforce_theme_limit(topic_qid, config.theme_max_papers)
    imported_titles.append(record.title)
    return True


def harvest_step(
    config: Config,
    state: State,
    *,
    topics: list[Topic],
    kg,
    fetch=default_harvest,
    fetch_oa=None,
    fetch_zb=None,
    lookup_zb=None,
    lookup_oa_enrichment=None,
    classify=classify_paper,
    summarize=summarize_paper,
    keyworder=keywords_paper,
    llm=None,
    model: str | None = None,
    publisher=None,
) -> int:
    imported = 0
    imported_titles: list[str] = []
    imported_qids: list[str] = []
    theme_added: dict[str, list[tuple[str, str]]] = {}
    topic_label = {t.qid: t.label for t in topics}
    llm = llm or make_llm_client(config)
    model = model or config.model_qid

    from .harvest.zbmath import lookup_by_arxiv_id as _zb_lookup
    _lookup_zb = lookup_zb if lookup_zb is not None else _zb_lookup
    _lookup_oa_enrichment = lookup_oa_enrichment if lookup_oa_enrichment is not None else lookup_openalex_enrichment

    sparql_endpoint = config.sparql_endpoint_url or ""
    openalex_email = config.openalex_email or ""

    mediawiki_api_url = config.mediawiki_api_url or ""
    s2_api_key = config.s2_api_key or ""

    def _link_cites(paper_qid: str, record, zb_record=None) -> None:
        """Fetch cited paper IDs and write P223 claims, skipping silently on any error."""
        if not sparql_endpoint:
            return
        zbmath_id = getattr(record, "zbmath_id", "") or (
            zb_record.zbmath_id if zb_record else ""
        )
        doi = getattr(record, "doi", "") or ""

        cited_qids: list[str] = []

        # 1. zbMATH (highest quality — includes MSC-resolved references)
        if zbmath_id:
            doc_ids = fetch_zbmath_references(zbmath_id)
            cited_qids = resolve_qids_by_zbmath_doc_ids(doc_ids, sparql_endpoint)

        # 2. Crossref (publisher DOIs only — more complete than OpenAlex for journal papers)
        if not cited_qids and doi:
            crossref = fetch_crossref_data(doi)
            cited_qids = resolve_qids_by_dois(crossref["reference_dois"], sparql_endpoint)
            if crossref["license_url"] and not config.dry_run:
                try:
                    kg.add_crossref_license(paper_qid, crossref["license_url"])
                except Exception as exc:
                    log.warning("Failed to write Crossref license for %s: %s", paper_qid, exc)

        # 3. OpenAlex (covers arXiv preprints and papers not in Crossref)
        if not cited_qids and getattr(record, "openalex_id", ""):
            oa_work_ids = fetch_openalex_referenced_works(
                record.openalex_id, email=openalex_email
            )
            cited_qids = resolve_qids_by_openalex_ids(oa_work_ids, sparql_endpoint)

        if cited_qids:
            kg.link_citations(paper_qid, cited_qids)
            return

        # 4. Semantic Scholar fallback
        s2_refs = fetch_s2_references(
            arxiv_id=getattr(record, "arxiv_id", "") or "",
            doi=doi,
            api_key=s2_api_key,
        )
        s2_qids = resolve_s2_references(s2_refs, sparql_endpoint, mediawiki_api_url)
        if s2_qids:
            kg.link_citations(paper_qid, s2_qids, reference_qid=M.Q_SEMANTIC_SCHOLAR)

    # --- zbMATH pass (highest quality — runs first; records carry author codes for P676 resolution) ---
    _fetch_zb = fetch_zb or (lambda qs, sd, **kw: fetch_zbmath_records(qs, sd))
    try:
        for zb_query, zb_since_days in _zbmath_query_configs(config, topics):
            covering = [
                t for t in topics
                if (t.zbmath_query or config.zbmath_query).strip() == zb_query
                and _effective_since_days(t, config) == zb_since_days
            ]
            log.info(
                "Harvesting zbMATH query %r (since_days=%d) for %d theme(s): %s",
                zb_query, zb_since_days, len(covering),
                ", ".join(f"{t.label} ({t.qid})" for t in covering),
            )
            query_imported = 0
            for record in _fetch_zb(zb_query, zb_since_days):
                try:
                    if record.doi or record.arxiv_id:
                        oa_enrichment = _lookup_oa_enrichment(
                            record.doi, record.arxiv_id,
                            email=config.openalex_email,
                        )
                        if oa_enrichment:
                            if oa_enrichment["published"]:
                                log.info(
                                    "OpenAlex date enrichment for zbMATH %s: %s → %s",
                                    record.record_id, record.published, oa_enrichment["published"],
                                )
                                record.published = oa_enrichment["published"]
                            record.oa_status = oa_enrichment["oa_status"]
                            record.concepts = oa_enrichment["concepts"]
                            record.openalex_keywords = oa_enrichment["openalex_keywords"]
                    did_import = _process_record(
                        record, "zbMATH", covering, config, state, kg,
                        classify, summarize, keyworder, llm, model,
                        topic_label, imported_titles, imported_qids, theme_added,
                        link_cites_fn=_link_cites,
                    )
                except PipelineError:
                    raise
                except Exception as exc:
                    log.warning(
                        "Skipping zbMATH paper %s due to error: %s",
                        getattr(record, "zbmath_id", "?"), exc,
                    )
                    continue
                if did_import:
                    imported += 1
                    query_imported += 1
                if config.harvest_limit and query_imported >= config.harvest_limit:
                    break
    except PipelineError:
        raise
    except Exception as exc:
        log.warning("zbMATH harvest pass failed (%s) — continuing with OpenAlex and arXiv", exc)

    # --- OpenAlex pass ---
    _fetch_oa = fetch_oa or (
        lambda qs, sd, **kw: fetch_openalex_records(
            qs, sd, email=config.openalex_email
        )
    )
    try:
        for oa_query, oa_since_days in _openalex_query_configs(config, topics):
            covering = [
                t for t in topics
                if t.openalex_query.strip() == oa_query
                and _effective_since_days(t, config) == oa_since_days
            ]
            log.info(
                "Harvesting OpenAlex query %r (since_days=%d) for %d theme(s): %s",
                oa_query, oa_since_days, len(covering),
                ", ".join(f"{t.label} ({t.qid})" for t in covering),
            )
            query_imported = 0
            for record in _fetch_oa(oa_query, oa_since_days):
                try:
                    did_import = _process_record(
                        record, "OpenAlex", covering, config, state, kg,
                        classify, summarize, keyworder, llm, model,
                        topic_label, imported_titles, imported_qids, theme_added,
                        lookup_zb=_lookup_zb,
                        link_cites_fn=_link_cites,
                    )
                except PipelineError:
                    raise
                except Exception as exc:
                    log.warning(
                        "Skipping OpenAlex paper %s due to error: %s",
                        getattr(record, "openalex_id", "?"), exc,
                    )
                    continue
                if did_import:
                    imported += 1
                    query_imported += 1
                if config.harvest_limit and query_imported >= config.harvest_limit:
                    break
    except PipelineError:
        raise
    except Exception as exc:
        log.warning("OpenAlex harvest pass failed (%s) — continuing with arXiv", exc)

    # --- arXiv pass (fallback for papers not yet indexed by zbMATH or OpenAlex) ---
    first_arxiv_query = True
    for harvest_config in _harvest_configs(config, topics):
        if not first_arxiv_query:
            time.sleep(3)
        first_arxiv_query = False
        covering = [
            t for t in topics
            if (t.arxiv_query or config.arxiv_query).strip() == harvest_config.arxiv_query
            and _effective_since_days(t, config) == harvest_config.since_days
        ]
        log.info(
            "Harvesting arXiv query %r for %d theme(s): %s",
            harvest_config.arxiv_query, len(covering),
            ", ".join(f"{t.label} ({t.qid})" for t in covering),
        )
        query_imported = 0
        for record in fetch(harvest_config):
            try:
                did_import = _process_record(
                    record, "arXiv", covering, config, state, kg,
                    classify, summarize, keyworder, llm, model,
                    topic_label, imported_titles, imported_qids, theme_added,
                    lookup_zb=_lookup_zb,
                    link_cites_fn=_link_cites,
                )
            except PipelineError:
                raise
            except Exception as exc:
                log.warning("Skipping paper %s due to error: %s", record.record_id, exc)
                continue
            if did_import:
                imported += 1
                query_imported += 1
            if config.harvest_limit and query_imported >= config.harvest_limit:
                break

    if publisher is not None and not config.dry_run and imported_titles:
        publisher.purge(imported_titles)

    # Per-theme summary
    for topic in topics:
        entries = theme_added.get(topic.qid, [])
        if entries:
            log.info(
                "Theme %r (%s): %d new paper(s)",
                topic.label, topic.qid, len(entries),
            )
            for qid, title in entries:
                log.info("  %s — %s", qid, title)
        else:
            log.info("Theme %r (%s): 0 new paper(s)", topic.label, topic.qid)

    if imported_qids:
        log.info("Harvest inserted %d paper(s): %s", imported, ", ".join(imported_qids))
    else:
        log.info("Harvest inserted 0 papers")
    return imported


def ensure_theme_pages_step(
    config: Config,
    *,
    topics: list[Topic],
    publisher,
    kg,
) -> list[str]:
    """Make every research theme renderable: ensure each has a ``{{ResearchTheme}}``
    wiki page connected to its item via the ``mardi`` sitelink.

    Per theme:
      - already has a sitelink -> leave it (page is wired; curator owns content);
      - else if a page with the target title already exists -> log and skip (do
        not hijack an unrelated page);
      - else -> create the stub page and set the item's sitelink to it.

    The page title is the theme label (main namespace, Person convention).
    Returns the theme labels processed.
    """
    titles = [topic.label for topic in topics]
    if config.dry_run:
        return titles

    page_titles: list[str] = []
    for topic in topics:
        connected = kg.get_theme_sitelink(topic.qid)
        if connected:
            page_titles.append(connected)  # already wired; refresh its (cached) table
            continue
        title = topic.label
        if publisher.page_exists(title):
            log.warning(
                "Theme %s (%s): page %r already exists and is not linked — "
                "skipping to avoid hijacking it; connect it manually.",
                topic.qid, topic.label, title,
            )
            continue
        publisher.edit(title, RESEARCH_THEME_STUB, "Create research theme page")
        kg.set_theme_sitelink(topic.qid, title)
        page_titles.append(title)

    # Purge theme pages so their live tables refresh.
    publisher.purge(page_titles)
    return titles
