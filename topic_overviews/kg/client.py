"""Write papers (canonical) and topic membership into the MaRDI Wikibase.

Membership lives on the TOPIC item: ``topic --has part(s) (P265)--> paper``.
Paper items stay topic-agnostic — they carry no statement about belonging to a
topic, so the canonical paper entity is never polluted by this feature.
"""
from __future__ import annotations

from ..harvest.arxiv_oai import PaperRecord
from . import model as M


def to_wbi_time(date: str) -> str:
    return f"+{date}T00:00:00Z"


class KGClient:
    def __init__(self, mc):
        self.mc = mc

    def import_paper(self, record: PaperRecord) -> str:
        """Upsert the canonical paper item (idempotent by arXiv ID). Returns its QID.

        Writes only paper-intrinsic statements; nothing about topics.
        """
        existing = self.mc.search_entity_by_value(f"wdt:{M.P_ARXIV_ID}", record.arxiv_id)
        if existing:
            item = self.mc.item.get(entity_id=existing[0])
        else:
            item = self.mc.item.new()
            item.labels.set("en", record.title[:250])

        item.add_claim(f"wdt:{M.P_INSTANCE_OF}", value=f"wd:{M.Q_SCHOLARLY_ARTICLE}")
        item.add_claim(f"wdt:{M.P_ARXIV_ID}", value=record.arxiv_id)
        if record.doi:
            item.add_claim(f"wdt:{M.P_DOI}", value=record.doi)
        item.add_claim(f"wdt:{M.P_TITLE}", value=record.title)
        if record.published:
            item.add_claim(f"wdt:{M.P_PUBLICATION_DATE}", value=to_wbi_time(record.published))
        for cat in record.categories:
            item.add_claim(f"wdt:{M.P_ARXIV_CLASSIFICATION}", value=cat)
        for name in record.authors:
            item.add_claim(f"wdt:{M.P_AUTHOR_NAME_STRING}", value=name)

        return item.write().id

    def link_topic(self, topic_qid: str, paper_qid: str) -> None:
        """Add the paper to the topic's ``has part(s)`` (P265) list, idempotently.

        Reads the topic's current membership first and only writes when the paper
        is not already listed, so re-runs never duplicate entries.
        """
        topic_item = self.mc.item.get(entity_id=topic_qid)
        current = topic_item.get_value(f"wdt:{M.P_HAS_PART}") or []
        if paper_qid in current:
            return
        topic_item.add_claim(f"wdt:{M.P_HAS_PART}", value=f"wd:{paper_qid}")
        topic_item.write()


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
    return KGClient(mc)
