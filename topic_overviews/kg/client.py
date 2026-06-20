"""Write papers and topic links into the MaRDI Wikibase via mardiclient."""
from __future__ import annotations

from ..harvest.arxiv_oai import PaperRecord
from . import model as M


def to_wbi_time(date: str) -> str:
    return f"+{date}T00:00:00Z"


class KGClient:
    def __init__(self, mc):
        self.mc = mc

    def import_paper(self, record: PaperRecord, topic_qids: list[str]) -> str:
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
        for tq in topic_qids:
            item.add_claim(f"wdt:{M.P_MAIN_SUBJECT}", value=f"wd:{tq}")

        return item.write().id


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
