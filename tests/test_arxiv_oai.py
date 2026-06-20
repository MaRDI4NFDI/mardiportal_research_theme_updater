from pathlib import Path

from topic_overviews.harvest.arxiv_oai import PaperRecord, parse_oai_response, fetch_records

FIXTURE = (Path(__file__).parent / "fixtures" / "oai_listrecords.xml").read_text()


def test_parse_extracts_record_and_token():
    records, token = parse_oai_response(FIXTURE)
    assert token == "TOKEN123"
    assert len(records) == 1                      # deleted record skipped
    r = records[0]
    assert r == PaperRecord(
        arxiv_id="2401.00001",
        title="A New Bound for Online Caching",
        abstract="We prove a tighter competitive ratio for caching.",
        authors=["Jane Doe", "John Smith"],
        categories=["math.OC", "cs.DS"],
        published="2024-01-02",
        doi="10.1000/xyz",
    )


def test_fetch_records_follows_resumption_token():
    page1 = FIXTURE
    page2 = FIXTURE.replace("TOKEN123", "").replace("2401.00001", "2402.00009")

    class FakeResp:
        def __init__(self, text): self.text = text; self.status_code = 200
        def raise_for_status(self): pass

    calls = []

    class FakeSession:
        def get(self, url, params=None, timeout=None):
            calls.append(params)
            return FakeResp(page1 if len(calls) == 1 else page2)

    ids = [r.arxiv_id for r in fetch_records(None, session=FakeSession(), sleep=lambda s: None)]
    assert ids == ["2401.00001", "2402.00009"]
    assert "from" not in calls[0] or calls[0].get("from") is None
    assert calls[1]["resumptionToken"] == "TOKEN123"
