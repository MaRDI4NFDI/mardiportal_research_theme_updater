import datetime
import logging

from topic_overviews.harvest.arxiv_oai import PaperRecord
from topic_overviews.harvest.arxiv_search import parse_atom, search_records

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2606.01234v1</id>
    <published>2026-06-20T00:00:00Z</published>
    <title>A New Finite Element Method
       for Stability</title>
    <summary>We propose a numerical scheme with error analysis.</summary>
    <author><name>Jane Doe</name></author>
    <author><name>John Smith</name></author>
    <arxiv:doi>10.1000/xyz</arxiv:doi>
    <category term="math.NA"/>
    <category term="cs.NA"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2605.09999v2</id>
    <published>2026-05-01T00:00:00Z</published>
    <title>An Older Paper</title>
    <summary>Out of the window.</summary>
    <author><name>Old Author</name></author>
    <category term="math.NA"/>
  </entry>
</feed>"""

EMPTY = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"""


def test_parse_atom_extracts_fields():
    recs = parse_atom(ATOM)
    assert recs[0] == PaperRecord(
        arxiv_id="2606.01234",
        title="A New Finite Element Method for Stability",
        abstract="We propose a numerical scheme with error analysis.",
        authors=["Jane Doe", "John Smith"],
        categories=["math.NA", "cs.NA"],
        published="2026-06-20",
        doi="10.1000/xyz",
    )
    assert recs[1].arxiv_id == "2605.09999" and recs[1].doi == "10.48550/arXiv.2605.09999"


class FakeResp:
    status_code = 200
    def __init__(self, text): self.text = text
    def raise_for_status(self): pass


class FakeSession:
    def __init__(self, pages): self._pages = list(pages); self.calls = []
    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResp(self._pages.pop(0) if self._pages else EMPTY)


def test_search_stops_at_window_boundary():
    session = FakeSession([ATOM])
    got = list(search_records(
        "cat:math.NA", since_days=10, session=session,
        sleep=lambda s: None, today=datetime.date(2026, 6, 21),
    ))
    # cutoff = 2026-06-11; first entry (06-20) kept, second (05-01) ends iteration
    assert [r.arxiv_id for r in got] == ["2606.01234"]
    assert len(session.calls) == 1                       # stopped before paging
    assert session.calls[0]["sortBy"] == "submittedDate"
    assert session.calls[0]["sortOrder"] == "descending"


def test_search_logs_fetch_and_result_count(caplog):
    session = FakeSession([ATOM])
    with caplog.at_level(logging.INFO, logger="topic_overviews.harvest.arxiv_search"):
        list(
            search_records(
                "cat:math.NA",
                since_days=10,
                session=session,
                sleep=lambda s: None,
                today=datetime.date(2026, 6, 21),
            )
        )
    assert "Fetching arXiv results for query='cat:math.NA' start=0 page_size=100 cutoff=2026-06-11" in caplog.text
    assert "Got 2 arXiv results for query='cat:math.NA'" in caplog.text
