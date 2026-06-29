"""Verify that _process_record triggers lakeFS upload when credentials are configured."""
from unittest.mock import patch, MagicMock
from topic_overviews.pipeline import _process_record
from topic_overviews.config import Config
from topic_overviews.harvest.arxiv_oai import PaperRecord


def _make_config(**overrides) -> Config:
    base = dict(
        arxiv_query="", arxiv_query_property="P1965",
        openalex_query_property="", openalex_email="",
        zbmath_query="", zbmath_query_property="", since_days_property="",
        auto_classify_keywords_property="", maintainer_qid="",
        since_days=10, llm_provider="openai",
        openai_base_url="", openai_api_key="",
        research_theme_qid="Q1", model_qid="", harvest_limit=0,
        theme_max_papers=100, anthropic_api_key="",
        mediawiki_api_url="", mediawiki_bot_user="", mediawiki_bot_password="",
        wikibase_url="", sparql_endpoint_url="", s2_api_key="",
        dry_run=False,
        lakefs_url="https://lakefs.example.org",
        lakefs_user="AKID", lakefs_password="SECRET",
        lakefs_repo="mardi-portal", lakefs_branch="main",
    )
    base.update(overrides)
    return Config(**base)


def _make_record(arxiv_id="2606.28184", openalex_id=""):
    return PaperRecord(
        arxiv_id=arxiv_id, openalex_id=openalex_id,
        title="Test Paper", abstract="An abstract.",
        authors=["Alice"], categories=["math.NA"], published="2026-01-01",
    )


class _FakeState:
    def __init__(self):
        self.seen_ids: set = set()


def test_upload_called_after_import():
    record = _make_record()
    config = _make_config()

    kg = MagicMock()
    kg.find_existing_paper.return_value = None
    kg.paper_has_tldr.return_value = False
    kg.import_paper.return_value = "Q9999"

    with patch("topic_overviews.pipeline.fetch_and_convert", return_value="# md") as mock_fetch, \
         patch("topic_overviews.pipeline.upload_markdown", return_value="main/00/00/99/Q9999/fulltext/Q9999.md") as mock_upload:
        _process_record(
            record, "arxiv", [MagicMock(qid="Q1", matches_keywords=MagicMock(return_value=["kw"]))],
            config, _FakeState(), kg,
            classify=MagicMock(return_value=["Q1"]),
            summarize=MagicMock(return_value="tldr"),
            keyworder=MagicMock(return_value=["kw"]),
            llm=None, model=None,
            topic_label={"Q1": "Theme"},
            imported_titles=[], imported_qids=[],
            theme_added={},
        )

    mock_fetch.assert_called_once_with("2606.28184")
    mock_upload.assert_called_once_with(
        "Q9999", "# md",
        url="https://lakefs.example.org",
        user="AKID",
        password="SECRET",
        repo="mardi-portal",
        branch="main",
    )


def test_upload_skipped_when_no_arxiv_id():
    record = _make_record(arxiv_id="", openalex_id="W9999")
    config = _make_config()
    kg = MagicMock()
    kg.find_existing_paper.return_value = None
    kg.paper_has_tldr.return_value = False
    kg.import_paper.return_value = "Q9999"

    with patch("topic_overviews.pipeline.fetch_and_convert") as mock_fetch, \
         patch("topic_overviews.pipeline.upload_markdown") as mock_upload:
        _process_record(
            record, "zbmath", [MagicMock(qid="Q1", matches_keywords=MagicMock(return_value=["kw"]))],
            config, _FakeState(), kg,
            classify=MagicMock(return_value=["Q1"]),
            summarize=MagicMock(return_value="tldr"),
            keyworder=MagicMock(return_value=["kw"]),
            llm=None, model=None,
            topic_label={"Q1": "Theme"},
            imported_titles=[], imported_qids=[],
            theme_added={},
        )

    mock_fetch.assert_not_called()
    mock_upload.assert_not_called()


def test_upload_failure_does_not_abort():
    record = _make_record()
    config = _make_config()
    kg = MagicMock()
    kg.find_existing_paper.return_value = None
    kg.paper_has_tldr.return_value = False
    kg.import_paper.return_value = "Q9999"

    with patch("topic_overviews.pipeline.fetch_and_convert", side_effect=RuntimeError("network")), \
         patch("topic_overviews.pipeline.upload_markdown") as mock_upload:
        result = _process_record(
            record, "arxiv", [MagicMock(qid="Q1", matches_keywords=MagicMock(return_value=["kw"]))],
            config, _FakeState(), kg,
            classify=MagicMock(return_value=["Q1"]),
            summarize=MagicMock(return_value="tldr"),
            keyworder=MagicMock(return_value=["kw"]),
            llm=None, model=None,
            topic_label={"Q1": "Theme"},
            imported_titles=[], imported_qids=[],
            theme_added={},
        )

    assert result is True  # paper was still imported
    mock_upload.assert_not_called()
