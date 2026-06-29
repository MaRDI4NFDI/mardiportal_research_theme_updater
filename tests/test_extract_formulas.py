import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
import requests


def _make_obj(path):
    obj = MagicMock()
    obj.path = path
    return obj


def test_list_lakefs_papers_extracts_qids():
    """QIDs are parsed from the 4th path segment of .md.txt objects."""
    from maintenance.extract_formulas import list_lakefs_papers

    fake_objects = [
        _make_obj("61/90/92/Q6190920/fulltext/Q6190920.md.txt"),
        _make_obj("00/00/01/Q1/fulltext/Q1.md.txt"),
        _make_obj("61/90/92/Q6190920/fulltext/Q6190920.other"),  # not .md.txt → ignored
        _make_obj("some/other/file.txt"),  # too few segments → ignored
    ]

    with patch("maintenance.extract_formulas.lakefs") as mock_lf:
        mock_branch = MagicMock()
        mock_branch.objects.return_value = fake_objects
        mock_lf.Client.return_value = MagicMock()
        mock_lf.repository.return_value.branch.return_value = mock_branch

        result = list_lakefs_papers("http://fake", "user", "pass", "repo", "main")

    assert result == ["Q6190920", "Q1"]


def test_list_lakefs_papers_deduplicates():
    """Duplicate paths for the same QID produce one entry."""
    from maintenance.extract_formulas import list_lakefs_papers

    fake_objects = [
        _make_obj("61/90/92/Q6190920/fulltext/Q6190920.md.txt"),
        _make_obj("61/90/92/Q6190920/fulltext/Q6190920.md.txt"),  # duplicate
    ]

    with patch("maintenance.extract_formulas.lakefs") as mock_lf:
        mock_branch = MagicMock()
        mock_branch.objects.return_value = fake_objects
        mock_lf.Client.return_value = MagicMock()
        mock_lf.repository.return_value.branch.return_value = mock_branch

        result = list_lakefs_papers("http://fake", "user", "pass", "repo", "main")

    assert result == ["Q6190920"]


def test_get_paper_titles_maps_qids():
    from maintenance.extract_formulas import get_paper_titles

    fake_response = {
        "results": {
            "bindings": [
                {
                    "paper": {"value": "https://portal.mardi4nfdi.de/entity/Q6190920"},
                    "title": {"value": "A fast algorithm for sparse matrices"},
                },
            ]
        }
    }

    with patch("requests.Session.get") as mock_get:
        mock_get.return_value.json.return_value = fake_response
        mock_get.return_value.raise_for_status = MagicMock()

        session = requests.Session()
        result = get_paper_titles(["Q6190920", "Q9999"], "http://sparql", session)

    assert result["Q6190920"] == "A fast algorithm for sparse matrices"
    assert result["Q9999"] == ""  # not in response → empty string


def test_download_markdown_returns_content():
    from maintenance.extract_formulas import download_markdown

    fake_content = b"# Paper\n\n$$E = mc^2$$\n"

    with patch("maintenance.extract_formulas.lakefs") as mock_lf:
        mock_obj = MagicMock()
        mock_obj.reader.return_value.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=fake_content)))
        mock_obj.reader.return_value.__exit__ = MagicMock(return_value=False)
        mock_lf.Client.return_value = MagicMock()
        mock_lf.repository.return_value.branch.return_value.object.return_value = mock_obj

        result = download_markdown("Q6190920", "http://fake", "u", "p", "repo", "main")

    assert result == fake_content.decode("utf-8")


def test_download_markdown_raises_on_missing():
    from maintenance.extract_formulas import download_markdown

    with patch("maintenance.extract_formulas.lakefs") as mock_lf:
        mock_obj = MagicMock()
        mock_obj.reader.side_effect = Exception("not found")
        mock_lf.Client.return_value = MagicMock()
        mock_lf.repository.return_value.branch.return_value.object.return_value = mock_obj

        try:
            download_markdown("Q9999", "http://fake", "u", "p", "repo", "main")
            assert False, "should have raised"
        except FileNotFoundError:
            pass
