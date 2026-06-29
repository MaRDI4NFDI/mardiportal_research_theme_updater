import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch


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
