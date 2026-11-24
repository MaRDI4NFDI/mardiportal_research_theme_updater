from unittest.mock import patch, MagicMock
from topic_overviews.lakefs_upload import shard_qid, component_path, upload_markdown


def test_shard_qid_basic():
    assert shard_qid("Q6190920") == "61/90/92/Q6190920"


def test_shard_qid_short():
    # Q123 → digits "123" → zero-padded "000123" → 00/01/23/Q123
    assert shard_qid("Q123") == "00/01/23/Q123"


def test_shard_qid_lowercase():
    assert shard_qid("q6190920") == "61/90/92/Q6190920"


def test_shard_qid_invalid():
    try:
        shard_qid("X123")
        assert False
    except ValueError:
        pass


def test_component_path():
    assert component_path("Q6190920") == "61/90/92/Q6190920/fulltext/Q6190920.md.txt"


def test_upload_markdown_calls_lakefs():
    mock_client = MagicMock()
    mock_branch = MagicMock()
    mock_obj = MagicMock()
    mock_branch.object.return_value = mock_obj

    with patch("topic_overviews.lakefs_upload.lakefs.Client", return_value=mock_client), \
         patch("topic_overviews.lakefs_upload.lakefs.repository") as mock_repo:
        mock_repo.return_value.branch.return_value = mock_branch
        result = upload_markdown(
            "Q6190920", "# Hello\n$x=1$",
            url="https://lakefs.example.org",
            user="AKID",
            password="SECRET",
            repo="mardi-portal",
            branch="main",
        )

    mock_obj.upload.assert_called_once()
    call_kwargs = mock_obj.upload.call_args
    content = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("data", b"")
    assert b"# Hello" in content
    assert result == "main/61/90/92/Q6190920/fulltext/Q6190920.md.txt"


# --- "commit: no changes" is not a failure -----------------------------------

import json
import logging

import pytest
from lakefs.exceptions import ServerException

from topic_overviews.lakefs_upload import commit_upload, is_no_changes

COMMIT_ARGS = dict(url="https://lakefs.example.org", user="u", password="p", repo="r", branch="main")


def _no_changes():
    return ServerException(400, "Bad Request", json.dumps({"message": "commit: no changes"}))


def _commit_raising(exc):
    branch = MagicMock()
    branch.commit.side_effect = exc
    return branch


def _run_commit(branch):
    with patch("topic_overviews.lakefs_upload.lakefs.Client"), \
         patch("topic_overviews.lakefs_upload.lakefs.repository") as repo:
        repo.return_value.branch.return_value = branch
        return commit_upload("msg", **COMMIT_ARGS)


def test_is_no_changes_recognises_the_lakefs_response():
    assert is_no_changes(_no_changes())
    assert not is_no_changes(ServerException(400, "Bad Request", json.dumps({"message": "other"})))
    assert not is_no_changes(ServerException(500, "Err", json.dumps({"message": "commit: no changes"})))
    assert not is_no_changes(RuntimeError("commit: no changes"))


def test_commit_upload_returns_commit_id():
    branch = MagicMock()
    branch.commit.return_value.id = "abc123"
    assert _run_commit(branch) == "abc123"


def test_commit_upload_with_no_changes_is_not_an_error(caplog):
    with caplog.at_level(logging.INFO):
        assert _run_commit(_commit_raising(_no_changes())) == ""
    assert "nothing to commit" in caplog.text


def test_commit_upload_still_raises_other_errors():
    with pytest.raises(ServerException):
        _run_commit(_commit_raising(ServerException(500, "Server Error", json.dumps({"message": "boom"}))))
    with pytest.raises(RuntimeError):
        _run_commit(_commit_raising(RuntimeError("network")))
