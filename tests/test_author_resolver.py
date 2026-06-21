"""Tests for AuthorResolver — all HTTP mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from topic_overviews.kg.author_resolver import AuthorResolver


def _mock_session(search_hits: list[list[dict]], orcid_claims: dict[str, str] = {}) -> MagicMock:
    """Build a mock requests.Session whose .get returns canned responses."""
    session = MagicMock()
    responses = []

    for hits in search_hits:
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"search": hits}
        responses.append(r)

    for qid, orcid in orcid_claims.items():
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {
            "entities": {
                qid: {"claims": {"P20": [{"mainsnak": {"datavalue": {"type": "string", "value": orcid}}}]}}
            }
        }
        responses.append(r)

    session.get.side_effect = responses
    return session


def _hit(qid: str, label: str) -> dict:
    return {"id": qid, "label": label}


class TestAuthorResolver:
    def test_single_kg_hit_returns_qid(self):
        session = _mock_session([[_hit("Q123", "Alice Smith")]])
        resolver = AuthorResolver("https://example.org/api", session=session)
        assert resolver.resolve("Alice Smith") == "Q123"

    def test_no_kg_hit_returns_none(self):
        session = _mock_session([[]])
        resolver = AuthorResolver("https://example.org/api", session=session)
        assert resolver.resolve("Unknown Person") is None

    def test_multiple_hits_disambiguated_by_orcid(self):
        # Two KG hits; second one matches the ORCID from OpenAlex.
        session = _mock_session(
            [[_hit("Q1", "Stephan Rave"), _hit("Q2", "Stephan Rave")]],
            orcid_claims={"Q1": "0000-0000-0000-0001", "Q2": "0000-0003-0439-7212"},
        )
        resolver = AuthorResolver("https://example.org/api", session=session)
        with patch.object(resolver, "_openalex_orcid", return_value="0000-0003-0439-7212"):
            result = resolver.resolve("Stephan Rave")
        assert result == "Q2"

    def test_multiple_hits_no_orcid_match_returns_none(self):
        session = _mock_session(
            [[_hit("Q1", "Common Name"), _hit("Q2", "Common Name")]],
            orcid_claims={"Q1": "0000-0000-0000-0001", "Q2": "0000-0000-0000-0002"},
        )
        resolver = AuthorResolver("https://example.org/api", session=session)
        with patch.object(resolver, "_openalex_orcid", return_value="0000-0000-0000-9999"):
            result = resolver.resolve("Common Name")
        assert result is None

    def test_result_is_cached(self):
        session = _mock_session([[_hit("Q123", "Alice Smith")]])
        resolver = AuthorResolver("https://example.org/api", session=session)
        assert resolver.resolve("Alice Smith") == "Q123"
        assert resolver.resolve("Alice Smith") == "Q123"
        assert session.get.call_count == 1  # only one HTTP call despite two resolves

    def test_kg_search_exception_returns_none(self):
        session = MagicMock()
        session.get.side_effect = Exception("network error")
        resolver = AuthorResolver("https://example.org/api", session=session)
        assert resolver.resolve("Someone") is None

    def test_openalex_orcid_strips_url_prefix(self):
        resolver = AuthorResolver("https://example.org/api")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [{"orcid": "https://orcid.org/0000-0002-6260-3574", "works_count": 100}]
        }
        with patch("requests.get", return_value=mock_resp):
            orcid = resolver._openalex_orcid("Mario Ohlberger")
        assert orcid == "0000-0002-6260-3574"

    def test_openalex_prefers_highest_works_count(self):
        resolver = AuthorResolver("https://example.org/api")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {"orcid": "https://orcid.org/0000-0000-0000-0001", "works_count": 1},
                {"orcid": "https://orcid.org/0000-0003-2521-4921", "works_count": 22},
            ]
        }
        with patch("requests.get", return_value=mock_resp):
            orcid = resolver._openalex_orcid("Dmitry Kabanov")
        assert orcid == "0000-0003-2521-4921"

    def test_openalex_exception_returns_none(self):
        resolver = AuthorResolver("https://example.org/api")
        with patch("requests.get", side_effect=Exception("timeout")):
            assert resolver._openalex_orcid("Anyone") is None
