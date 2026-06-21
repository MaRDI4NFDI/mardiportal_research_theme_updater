import pytest

from topic_overviews.kg.model_items import get_llm_model_identifier


class FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, *, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResp(self.payload)


def test_get_llm_model_identifier_reads_p1966_string():
    session = FakeSession(
        {
            "entities": {
                "Q1": {
                    "claims": {
                        "P1966": [
                            {
                                "mainsnak": {
                                    "datavalue": {
                                        "type": "string",
                                        "value": "claude-haiku-4-5",
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
    )
    assert (
        get_llm_model_identifier("http://api", "Q1", session=session)
        == "claude-haiku-4-5"
    )
    assert session.calls[0][1]["ids"] == "Q1"


def test_get_llm_model_identifier_requires_qid():
    with pytest.raises(ValueError):
        get_llm_model_identifier("http://api", "")


def test_get_llm_model_identifier_errors_when_missing_property():
    session = FakeSession({"entities": {"Q1": {"claims": {}}}})
    with pytest.raises(ValueError):
        get_llm_model_identifier("http://api", "Q1", session=session)
