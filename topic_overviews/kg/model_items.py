"""Read LLM model metadata from MaRDI Wikibase items."""
from __future__ import annotations

import requests

from ..http_utils import http_get

P_LLM_MODEL_IDENTIFIER = "P1966"


def get_llm_model_identifier(
    mediawiki_api_url: str,
    model_qid: str,
    *,
    property_id: str = P_LLM_MODEL_IDENTIFIER,
    session=None,
) -> str:
    """Return the provider-specific runtime model identifier stored on a model item."""
    if not model_qid:
        raise ValueError("TOPIC_OVERVIEWS_MODEL_QID is required")

    session = session or requests.Session()
    resp = http_get(
        session,
        mediawiki_api_url,
        params={
            "action": "wbgetentities",
            "ids": model_qid,
            "props": "claims",
            "format": "json",
        },
        timeout=60,
    )
    entity = resp.json()["entities"][model_qid]
    claims = entity.get("claims", {}).get(property_id, [])
    for claim in claims:
        datavalue = claim.get("mainsnak", {}).get("datavalue", {})
        if datavalue.get("type") == "string" and datavalue.get("value"):
            return str(datavalue["value"])
    raise ValueError(f"{model_qid} has no {property_id} LLM model identifier")
