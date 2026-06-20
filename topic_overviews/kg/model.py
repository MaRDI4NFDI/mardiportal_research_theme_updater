"""Resolved MaRDI Wikibase property/class IDs. Verified 2026-06-20."""
from __future__ import annotations

# Properties
P_INSTANCE_OF = "P31"
P_ARXIV_ID = "P21"
P_ARXIV_CLASSIFICATION = "P22"
P_AUTHOR = "P16"
P_AUTHOR_NAME_STRING = "P43"
P_PUBLICATION_DATE = "P28"
P_TITLE = "P159"
P_DOI = "P27"
P_MAIN_SUBJECT = "P30"
P_SUBCLASS_OF = "P36"
P_AFFILIATION = "P55"

# Classes
Q_SCHOLARLY_ARTICLE = "Q56887"
Q_PREPRINT = "Q159099"


def qid_from_uri(uri: str) -> str:
    """Extract the trailing Q-id from an entity URI (or pass through a bare QID)."""
    return uri.rstrip("/").rsplit("/", 1)[-1]
