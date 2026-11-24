"""Helpers for author names and identifiers."""
from __future__ import annotations

import re

_ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


def normalize_author_name(name: str) -> str:
    """Convert 'Last, First [Middle]' → 'First [Middle] Last'; pass through otherwise.

    "First Last" is how the portal stores author name strings; zbMATH delivers
    "Last, First".
    """
    if "," in name:
        last, _, rest = name.partition(",")
        return f"{rest.strip()} {last.strip()}".strip()
    return name


def normalize_orcid(value: str | None) -> str:
    """Bare 16-character ORCID iD (``0000-0002-1825-0097``) from a bare ID or an
    ``https://orcid.org/...`` URL; "" if *value* is not a valid ORCID iD."""
    orcid = (value or "").strip().rstrip("/").rsplit("/", 1)[-1].upper()
    return orcid if _ORCID.match(orcid) else ""
