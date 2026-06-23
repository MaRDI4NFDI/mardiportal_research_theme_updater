"""In-run deduplication of seen paper record IDs."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class State:
    seen_ids: set[str] = field(default_factory=set)
