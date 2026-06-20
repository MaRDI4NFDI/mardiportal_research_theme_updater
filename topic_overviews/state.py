"""Persistent harvest cursor + de-duplication of seen arXiv IDs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class State:
    last_harvest: str | None = None      # ISO date "YYYY-MM-DD"
    seen_ids: set[str] = field(default_factory=set)


def load_state(path: str) -> State:
    if not os.path.exists(path):
        return State()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return State(
        last_harvest=data.get("last_harvest"),
        seen_ids=set(data.get("seen_ids", [])),
    )


def save_state(path: str, state: State) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {"last_harvest": state.last_harvest, "seen_ids": sorted(state.seen_ids)},
            f,
        )
    os.replace(tmp, path)
