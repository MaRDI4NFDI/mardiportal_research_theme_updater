"""HTTP helpers with built-in retry logic for transient server errors."""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

_RETRIES = 3
_DELAY = 5.0


def http_get(session, url: str, *, retries: int = _RETRIES, delay: float = _DELAY, **kwargs) -> requests.Response:
    return _request(session, "get", url, retries=retries, delay=delay, **kwargs)


def http_post(session, url: str, *, retries: int = _RETRIES, delay: float = _DELAY, **kwargs) -> requests.Response:
    return _request(session, "post", url, retries=retries, delay=delay, **kwargs)


def _request(session, method: str, url: str, *, retries: int, delay: float, **kwargs) -> requests.Response:
    last_exc: Exception
    for attempt in range(1, retries + 1):
        try:
            resp = getattr(session, method)(url, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                log.warning(
                    "HTTP %s %s failed (attempt %d/%d): %s — retrying in %.0fs",
                    method.upper(), url, attempt, retries, exc, delay,
                )
                time.sleep(delay)
    raise last_exc
