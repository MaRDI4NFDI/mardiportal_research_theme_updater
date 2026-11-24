"""HTTP helpers that wait out temporary outages of the services we call.

The MaRDI portal sits behind an ingress, so a restarting backend can answer
with 404 as well as 502/503. Requests are therefore retried with a growing
pause on connection errors, timeouts and most HTTP errors. Only errors that a
retry cannot fix (bad request, bad credentials) fail at once.

POST requests change data, so they are retried less eagerly: only when the
server most likely never processed the request (connection error, 404, 429,
502, 503). After a read timeout or a 500/504 the write may have gone through,
and a retry could create a duplicate claim; those errors are raised.
"""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

_RETRIES = 5            # attempts in total
_DELAY = 5.0            # pause after the first failed attempt; doubles each time
_MAX_DELAY = 60.0       # cap on one pause
_MAX_RETRY_AFTER = 120.0  # cap on a server-requested Retry-After wait

# A retry cannot fix these: the request itself or the credentials are wrong.
_NO_RETRY_STATUS = {400, 401, 403, 422}
# Safe for POST: the server answered without having processed the request.
_POST_RETRY_STATUS = {404, 429, 502, 503}


def http_get(session, url: str, *, retries: int = _RETRIES, delay: float = _DELAY, **kwargs) -> requests.Response:
    return _request(session, "get", url, retries=retries, delay=delay, **kwargs)


def http_post(session, url: str, *, retries: int = _RETRIES, delay: float = _DELAY, **kwargs) -> requests.Response:
    return _request(session, "post", url, retries=retries, delay=delay, **kwargs)


def _should_retry(method: str, exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if method == "post":
            return status in _POST_RETRY_STATUS
        return status not in _NO_RETRY_STATUS
    if method == "post":
        # ConnectionError includes ConnectTimeout; a ReadTimeout is not retried.
        return isinstance(exc, requests.exceptions.ConnectionError)
    return isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout))


def _pause(exc: Exception, attempt: int, delay: float) -> float:
    """Seconds to wait after failed *attempt*: Retry-After if given, else backoff."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return min(float(response.headers.get("Retry-After", "")), _MAX_RETRY_AFTER)
        except (TypeError, ValueError, AttributeError):
            pass
    return min(delay * 2 ** (attempt - 1), _MAX_DELAY)


def _request(session, method: str, url: str, *, retries: int, delay: float, **kwargs) -> requests.Response:
    for attempt in range(1, retries + 1):
        try:
            resp = getattr(session, method)(url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            if attempt >= retries or not _should_retry(method, exc):
                raise
            wait = _pause(exc, attempt, delay)
            log.warning(
                "HTTP %s %s failed (attempt %d/%d): %s — retrying in %.0fs",
                method.upper(), url, attempt, retries, exc, wait,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")
