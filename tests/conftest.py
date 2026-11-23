"""Test-wide guards: no real HTTP and no real sleeping.

- No real network at all: DNS lookups and outgoing socket connections raise
  ``NetworkBlocked`` too, which also covers clients that bypass ``requests``
  (the lakeFS SDK's urllib3, httpx). Unix sockets stay allowed.
- Any request that reaches ``requests.Session.request`` (also used by
  ``requests.get``/``requests.post``) fails with ``NetworkBlocked``. Tests that
  need HTTP must stub it (patch ``requests.get``, pass a fake session, ...).
  ``NetworkBlocked`` is a ``requests.ConnectionError``, so code under test sees
  an ordinary network failure.
- ``time.sleep`` is a no-op, so retry back-offs and rate-limit pauses cost
  nothing. Tests can still patch it themselves to assert on calls.
"""
import socket
import time

import pytest
import requests


class NetworkBlocked(requests.exceptions.ConnectionError):
    pass


@pytest.fixture(autouse=True)
def _no_network_no_sleep(monkeypatch):
    def blocked(self, method, url, *args, **kwargs):
        raise NetworkBlocked(f"Real HTTP request blocked in tests: {method} {url!r}")

    def blocked_dns(host, *args, **kwargs):
        raise NetworkBlocked(f"Real DNS lookup blocked in tests: {host!r}")

    real_connect = socket.socket.connect

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise NetworkBlocked(f"Real network connection blocked in tests: {address!r}")
        return real_connect(sock, address)

    monkeypatch.setattr(requests.Session, "request", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked_dns)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
