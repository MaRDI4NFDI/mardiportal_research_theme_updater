"""Publish wikitext pages to MediaWiki via the action API (bot login)."""
from __future__ import annotations

import requests


class WikiPublisher:
    def __init__(self, api_url: str, user: str, password: str, session=None):
        self.api_url = api_url
        self.user = user
        self.password = password
        self.session = session or requests.Session()

    def _get_token(self, kind: str) -> str:
        resp = self.session.get(
            self.api_url,
            params={"action": "query", "meta": "tokens", "type": kind, "format": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        key = "logintoken" if kind == "login" else "csrftoken"
        return resp.json()["query"]["tokens"][key]

    def login(self) -> None:
        token = self._get_token("login")
        resp = self.session.post(
            self.api_url,
            data={
                "action": "login",
                "lgname": self.user,
                "lgpassword": self.password,
                "lgtoken": token,
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        if resp.json().get("login", {}).get("result") != "Success":
            raise RuntimeError(f"MediaWiki login failed: {resp.json()}")

    def edit(self, title: str, text: str, summary: str) -> None:
        token = self._get_token("csrf")
        resp = self.session.post(
            self.api_url,
            data={
                "action": "edit",
                "title": title,
                "text": text,
                "summary": summary,
                "bot": "1",
                "token": token,
                "format": "json",
            },
            timeout=60,
        )
        resp.raise_for_status()
        if resp.json().get("edit", {}).get("result") != "Success":
            raise RuntimeError(f"MediaWiki edit failed for {title}: {resp.json()}")

    def purge(self, titles) -> None:
        """Purge the MediaWiki parser cache for the given page titles so newly
        created items and updated theme tables render fresh. Batched (<=50)."""
        titles = [t for t in titles if t]
        for i in range(0, len(titles), 50):
            chunk = titles[i:i + 50]
            resp = self.session.post(
                self.api_url,
                data={"action": "purge", "titles": "|".join(chunk), "format": "json"},
                timeout=60,
            )
            resp.raise_for_status()

    def page_exists(self, title: str) -> bool:
        resp = self.session.get(
            self.api_url,
            params={"action": "query", "titles": title, "prop": "info", "format": "json"},
            timeout=60,
        )
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        # MediaWiki marks absent titles with a "missing" key (and a negative pageid).
        return not any("missing" in page for page in pages.values())


def make_publisher(config) -> WikiPublisher:
    pub = WikiPublisher(config.mediawiki_api_url, config.mediawiki_bot_user, config.mediawiki_bot_password)
    pub.login()
    return pub
