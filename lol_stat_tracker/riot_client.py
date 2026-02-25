"""Riot API client with basic retry/backoff."""

from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote

import requests

PLATFORM_ROUTING = {
    "na1": "americas",
    "br1": "americas",
    "la1": "americas",
    "la2": "americas",
    "euw1": "europe",
    "eun1": "europe",
    "tr1": "europe",
    "ru": "europe",
    "kr": "asia",
    "jp1": "asia",
}


class RiotAPIError(RuntimeError):
    """Raised when Riot API returns an unrecoverable response."""


class RiotClient:
    def __init__(self, api_key: str | None = None, region: str = "americas", timeout: int = 30) -> None:
        self.api_key = api_key
        self.region = region
        self.timeout = timeout
        self.proxy_base_url = os.getenv("RIOT_PROXY_URL", "").strip().rstrip("/")
        self.proxy_access_token = os.getenv("RIOT_PROXY_ACCESS_TOKEN", "").strip()
        self.use_proxy = bool(self.proxy_base_url)
        self.session = requests.Session()
        if self.use_proxy:
            if not self.proxy_access_token:
                raise RiotAPIError("RIOT_PROXY_ACCESS_TOKEN is required when RIOT_PROXY_URL is set.")
            self.session.headers.update({"Authorization": f"Bearer {self.proxy_access_token}"})
        elif self.api_key:
            self.session.headers.update({"X-Riot-Token": self.api_key})

    def _request(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(5):
            response = self.session.get(url, params=params, timeout=self.timeout)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "1"))
                time.sleep(max(1, retry_after))
                continue
            if 500 <= response.status_code < 600:
                time.sleep(2**attempt)
                continue
            if response.ok:
                return response.json()
            raise RiotAPIError(f"Riot API error {response.status_code}: {response.text}")
        raise RiotAPIError(f"Riot API retries exhausted for URL: {url}")

    def _proxy_url(self, path: str) -> str:
        if not self.use_proxy:
            raise RiotAPIError("Proxy URL requested but RIOT_PROXY_URL is not configured.")
        return f"{self.proxy_base_url}{path}"

    def get_puuid(self, game_name: str, tag_line: str) -> str:
        if self.use_proxy:
            url = self._proxy_url(
                f"/riot/account/by-riot-id/{self.region}/{quote(game_name, safe='')}/{quote(tag_line, safe='')}"
            )
        else:
            url = (
                f"https://{self.region}.api.riotgames.com/riot/account/v1/accounts/"
                f"by-riot-id/{game_name}/{tag_line}"
            )
        payload = self._request(url)
        puuid = payload.get("puuid")
        if not puuid:
            raise RiotAPIError("Missing puuid in account response.")
        return puuid

    def get_match_ids(self, puuid: str, count: int = 100, start: int = 0) -> list[str]:
        if self.use_proxy:
            url = self._proxy_url(f"/riot/matches/by-puuid/{self.region}/{quote(puuid, safe='')}/ids")
        else:
            url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        payload = self._request(url, params={"start": start, "count": count})
        if not isinstance(payload, list):
            raise RiotAPIError("Unexpected match id response format.")
        return payload

    def get_match(self, match_id: str) -> dict[str, Any]:
        if self.use_proxy:
            url = self._proxy_url(f"/riot/matches/{self.region}/{quote(match_id, safe='')}")
        else:
            url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        payload = self._request(url)
        if not isinstance(payload, dict):
            raise RiotAPIError(f"Unexpected match response format for {match_id}.")
        return payload

    def get_match_timeline(self, match_id: str) -> dict[str, Any]:
        if self.use_proxy:
            url = self._proxy_url(f"/riot/matches/{self.region}/{quote(match_id, safe='')}/timeline")
        else:
            url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        payload = self._request(url)
        if not isinstance(payload, dict):
            raise RiotAPIError(f"Unexpected timeline response format for {match_id}.")
        return payload

