"""Riot API client with basic retry/backoff."""

from __future__ import annotations

import time
from typing import Any

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
    def __init__(self, api_key: str, region: str = "americas", timeout: int = 30) -> None:
        self.api_key = api_key
        self.region = region
        self.timeout = timeout
        self.session = requests.Session()
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

    def get_puuid(self, game_name: str, tag_line: str) -> str:
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
        url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        payload = self._request(url, params={"start": start, "count": count})
        if not isinstance(payload, list):
            raise RiotAPIError("Unexpected match id response format.")
        return payload

    def get_match(self, match_id: str) -> dict[str, Any]:
        url = f"https://{self.region}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        payload = self._request(url)
        if not isinstance(payload, dict):
            raise RiotAPIError(f"Unexpected match response format for {match_id}.")
        return payload

