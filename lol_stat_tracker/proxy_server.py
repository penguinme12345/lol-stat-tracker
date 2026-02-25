"""Lightweight Riot API proxy for demo distribution."""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from typing import Any
from urllib.parse import quote

import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="LoL Coach Tracker Riot Proxy", version="1.0.0")

_session = requests.Session()
_rate_windows: dict[str, deque[float]] = defaultdict(deque)


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise HTTPException(status_code=500, detail=f"Server missing required env var: {name}")
    return value


def _validate_auth(authorization: str | None) -> None:
    expected = _env("DEMO_ACCESS_TOKEN")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    received = authorization.replace("Bearer ", "", 1).strip()
    if received != expected:
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _enforce_rate_limit(ip: str, limit: int = 120, window_seconds: int = 60) -> None:
    now = time.time()
    window = _rate_windows[ip]
    while window and now - window[0] > window_seconds:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    window.append(now)


def _forward(url: str, params: dict[str, Any] | None = None) -> JSONResponse:
    headers = {"X-Riot-Token": _env("RIOT_API_KEY")}
    response = _session.get(url, params=params, headers=headers, timeout=30)
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        data: Any = response.json()
    else:
        data = {"detail": response.text}
    return JSONResponse(status_code=response.status_code, content=data)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/riot/account/by-riot-id/{region}/{game_name}/{tag_line}")
def account_by_riot_id(
    region: str,
    game_name: str,
    tag_line: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _validate_auth(authorization)
    _enforce_rate_limit(request.client.host if request.client else "unknown")
    url = (
        f"https://{region}.api.riotgames.com/riot/account/v1/accounts/"
        f"by-riot-id/{quote(game_name, safe='')}/{quote(tag_line, safe='')}"
    )
    return _forward(url)


@app.get("/riot/matches/by-puuid/{region}/{puuid}/ids")
def match_ids(
    region: str,
    puuid: str,
    request: Request,
    start: int = 0,
    count: int = 100,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _validate_auth(authorization)
    _enforce_rate_limit(request.client.host if request.client else "unknown")
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{quote(puuid, safe='')}/ids"
    return _forward(url, params={"start": start, "count": count})


@app.get("/riot/matches/{region}/{match_id}")
def match_by_id(
    region: str,
    match_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    _validate_auth(authorization)
    _enforce_rate_limit(request.client.host if request.client else "unknown")
    url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/{quote(match_id, safe='')}"
    return _forward(url)
