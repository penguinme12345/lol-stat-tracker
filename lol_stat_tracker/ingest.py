"""Data ingestion from Riot API into local raw JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lol_stat_tracker.config import MANIFEST_PATH, RAW_DIR, ensure_directories, get_api_key
from lol_stat_tracker.riot_client import RiotClient


def _load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"match_ids": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ingest_matches(
    game_name: str,
    tag_line: str,
    region: str = "americas",
    api_key: str | None = None,
    count: int = 100,
) -> int:
    ensure_directories()
    client = RiotClient(api_key=get_api_key(api_key), region=region)
    manifest = _load_manifest()
    existing = set(manifest.get("match_ids", []))

    puuid = client.get_puuid(game_name=game_name, tag_line=tag_line)
    match_ids = client.get_match_ids(puuid=puuid, count=count)

    new_saved = 0
    for match_id in match_ids:
        if match_id in existing:
            continue
        payload = client.get_match(match_id)
        (RAW_DIR / f"{match_id}.json").write_text(json.dumps(payload), encoding="utf-8")
        existing.add(match_id)
        new_saved += 1

    manifest["match_ids"] = sorted(existing)
    manifest["target_puuid"] = puuid
    _save_manifest(manifest)
    return new_saved

