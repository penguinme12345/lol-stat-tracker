"""Feature extraction from raw match JSON into tabular dataset."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from lol_stat_tracker.config import MANIFEST_PATH, MATCHES_CSV_PATH, RAW_DIR, ensure_directories


def _read_match_files(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _resolve_target_puuid(matches: list[dict[str, Any]]) -> str:
    env_puuid = os.getenv("TRACKER_PUUID")
    if env_puuid:
        return env_puuid

    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest_puuid = manifest.get("target_puuid")
        if manifest_puuid:
            return str(manifest_puuid)

    participant_sets = [set(match.get("metadata", {}).get("participants", [])) for match in matches]
    participant_sets = [participants for participants in participant_sets if participants]
    if participant_sets:
        common_puuids = set.intersection(*participant_sets)
        if len(common_puuids) == 1:
            return next(iter(common_puuids))

    puuids = []
    for match in matches:
        participants = match.get("metadata", {}).get("participants", [])
        puuids.extend(participants)
    if not puuids:
        raise ValueError("Could not infer target puuid. Set TRACKER_PUUID in environment.")
    return Counter(puuids).most_common(1)[0][0]


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _extract_row(match: dict[str, Any], target_puuid: str) -> dict[str, Any] | None:
    info = match.get("info", {})
    metadata = match.get("metadata", {})
    participants = info.get("participants", [])
    target = next((p for p in participants if p.get("puuid") == target_puuid), None)
    if not target:
        return None

    team_id = target.get("teamId")
    team_participants = [p for p in participants if p.get("teamId") == team_id]
    team_kills = sum(int(p.get("kills", 0)) for p in team_participants)
    team_damage = sum(int(p.get("totalDamageDealtToChampions", 0)) for p in team_participants)

    kills = int(target.get("kills", 0))
    deaths = int(target.get("deaths", 0))
    assists = int(target.get("assists", 0))
    duration_seconds = float(info.get("gameDuration", 0))
    duration_min = _safe_div(duration_seconds, 60.0)

    total_minions = int(target.get("totalMinionsKilled", 0)) + int(target.get("neutralMinionsKilled", 0))
    damage = int(target.get("totalDamageDealtToChampions", 0))
    gold = int(target.get("goldEarned", 0))
    vision = int(target.get("visionScore", 0))

    timestamp_ms = info.get("gameEndTimestamp") or info.get("gameCreation")
    return {
        "match_id": metadata.get("matchId", ""),
        "timestamp": int(timestamp_ms) if timestamp_ms else 0,
        "queue_id": int(info.get("queueId", 0)),
        "duration_min": duration_min,
        "champion": target.get("championName", "UNKNOWN"),
        "role": target.get("teamPosition") or target.get("individualPosition") or target.get("lane") or "UNKNOWN",
        "win": int(bool(target.get("win", False))),
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "kda": _safe_div(kills + assists, max(1, deaths)),
        "cs_total": total_minions,
        "cs_per_min": _safe_div(total_minions, duration_min),
        "damage_to_champions": damage,
        "damage_per_min": _safe_div(damage, duration_min),
        "gold_earned": gold,
        "gold_per_min": _safe_div(gold, duration_min),
        "vision_score": vision,
        "vision_per_min": _safe_div(vision, duration_min),
        "kill_participation": _safe_div(kills + assists, team_kills),
        "team_damage_share": _safe_div(damage, team_damage),
    }


def build_dataset() -> str:
    ensure_directories()
    matches = _read_match_files()
    if not matches:
        raise ValueError("No raw matches found. Run ingest first.")

    target_puuid = _resolve_target_puuid(matches)
    rows = [row for row in (_extract_row(m, target_puuid) for m in matches) if row]
    if not rows:
        raise ValueError("No rows extracted for target puuid. Check TRACKER_PUUID.")

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    MATCHES_CSV_PATH.write_text(df.to_csv(index=False), encoding="utf-8")
    return str(MATCHES_CSV_PATH)

