"""Feature extraction from raw match JSON into tabular dataset."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from lol_stat_tracker.config import MANIFEST_PATH, MATCHES_CSV_PATH, RAW_DIR, TIMELINE_DIR, ensure_directories
from lol_stat_tracker.timeline import extract_timeline_features


def _read_match_files(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _read_timeline(match_id: str, timeline_dir: Path = TIMELINE_DIR) -> dict[str, Any] | None:
    path = timeline_dir / f"{match_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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


def _participant_lookup(
    participants: list[dict[str, Any]], puuid: str, timeline_payload: dict[str, Any] | None = None
) -> tuple[dict[str, Any] | None, int, int | None, str, str, bool]:
    target = next((p for p in participants if p.get("puuid") == puuid), None)
    if not target:
        return None, 0, None, "unknown", "unknown", False

    participant_id = int(target.get("participantId", 0))
    team_id = int(target.get("teamId", 0))
    role = target.get("teamPosition") or target.get("individualPosition") or ""

    opponent = next(
        (
            p
            for p in participants
            if int(p.get("teamId", 0)) != team_id
            and ((p.get("teamPosition") or p.get("individualPosition") or "") == role)
        ),
        None,
    )
    if opponent:
        opponent_participant_id = int(opponent.get("participantId", 0))
        return target, participant_id, opponent_participant_id, "role_match", "high", True

    if timeline_payload:
        frames = timeline_payload.get("info", {}).get("frames", [])
        frame10 = None
        for frame in frames:
            if int(frame.get("timestamp", 0)) <= 10 * 60_000:
                frame10 = frame
            else:
                break

        if frame10:
            participant_frames = frame10.get("participantFrames", {})
            target_pf = participant_frames.get(str(participant_id), {})
            target_cs = int(target_pf.get("minionsKilled", 0)) + int(target_pf.get("jungleMinionsKilled", 0))

            enemy_candidates = [p for p in participants if int(p.get("teamId", 0)) != team_id]
            if enemy_candidates:
                best = min(
                    enemy_candidates,
                    key=lambda p: abs(
                        (
                            int(
                                participant_frames.get(str(int(p.get("participantId", 0))), {}).get(
                                    "minionsKilled", 0
                                )
                            )
                            + int(
                                participant_frames.get(str(int(p.get("participantId", 0))), {}).get(
                                    "jungleMinionsKilled", 0
                                )
                            )
                        )
                        - target_cs
                    ),
                )
                opponent_participant_id = int(best.get("participantId", 0))
                return target, participant_id, opponent_participant_id, "timeline_lane_proxy", "medium", True

    return target, participant_id, None, "team_average", "low", False


def _extract_row(match: dict[str, Any], target_puuid: str) -> dict[str, Any] | None:
    info = match.get("info", {})
    metadata = match.get("metadata", {})
    participants = info.get("participants", [])
    match_id = metadata.get("matchId", "")
    timeline_payload = _read_timeline(match_id)
    target, participant_id, opponent_participant_id, diff_reference, opponent_quality, opponent_found = _participant_lookup(
        participants, target_puuid, timeline_payload=timeline_payload
    )
    if not target:
        return None

    team_id = target.get("teamId")
    team_participants = [p for p in participants if p.get("teamId") == team_id]
    team_kills = sum(int(p.get("kills", 0)) for p in team_participants)
    team_damage = sum(int(p.get("totalDamageDealtToChampions", 0)) for p in team_participants)
    team_gold = sum(int(p.get("goldEarned", 0)) for p in team_participants)

    objectives = info.get("teams", [])
    team_objectives = next((t.get("objectives", {}) for t in objectives if t.get("teamId") == team_id), {})
    team_dragons = int(team_objectives.get("dragon", {}).get("kills", 0))
    team_barons = int(team_objectives.get("baron", {}).get("kills", 0))
    team_turrets = int(team_objectives.get("tower", {}).get("kills", 0))

    kills = int(target.get("kills", 0))
    deaths = int(target.get("deaths", 0))
    assists = int(target.get("assists", 0))
    duration_seconds = float(info.get("gameDuration", 0))
    duration_min = _safe_div(duration_seconds, 60.0)

    total_minions = int(target.get("totalMinionsKilled", 0)) + int(target.get("neutralMinionsKilled", 0))
    damage = int(target.get("totalDamageDealtToChampions", 0))
    gold = int(target.get("goldEarned", 0))
    vision = int(target.get("visionScore", 0))
    kills_near_enemy_turret = int(target.get("challenges", {}).get("killsNearEnemyTurret", 0))
    solo_kills = int(target.get("challenges", {}).get("soloKills", 0))
    lane_minions_first_10 = float(target.get("challenges", {}).get("laneMinionsFirst10Minutes", 0))
    wards_killed = int(target.get("wardsKilled", 0))
    wards_placed = int(target.get("wardsPlaced", 0))
    control_wards = int(target.get("visionWardsBoughtInGame", 0))
    dragon_kills = int(target.get("dragonKills", 0))
    baron_kills = int(target.get("baronKills", 0))
    turret_kills = int(target.get("turretKills", 0))
    plates = int(target.get("turretPlatesTaken", 0))
    team_plates = sum(int(p.get("turretPlatesTaken", 0)) for p in team_participants)
    double_kills = int(target.get("doubleKills", 0))
    triple_kills = int(target.get("tripleKills", 0))
    quadra_kills = int(target.get("quadraKills", 0))

    if not opponent_found:
        enemy_team = [p for p in participants if int(p.get("teamId", 0)) != int(team_id)]
        opp_gold = sum(int(p.get("goldEarned", 0)) for p in enemy_team) / max(1, len(enemy_team))
        opp_damage = sum(int(p.get("totalDamageDealtToChampions", 0)) for p in enemy_team) / max(1, len(enemy_team))
        opp_cs = sum(int(p.get("totalMinionsKilled", 0)) + int(p.get("neutralMinionsKilled", 0)) for p in enemy_team) / max(
            1, len(enemy_team)
        )
        opp_kills = sum(int(p.get("kills", 0)) for p in enemy_team) / max(1, len(enemy_team))
        damage = int(target.get("totalDamageDealtToChampions", 0))
        gold = int(target.get("goldEarned", 0))
        total_minions = int(target.get("totalMinionsKilled", 0)) + int(target.get("neutralMinionsKilled", 0))
        kills = int(target.get("kills", 0))
        # Fallback lane-relative features vs enemy team average when opponent is ambiguous.
        team_damage = team_damage if team_damage else 1
        team_gold = team_gold if team_gold else 1
    timeline_features = extract_timeline_features(
        timeline=timeline_payload,
        participant_id=participant_id,
        opponent_participant_id=opponent_participant_id,
        target_team_id=int(team_id),
        duration_min=duration_min,
        diff_reference=diff_reference,
        opponent_resolution_quality=opponent_quality,
        opponent_found=opponent_found,
    )

    aggression_index = _safe_div(kills + assists, max(1, deaths)) + 0.3 * solo_kills + 2.0 * _safe_div(damage, team_damage)
    farming_discipline_index = (_safe_div(total_minions, duration_min) + lane_minions_first_10 / 10.0 + _safe_div(gold, duration_min)) / 3.0
    objective_discipline_index = (
        _safe_div(dragon_kills, team_dragons)
        + _safe_div(baron_kills, team_barons)
        + _safe_div(turret_kills, team_turrets)
    ) / 3.0
    clutch_index = float(triple_kills + (2 * quadra_kills)) + float(double_kills * 0.5)

    timestamp_ms = info.get("gameEndTimestamp") or info.get("gameCreation")
    return {
        "match_id": match_id,
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
        "damage_per_gold": _safe_div(damage, gold),
        "damage_share": _safe_div(damage, team_damage),
        "gold_share": _safe_div(gold, team_gold),
        "kill_share": _safe_div(kills, team_kills),
        "solo_kills": solo_kills,
        "kills_near_enemy_turret": kills_near_enemy_turret,
        "multikill_score": float(double_kills + (2 * triple_kills) + (3 * quadra_kills)),
        "vision_control_ratio": _safe_div(wards_killed, max(1, wards_placed)),
        "control_wards_per_game": float(control_wards),
        "dragon_participation_rate": _safe_div(dragon_kills, team_dragons),
        "baron_participation_rate": _safe_div(baron_kills, team_barons),
        "turret_participation_rate": _safe_div(turret_kills, team_turrets),
        "plate_control_pct": _safe_div(plates, team_plates),
        "aggression_index": aggression_index,
        "farming_discipline_index": farming_discipline_index,
        "objective_discipline_index": objective_discipline_index,
        "clutch_index": clutch_index,
        **timeline_features,
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

