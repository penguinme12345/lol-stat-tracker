"""Timeline feature extraction helpers."""

from __future__ import annotations

from typing import Any


def _safe_div(num: float, denom: float) -> float:
    return float(num) / float(denom) if denom else 0.0


def _frame_at_minute(frames: list[dict[str, Any]], minute: int) -> dict[str, Any] | None:
    target_ms = minute * 60_000
    frame = None
    for item in frames:
        if int(item.get("timestamp", 0)) <= target_ms:
            frame = item
        else:
            break
    return frame


def _participant_frame(frame: dict[str, Any] | None, participant_id: int) -> dict[str, Any]:
    if not frame:
        return {}
    participant_frames = frame.get("participantFrames", {})
    return participant_frames.get(str(participant_id), {}) or participant_frames.get(participant_id, {}) or {}


def _enemy_team_average_frame(frame: dict[str, Any] | None, target_team_id: int) -> dict[str, float]:
    if not frame:
        return {"totalGold": 0.0, "xp": 0.0, "level": 0.0, "cs": 0.0}
    participant_frames = frame.get("participantFrames", {})
    enemy_ids = range(6, 11) if int(target_team_id) == 100 else range(1, 6)
    rows = []
    for pid in enemy_ids:
        pf = participant_frames.get(str(pid), {})
        if not pf:
            continue
        rows.append(
            {
                "totalGold": float(pf.get("totalGold", 0)),
                "xp": float(pf.get("xp", 0)),
                "level": float(pf.get("level", 0)),
                "cs": float(int(pf.get("minionsKilled", 0)) + int(pf.get("jungleMinionsKilled", 0))),
            }
        )
    if not rows:
        return {"totalGold": 0.0, "xp": 0.0, "level": 0.0, "cs": 0.0}
    denom = float(len(rows))
    return {
        "totalGold": sum(r["totalGold"] for r in rows) / denom,
        "xp": sum(r["xp"] for r in rows) / denom,
        "level": sum(r["level"] for r in rows) / denom,
        "cs": sum(r["cs"] for r in rows) / denom,
    }


def _sum_cs(frame_data: dict[str, Any]) -> int:
    return int(frame_data.get("minionsKilled", 0)) + int(frame_data.get("jungleMinionsKilled", 0))


def _event_timestamps(
    timeline: dict[str, Any], participant_id: int, target_team_id: int
) -> tuple[list[int], list[int], list[int], list[int]]:
    frames = timeline.get("info", {}).get("frames", [])
    death_times: list[int] = []
    dragon_times: list[int] = []
    baron_times: list[int] = []
    team_tower_times: list[int] = []

    for frame in frames:
        for event in frame.get("events", []):
            ts = int(event.get("timestamp", 0))
            event_type = event.get("type", "")
            if event_type == "CHAMPION_KILL" and int(event.get("victimId", 0)) == participant_id:
                death_times.append(ts)
            if event_type == "ELITE_MONSTER_KILL":
                monster = str(event.get("monsterType", "")).upper()
                if monster == "DRAGON":
                    dragon_times.append(ts)
                if monster == "BARON_NASHOR":
                    baron_times.append(ts)
            if event_type == "BUILDING_KILL" and str(event.get("buildingType", "")).upper() == "TOWER_BUILDING":
                if int(event.get("teamId", 0)) == target_team_id:
                    team_tower_times.append(ts)

    return death_times, dragon_times, baron_times, team_tower_times


def extract_timeline_features(
    timeline: dict[str, Any] | None,
    participant_id: int,
    opponent_participant_id: int | None,
    target_team_id: int,
    duration_min: float,
    diff_reference: str = "unknown",
    opponent_resolution_quality: str = "unknown",
    opponent_found: bool = False,
) -> dict[str, float | int]:
    defaults: dict[str, float | int] = {
        "timeline_missing": 1,
        "participant_id_used": int(participant_id),
        "opponent_participant_id_used": int(opponent_participant_id or 0),
        "team_id_used": int(target_team_id),
        "opponent_found": int(opponent_found),
        "diff_reference": diff_reference,
        "opponent_resolution_quality": opponent_resolution_quality,
        "gold_5": 0.0,
        "xp_5": 0.0,
        "cs_5": 0.0,
        "gold_diff_5": 0.0,
        "xp_diff_5": 0.0,
        "cs_diff_5": 0.0,
        "gold_10": 0.0,
        "xp_10": 0.0,
        "cs_10": 0.0,
        "gold_15": 0.0,
        "xp_15": 0.0,
        "cs_15": 0.0,
        "gold_diff_10": 0.0,
        "xp_diff_10": 0.0,
        "cs_diff_10": 0.0,
        "gold_diff_15": 0.0,
        "xp_diff_15": 0.0,
        "cs_diff_15": 0.0,
        "level_diff_10": 0.0,
        "level_diff_15": 0.0,
        "gold_diff_5_to_10": 0.0,
        "gold_diff_10_to_15": 0.0,
        "xp_diff_5_to_10": 0.0,
        "xp_diff_10_to_15": 0.0,
        "cs_diff_5_to_10": 0.0,
        "cs_diff_10_to_15": 0.0,
        "ahead_at_10": 0,
        "ahead_at_15": 0,
        "behind_at_10": 0,
        "behind_at_15": 0,
        "first_blood_participation": 0,
        "first_death_time_min": duration_min,
        "deaths_before_15": 0,
        "deaths_before_dragon_60s": 0,
        "deaths_before_herald_60s": 0,
        "deaths_before_baron_60s": 0,
        "late_game_deaths_post20": 0,
        "death_rate_after_15": 0.0,
        "time_to_first_tower_min": duration_min,
        "timeline_warning": "timeline_missing",
    }
    if not timeline:
        return defaults

    frames = timeline.get("info", {}).get("frames", [])
    if not frames:
        return defaults

    frame5 = _frame_at_minute(frames, 5)
    frame10 = _frame_at_minute(frames, 10)
    frame15 = _frame_at_minute(frames, 15)
    p5 = _participant_frame(frame5, participant_id)
    p10 = _participant_frame(frame10, participant_id)
    p15 = _participant_frame(frame15, participant_id)
    opp5 = _participant_frame(frame5, opponent_participant_id) if opponent_participant_id else {}
    opp10 = _participant_frame(frame10, opponent_participant_id) if opponent_participant_id else {}
    opp15 = _participant_frame(frame15, opponent_participant_id) if opponent_participant_id else {}
    if not opponent_participant_id:
        opp5_avg = _enemy_team_average_frame(frame5, target_team_id)
        opp10_avg = _enemy_team_average_frame(frame10, target_team_id)
        opp15_avg = _enemy_team_average_frame(frame15, target_team_id)
        opp_gold5 = float(opp5_avg["totalGold"])
        opp_xp5 = float(opp5_avg["xp"])
        opp_cs5 = float(opp5_avg["cs"])
        opp_level5 = float(opp5_avg["level"])
        opp_gold10 = float(opp10_avg["totalGold"])
        opp_xp10 = float(opp10_avg["xp"])
        opp_cs10 = float(opp10_avg["cs"])
        opp_level10 = float(opp10_avg["level"])
        opp_gold15 = float(opp15_avg["totalGold"])
        opp_xp15 = float(opp15_avg["xp"])
        opp_cs15 = float(opp15_avg["cs"])
        opp_level15 = float(opp15_avg["level"])
    else:
        opp_level5 = float(opp5.get("level", 0))
        opp_gold5 = float(opp5.get("totalGold", 0))
        opp_xp5 = float(opp5.get("xp", 0))
        opp_cs5 = float(_sum_cs(opp5))
        opp_gold10 = float(opp10.get("totalGold", 0))
        opp_xp10 = float(opp10.get("xp", 0))
        opp_cs10 = float(_sum_cs(opp10))
        opp_level10 = float(opp10.get("level", 0))
        opp_gold15 = float(opp15.get("totalGold", 0))
        opp_xp15 = float(opp15.get("xp", 0))
        opp_cs15 = float(_sum_cs(opp15))
        opp_level15 = float(opp15.get("level", 0))

    gold5 = float(p5.get("totalGold", 0))
    xp5 = float(p5.get("xp", 0))
    cs5 = float(_sum_cs(p5))
    gold10 = float(p10.get("totalGold", 0))
    xp10 = float(p10.get("xp", 0))
    cs10 = float(_sum_cs(p10))
    level10 = float(p10.get("level", 0))
    gold15 = float(p15.get("totalGold", 0))
    xp15 = float(p15.get("xp", 0))
    cs15 = float(_sum_cs(p15))
    level15 = float(p15.get("level", 0))

    death_times, dragon_times, baron_times, team_tower_times = _event_timestamps(
        timeline=timeline, participant_id=participant_id, target_team_id=target_team_id
    )

    herald_times: list[int] = []
    first_blood_participation = 0
    first_death_time_min = duration_min
    first_blood_seen = False
    for frame in frames:
        for event in frame.get("events", []):
            event_type = event.get("type", "")
            ts = int(event.get("timestamp", 0))
            if event_type == "ELITE_MONSTER_KILL" and str(event.get("monsterType", "")).upper() == "RIFTHERALD":
                herald_times.append(ts)
            if event_type == "CHAMPION_KILL":
                victim = int(event.get("victimId", 0))
                killer = int(event.get("killerId", 0))
                assists = [int(a) for a in event.get("assistingParticipantIds", [])]
                if victim == participant_id and first_death_time_min == duration_min:
                    first_death_time_min = ts / 60_000.0
                if not first_blood_seen and event.get("killType") == "KILL_FIRST_BLOOD":
                    first_blood_seen = True
                    if participant_id == killer or participant_id in assists:
                        first_blood_participation = 1

    def _deaths_before(objective_times: list[int]) -> int:
        total = 0
        for objective_ts in objective_times:
            if any(0 <= objective_ts - death_ts <= 60_000 for death_ts in death_times):
                total += 1
        return total

    late_deaths = sum(1 for ts in death_times if ts >= 20 * 60_000)
    deaths_before_15 = sum(1 for ts in death_times if ts < 15 * 60_000)
    deaths_after_15 = sum(1 for ts in death_times if ts >= 15 * 60_000)
    post_15_minutes = max(duration_min - 15.0, 1.0)

    time_to_first_tower_min = duration_min
    if team_tower_times:
        time_to_first_tower_min = min(team_tower_times) / 60_000.0

    defaults.update(
        {
            "timeline_missing": 0,
            "participant_id_used": int(participant_id),
            "opponent_participant_id_used": int(opponent_participant_id or 0),
            "team_id_used": int(target_team_id),
            "opponent_found": int(opponent_found),
            "diff_reference": diff_reference,
            "opponent_resolution_quality": opponent_resolution_quality,
            "gold_5": gold5,
            "xp_5": xp5,
            "cs_5": cs5,
            "gold_diff_5": gold5 - opp_gold5,
            "xp_diff_5": xp5 - opp_xp5,
            "cs_diff_5": cs5 - opp_cs5,
            "gold_10": gold10,
            "xp_10": xp10,
            "cs_10": cs10,
            "gold_15": gold15,
            "xp_15": xp15,
            "cs_15": cs15,
            "gold_diff_10": gold10 - opp_gold10,
            "xp_diff_10": xp10 - opp_xp10,
            "cs_diff_10": cs10 - opp_cs10,
            "gold_diff_15": gold15 - opp_gold15,
            "xp_diff_15": xp15 - opp_xp15,
            "cs_diff_15": cs15 - opp_cs15,
            "level_diff_10": level10 - opp_level10,
            "level_diff_15": level15 - opp_level15,
            "gold_diff_5_to_10": (gold10 - opp_gold10) - (gold5 - opp_gold5),
            "gold_diff_10_to_15": (gold15 - opp_gold15) - (gold10 - opp_gold10),
            "xp_diff_5_to_10": (xp10 - opp_xp10) - (xp5 - opp_xp5),
            "xp_diff_10_to_15": (xp15 - opp_xp15) - (xp10 - opp_xp10),
            "cs_diff_5_to_10": (cs10 - opp_cs10) - (cs5 - opp_cs5),
            "cs_diff_10_to_15": (cs15 - opp_cs15) - (cs10 - opp_cs10),
            "ahead_at_10": int(gold10 - opp_gold10 > 0),
            "ahead_at_15": int(gold15 - opp_gold15 > 0),
            "behind_at_10": int(gold10 - opp_gold10 < 0),
            "behind_at_15": int(gold15 - opp_gold15 < 0),
            "first_blood_participation": first_blood_participation,
            "first_death_time_min": first_death_time_min,
            "deaths_before_15": deaths_before_15,
            "deaths_before_dragon_60s": _deaths_before(dragon_times),
            "deaths_before_herald_60s": _deaths_before(herald_times),
            "deaths_before_baron_60s": _deaths_before(baron_times),
            "late_game_deaths_post20": late_deaths,
            "death_rate_after_15": _safe_div(deaths_after_15, post_15_minutes),
            "time_to_first_tower_min": time_to_first_tower_min,
            "timeline_warning": "ok",
        }
    )

    if not opponent_found:
        defaults["timeline_warning"] = "opponent_inference_low_confidence"
    return defaults


def timeline_summary_for_match(
    match_payload: dict[str, Any], timeline_payload: dict[str, Any] | None, target_puuid: str
) -> dict[str, Any]:
    info = match_payload.get("info", {})
    participants = info.get("participants", [])
    target = next((p for p in participants if p.get("puuid") == target_puuid), None)
    if not target:
        return {
            "timeline_missing": timeline_payload is None,
            "timeline_warning": "target_participant_missing",
            "participant_id": 0,
            "opponent_participant_id": 0,
            "team_id": 0,
            "features": {},
        }

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
    opponent_id = int(opponent.get("participantId", 0)) if opponent else None
    diff_reference = "role_match" if opponent_id else "team_average"
    quality = "high" if opponent_id else "low"

    features = extract_timeline_features(
        timeline=timeline_payload,
        participant_id=participant_id,
        opponent_participant_id=opponent_id,
        target_team_id=team_id,
        duration_min=float(info.get("gameDuration", 0)) / 60.0,
        diff_reference=diff_reference,
        opponent_resolution_quality=quality,
        opponent_found=bool(opponent_id),
    )
    return {
        "timeline_missing": timeline_payload is None,
        "timeline_warning": features.get("timeline_warning", "ok"),
        "participant_id": participant_id,
        "opponent_participant_id": int(opponent_id or 0),
        "team_id": team_id,
        "diff_reference": features.get("diff_reference", "unknown"),
        "opponent_resolution_quality": features.get("opponent_resolution_quality", "unknown"),
        "features": features,
    }
