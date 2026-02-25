from pathlib import Path

import lol_stat_tracker.features as features
from lol_stat_tracker.features import _extract_row, _resolve_target_puuid


def test_extract_row_basic_metrics() -> None:
    target_puuid = "target-puuid"
    payload = {
        "metadata": {"matchId": "NA1_123", "participants": [target_puuid, "other"]},
        "info": {
            "queueId": 420,
            "gameDuration": 1800,
            "gameEndTimestamp": 1_700_000_000_000,
            "participants": [
                {
                    "puuid": target_puuid,
                    "teamId": 100,
                    "championName": "Ahri",
                    "teamPosition": "MIDDLE",
                    "win": True,
                    "kills": 10,
                    "deaths": 2,
                    "assists": 5,
                    "totalMinionsKilled": 180,
                    "neutralMinionsKilled": 20,
                    "totalDamageDealtToChampions": 25000,
                    "goldEarned": 14500,
                    "visionScore": 22,
                },
                {
                    "puuid": "other",
                    "teamId": 100,
                    "kills": 20,
                    "deaths": 8,
                    "assists": 15,
                    "totalDamageDealtToChampions": 50000,
                },
            ],
        },
    }

    row = _extract_row(payload, target_puuid=target_puuid)
    assert row is not None
    assert row["match_id"] == "NA1_123"
    assert row["win"] == 1
    assert row["cs_total"] == 200
    assert round(row["cs_per_min"], 2) == 6.67
    assert round(row["damage_per_min"], 2) == 833.33
    assert round(row["kill_participation"], 2) == 0.50


def test_resolve_target_puuid_prefers_common_intersection(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TRACKER_PUUID", raising=False)
    fake_manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(features, "MANIFEST_PATH", fake_manifest)

    target = "target-puuid"
    matches = [
        {"metadata": {"participants": [target, "a", "b"]}},
        {"metadata": {"participants": [target, "a", "c"]}},
        {"metadata": {"participants": [target, "d", "e"]}},
    ]

    assert _resolve_target_puuid(matches) == target

