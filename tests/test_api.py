from fastapi.testclient import TestClient
from pathlib import Path
import json

import lol_stat_tracker.api as api_module
from lol_stat_tracker.api import app


def test_health() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_intelligence_report_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "intelligence_report_payload",
        lambda: {
            "performance_index": 72,
            "confidence": "High",
            "early_outlook": 0.58,
            "post_game_model_score": 0.64,
            "context_bucket": "champion_role",
            "context_sample_size": 22,
            "focus_goal": "Keep deaths ≤ 3",
            "top_improvements": ["Keep deaths ≤ 3", "Increase gold/min", "Increase damage/min"],
            "weekly_trend": "Your performance improved 8% this week.",
            "ai_feedback": "Stay disciplined early and avoid unnecessary skirmishes.",
            "overall_form": {"score": 72, "tier": "Gold"},
            "primary_archetype": "Clutch Fighter",
            "secondary_archetypes": ["Solo Carry", "Strategic Closer", "Objective General"],
            "player_tags": [
                {"name": "🔥 Aggro Starter", "emoji": "🔥", "reason": "High early KP"},
                {"name": "🪙 Economy Builder", "emoji": "🪙", "reason": "Strong gold pace"},
            ],
            "tier_ratings": {
                "early_game": {"tier": "Gold", "summary": "Strong opening tempo"},
                "mid_game": {"tier": "Silver", "summary": "Needs cleaner rotations"},
                "late_game": {"tier": "Bronze", "summary": "Decision quality drops"},
                "objective_control": {"tier": "Silver", "summary": "Average objective pressure"},
                "fighting_impact": {"tier": "Gold", "summary": "Good teamfight impact"},
                "consistency": {"tier": "Silver", "summary": "Performance fluctuates"},
            },
            "quests": [
                {
                    "title": "Low-Death Discipline",
                    "description": "Keep deaths low for better conversion",
                    "target": 8,
                    "current": 5,
                    "unit": "games",
                    "completed": False,
                }
            ],
            "behavioral_dimensions": {
                "early_pressure": 61,
                "lead_stability": 52,
                "late_game_control": 44,
                "objective_fight_impact": 58,
                "combat_efficiency": 63,
                "consistency": 46,
                "map_presence": 55,
                "snowball_efficiency": 49,
                "risk_index": 42,
                "clutch_reliability": 60,
            },
            "contribution_pct": [
                {"feature": "gold_diff_10", "impact_pct": 8.2, "direction": "positive"},
            ],
            "counterfactual_deltas": [
                {
                    "feature": "deaths",
                    "from": 6.0,
                    "to": 3.0,
                    "win_rate_delta_pct": 10.4,
                    "bounded_by_context": True,
                }
            ],
            "niche_improvements": ["Maintain death ≤3 in 3 consecutive games"],
            "context_warning": None,
            "model_system": {"primary_model": "lightgbm", "rf_baseline_agreement": 0.71},
            "quest_progress": {"completed": 0, "total": 1},
            "momentum": {"label": "🔥 Hot Streak", "reason": "Recent results trending up"},
        },
    )

    client = TestClient(app)
    response = client.get("/api/intelligence/report")
    assert response.status_code == 200
    payload = response.json()

    assert set(payload.keys()) == {
        "performance_index",
        "confidence",
        "early_outlook",
        "post_game_model_score",
        "context_bucket",
        "context_sample_size",
        "focus_goal",
        "top_improvements",
        "weekly_trend",
        "ai_feedback",
        "overall_form",
        "primary_archetype",
        "secondary_archetypes",
        "player_tags",
        "tier_ratings",
        "behavioral_dimensions",
        "contribution_pct",
        "counterfactual_deltas",
        "niche_improvements",
        "context_warning",
        "model_system",
        "quests",
        "quest_progress",
        "momentum",
    }


def test_intelligence_deep_report_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "intelligence_deep_report_payload",
        lambda: {
            "dataset": {"num_matches": 100, "win_rate": 0.52, "champions_used": 8, "roles_played": 2},
            "timeline_snapshot": {"gold_diff_10_avg": 120.0},
            "win_state": {"lead_conversion_rate": 0.55},
            "playstyle_indices": {"aggression_index_avg": 2.4},
            "model": {"confidence": "Moderate", "win_probability_last_game": 0.61, "top_model_links": []},
            "pattern_links": {"positive": [], "negative": []},
            "context_performance": [],
            "generative_breakdown": {
                "executive_summary": "summary",
                "inferences": ["inference"],
                "predictions": ["prediction"],
                "hidden_signals": {"gold_diff_10_to_15_avg": 12.0},
                "data_coverage": {"matches_used": 100, "recent_window": 15, "model_auc": 0.7, "rf_agreement": 0.68},
            },
        },
    )

    client = TestClient(app)
    response = client.get("/api/intelligence/deep-report")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {
        "dataset",
        "timeline_snapshot",
        "win_state",
        "playstyle_indices",
        "model",
        "pattern_links",
        "context_performance",
        "generative_breakdown",
    }


def test_timeline_summary_endpoint(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "matches"
    timeline_dir = tmp_path / "timelines"
    raw_dir.mkdir(parents=True)
    timeline_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.json"

    match_id = "NA1_TEST_1"
    target_puuid = "target-puuid"
    match_payload = {
        "metadata": {"matchId": match_id, "participants": [target_puuid, "enemy"]},
        "info": {
            "gameDuration": 1800,
            "participants": [
                {
                    "puuid": target_puuid,
                    "participantId": 1,
                    "teamId": 100,
                    "teamPosition": "MIDDLE",
                    "individualPosition": "MIDDLE",
                },
                {
                    "puuid": "enemy",
                    "participantId": 6,
                    "teamId": 200,
                    "teamPosition": "MIDDLE",
                    "individualPosition": "MIDDLE",
                },
            ],
        },
    }
    timeline_payload = {
        "info": {
            "frames": [
                {
                    "timestamp": 600000,
                    "participantFrames": {
                        "1": {"totalGold": 3500, "xp": 4200, "minionsKilled": 80, "jungleMinionsKilled": 5, "level": 9},
                        "6": {"totalGold": 3000, "xp": 3800, "minionsKilled": 70, "jungleMinionsKilled": 3, "level": 8},
                    },
                    "events": [],
                }
            ]
        }
    }

    (raw_dir / f"{match_id}.json").write_text(json.dumps(match_payload), encoding="utf-8")
    (timeline_dir / f"{match_id}.json").write_text(json.dumps(timeline_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps({"target_puuid": target_puuid}), encoding="utf-8")

    monkeypatch.setattr(api_module, "RAW_DIR", raw_dir)
    monkeypatch.setattr(api_module, "TIMELINE_DIR", timeline_dir)
    monkeypatch.setattr(api_module, "MANIFEST_PATH", manifest_path)

    client = TestClient(app)
    response = client.get(f"/api/matches/{match_id}/timeline/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["match_id"] == match_id
    assert "highlights" in payload

