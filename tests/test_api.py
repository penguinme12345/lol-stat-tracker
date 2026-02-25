from fastapi.testclient import TestClient

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
            "win_probability_last_game": 0.64,
            "focus_goal": "Keep deaths ≤ 3",
            "top_improvements": ["Keep deaths ≤ 3", "Increase gold/min", "Increase damage/min"],
            "weekly_trend": "Your performance improved 8% this week.",
            "ai_feedback": "Stay disciplined early and avoid unnecessary skirmishes.",
        },
    )

    client = TestClient(app)
    response = client.get("/api/intelligence/report")
    assert response.status_code == 200
    payload = response.json()

    assert set(payload.keys()) == {
        "performance_index",
        "confidence",
        "win_probability_last_game",
        "focus_goal",
        "top_improvements",
        "weekly_trend",
        "ai_feedback",
    }

