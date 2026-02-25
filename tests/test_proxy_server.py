from fastapi.testclient import TestClient

from lol_stat_tracker.proxy_server import app


def test_proxy_rejects_missing_token(monkeypatch) -> None:
    monkeypatch.setenv("DEMO_ACCESS_TOKEN", "demo-secret")
    client = TestClient(app)
    response = client.get("/riot/matches/americas/NA1_123")
    assert response.status_code == 401
