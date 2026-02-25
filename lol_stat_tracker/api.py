"""FastAPI backend for desktop app integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lol_stat_tracker.config import MANIFEST_PATH, METRICS_PATH, RAW_DIR, TIMELINE_DIR
from lol_stat_tracker.features import build_dataset
from lol_stat_tracker.ingest import ingest_matches
from lol_stat_tracker.insights import (
    build_last_game_report,
    build_weekly_summary,
    intelligence_deep_report_payload,
    intelligence_report_payload,
    last_game_payload,
    weekly_summary_payload,
)
from lol_stat_tracker.train import train_win_model
from lol_stat_tracker.timeline import timeline_summary_for_match

app = FastAPI(title="LoL Coach Tracker API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SyncRequest(BaseModel):
    game_name: str = Field(min_length=1)
    tag_line: str = Field(min_length=1)
    region: str = Field(default="americas", pattern="^(americas|europe|asia)$")
    api_key: str | None = Field(default=None)
    count: int = Field(default=100, ge=1, le=500)


class BuildResponse(BaseModel):
    dataset_path: str


class TrainResponse(BaseModel):
    metrics_path: str
    metrics: dict[str, Any]


class ReportResponse(BaseModel):
    report_path: str
    data: dict[str, Any]


class IntelligenceResponse(BaseModel):
    performance_index: int
    confidence: str
    early_outlook: float
    post_game_model_score: float
    context_bucket: str
    context_sample_size: int
    focus_goal: str
    top_improvements: list[str]
    weekly_trend: str
    ai_feedback: str
    overall_form: dict[str, Any]
    primary_archetype: str
    secondary_archetypes: list[str]
    player_tags: list[dict[str, Any]]
    tier_ratings: dict[str, dict[str, Any]]
    behavioral_dimensions: dict[str, int]
    contribution_pct: list[dict[str, Any]]
    counterfactual_deltas: list[dict[str, Any]]
    niche_improvements: list[str]
    context_warning: str | None
    model_system: dict[str, Any]
    quests: list[dict[str, Any]]
    quest_progress: dict[str, int]
    momentum: dict[str, str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sync")
def sync_matches(payload: SyncRequest) -> dict[str, Any]:
    try:
        saved = ingest_matches(
            game_name=payload.game_name,
            tag_line=payload.tag_line,
            region=payload.region,
            api_key=payload.api_key,
            count=payload.count,
        )
        return {"new_matches_downloaded": saved}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/build", response_model=BuildResponse)
def build() -> BuildResponse:
    try:
        return BuildResponse(dataset_path=build_dataset())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    try:
        metrics_path = train_win_model()
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return TrainResponse(metrics_path=metrics_path, metrics=metrics)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/last-game", response_model=ReportResponse)
def last_game() -> ReportResponse:
    try:
        report_path = build_last_game_report()
        return ReportResponse(report_path=report_path, data=last_game_payload())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/weekly", response_model=ReportResponse)
def weekly() -> ReportResponse:
    try:
        report_path = build_weekly_summary()
        return ReportResponse(report_path=report_path, data=weekly_summary_payload())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    try:
        if not METRICS_PATH.exists():
            raise ValueError("Metrics file not found. Run /train first.")
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/intelligence/report", response_model=IntelligenceResponse)
def intelligence_report() -> IntelligenceResponse:
    try:
        payload = intelligence_report_payload()
        return IntelligenceResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/intelligence/deep-report")
def intelligence_deep_report() -> dict[str, Any]:
    try:
        return intelligence_deep_report_payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _load_match_and_timeline(match_id: str) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    raw_path = RAW_DIR / f"{match_id}.json"
    if not raw_path.exists():
        raise HTTPException(status_code=404, detail=f"Match not found in cache: {match_id}")

    timeline_path = TIMELINE_DIR / f"{match_id}.json"
    match_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    timeline_payload = json.loads(timeline_path.read_text(encoding="utf-8")) if timeline_path.exists() else None

    target_puuid = ""
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        target_puuid = str(manifest.get("target_puuid", ""))
    if not target_puuid:
        participants = match_payload.get("metadata", {}).get("participants", [])
        if participants:
            target_puuid = str(participants[0])
    return match_payload, timeline_payload, target_puuid


@app.get("/api/matches/{match_id}/timeline/raw")
def timeline_raw(match_id: str) -> dict[str, Any]:
    _, timeline_payload, _ = _load_match_and_timeline(match_id)
    if timeline_payload is None:
        return {"timeline_missing": True, "data": {}}
    return {"timeline_missing": False, "data": timeline_payload}


@app.get("/api/matches/{match_id}/timeline/features")
def timeline_features(match_id: str) -> dict[str, Any]:
    match_payload, timeline_payload, target_puuid = _load_match_and_timeline(match_id)
    summary = timeline_summary_for_match(match_payload, timeline_payload, target_puuid)
    return {
        "timeline_missing": summary["timeline_missing"],
        "participant_id": summary["participant_id"],
        "opponent_participant_id": summary["opponent_participant_id"],
        "team_id": summary["team_id"],
        "diff_reference": summary["diff_reference"],
        "opponent_resolution_quality": summary["opponent_resolution_quality"],
        "features": summary["features"],
    }


@app.get("/api/matches/{match_id}/timeline/summary")
def timeline_summary(match_id: str) -> dict[str, Any]:
    match_payload, timeline_payload, target_puuid = _load_match_and_timeline(match_id)
    summary = timeline_summary_for_match(match_payload, timeline_payload, target_puuid)
    return {
        "match_id": match_id,
        "timeline_missing": summary["timeline_missing"],
        "timeline_warning": summary["timeline_warning"],
        "participant_id": summary["participant_id"],
        "opponent_participant_id": summary["opponent_participant_id"],
        "team_id": summary["team_id"],
        "diff_reference": summary["diff_reference"],
        "opponent_resolution_quality": summary["opponent_resolution_quality"],
        "highlights": {
            "gold_diff_10": summary["features"].get("gold_diff_10", 0.0),
            "xp_diff_10": summary["features"].get("xp_diff_10", 0.0),
            "cs_diff_10": summary["features"].get("cs_diff_10", 0.0),
            "ahead_at_10": summary["features"].get("ahead_at_10", 0),
            "ahead_at_15": summary["features"].get("ahead_at_15", 0),
            "deaths_before_dragon_60s": summary["features"].get("deaths_before_dragon_60s", 0),
            "deaths_before_baron_60s": summary["features"].get("deaths_before_baron_60s", 0),
            "time_to_first_tower_min": summary["features"].get("time_to_first_tower_min", 0.0),
        },
    }

