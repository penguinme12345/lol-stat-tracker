"""FastAPI backend for desktop app integration."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lol_stat_tracker.config import METRICS_PATH
from lol_stat_tracker.features import build_dataset
from lol_stat_tracker.ingest import ingest_matches
from lol_stat_tracker.insights import (
    build_last_game_report,
    build_weekly_summary,
    last_game_payload,
    weekly_summary_payload,
)
from lol_stat_tracker.train import train_win_model

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

