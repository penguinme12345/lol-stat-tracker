"""Shared configuration and paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "matches"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MANIFEST_PATH = DATA_DIR / "raw" / "manifest.json"
MATCHES_CSV_PATH = PROCESSED_DIR / "matches.csv"
MODEL_PATH = MODELS_DIR / "win_rf.pkl"
METRICS_PATH = MODELS_DIR / "win_rf_metrics.json"
LAST_GAME_REPORT_PATH = REPORTS_DIR / "last_game_report.md"
WEEKLY_REPORT_PATH = REPORTS_DIR / "weekly_summary.md"


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DATA_DIR / "raw"]:
        path.mkdir(parents=True, exist_ok=True)


def get_api_key(cli_key: str | None = None) -> str:
    api_key = cli_key or os.getenv("RIOT_API_KEY")
    if not api_key:
        raise ValueError("Missing Riot API key. Set RIOT_API_KEY or pass --api-key.")
    return api_key

