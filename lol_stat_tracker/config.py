"""Shared configuration and paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(os.getenv("TRACKER_BASE_DIR", str(DEFAULT_BASE_DIR))).resolve()
ENV_PATH = Path(os.getenv("TRACKER_ENV_PATH", str(BASE_DIR / ".env"))).resolve()
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "matches"
TIMELINE_DIR = DATA_DIR / "raw" / "timelines"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

MANIFEST_PATH = DATA_DIR / "raw" / "manifest.json"
MATCHES_CSV_PATH = PROCESSED_DIR / "matches.csv"
MODEL_PATH = MODELS_DIR / "win_primary.pkl"
RF_BASELINE_MODEL_PATH = MODELS_DIR / "win_rf_baseline.pkl"
EARLY_MODEL_PATH = MODELS_DIR / "win_early.pkl"
METRICS_PATH = MODELS_DIR / "win_rf_metrics.json"
EARLY_METRICS_PATH = MODELS_DIR / "win_early_metrics.json"
LAST_GAME_REPORT_PATH = REPORTS_DIR / "last_game_report.md"
WEEKLY_REPORT_PATH = REPORTS_DIR / "weekly_summary.md"


def ensure_directories() -> None:
    for path in [RAW_DIR, TIMELINE_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DATA_DIR / "raw"]:
        path.mkdir(parents=True, exist_ok=True)


def get_api_key(cli_key: str | None = None) -> str:
    if os.getenv("RIOT_PROXY_URL"):
        return cli_key or ""

    api_key = cli_key or os.getenv("RIOT_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Riot API key. Add RIOT_API_KEY to your project .env file or pass --api-key."
        )
    return api_key

