"""Insight generation and report orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from lol_stat_tracker.config import (
    LAST_GAME_REPORT_PATH,
    MATCHES_CSV_PATH,
    METRICS_PATH,
    MODEL_PATH,
    WEEKLY_REPORT_PATH,
    ensure_directories,
)
from lol_stat_tracker.report import render_last_game_markdown, render_weekly_markdown
from lol_stat_tracker.train import FEATURE_COLS


@dataclass
class Leak:
    metric: str
    win_avg: float
    loss_avg: float
    direction: str
    delta: float


def _load_data() -> tuple[pd.DataFrame, dict]:
    if not MATCHES_CSV_PATH.exists():
        raise ValueError("No dataset found. Run build-dataset first.")
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise ValueError("Model artifacts missing. Run train first.")
    df = pd.read_csv(MATCHES_CSV_PATH).sort_values("timestamp").reset_index(drop=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return df, metrics


def _compute_leaks(df: pd.DataFrame) -> list[Leak]:
    candidate_metrics = [
        "deaths",
        "cs_per_min",
        "damage_per_min",
        "vision_per_min",
        "gold_per_min",
        "kill_participation",
    ]
    leaks: list[Leak] = []
    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]
    if wins.empty or losses.empty:
        return leaks

    higher_is_better = {"cs_per_min", "damage_per_min", "vision_per_min", "gold_per_min", "kill_participation"}
    for metric in candidate_metrics:
        win_avg = float(wins[metric].mean())
        loss_avg = float(losses[metric].mean())
        if metric in higher_is_better:
            delta = win_avg - loss_avg
            direction = "increase"
        else:
            delta = loss_avg - win_avg
            direction = "decrease"
        leaks.append(Leak(metric=metric, win_avg=win_avg, loss_avg=loss_avg, direction=direction, delta=abs(delta)))
    return sorted(leaks, key=lambda x: x.delta, reverse=True)[:3]


def _goal_text(leak: Leak) -> str:
    if leak.direction == "decrease":
        target = max(leak.win_avg, 0.0)
        return f"Reduce {leak.metric} toward {target:.2f} (wins avg: {leak.win_avg:.2f})"
    return f"Increase {leak.metric} toward {leak.win_avg:.2f} (wins avg: {leak.win_avg:.2f})"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def last_game_payload() -> dict[str, Any]:
    ensure_directories()
    df, metrics = _load_data()
    model = joblib.load(MODEL_PATH)

    last = df.iloc[-1].copy()
    recent = df.tail(min(50, len(df))).copy()
    percentile_stats = {}
    for col in ["kills", "deaths", "assists", "cs_per_min", "damage_per_min", "vision_per_min"]:
        percentile_stats[col] = float((recent[col] <= float(last[col])).mean() * 100.0)

    win_prob = float(model.predict_proba(pd.DataFrame([last[FEATURE_COLS]]))[:, 1][0])
    leaks = _compute_leaks(df)
    goals = [_goal_text(leak) for leak in leaks]
    focus_goal = goals[0] if goals else "Play with consistency and avoid high-variance fights."

    return {
        "match_id": str(last["match_id"]),
        "predicted_win_probability": win_prob,
        "result": "Win" if int(last.get("win", 0)) == 1 else "Loss",
        "champion": str(last.get("champion", "unknown")),
        "role": str(last.get("role", "unknown")),
        "key_stats": {
            "kills": float(last.get("kills", 0)),
            "deaths": float(last.get("deaths", 0)),
            "assists": float(last.get("assists", 0)),
            "cs_per_min": float(last.get("cs_per_min", 0)),
            "damage_per_min": float(last.get("damage_per_min", 0)),
            "gold_per_min": float(last.get("gold_per_min", 0)),
            "vision_per_min": float(last.get("vision_per_min", 0)),
        },
        "percentiles": percentile_stats,
        "top_drivers": metrics.get("feature_importance", [])[:3],
        "improvement_targets": goals,
        "focus_goal": focus_goal,
    }


def build_last_game_report() -> str:
    payload = last_game_payload()
    markdown = render_last_game_markdown(
        match_id=payload["match_id"],
        predicted_win_probability=float(payload["predicted_win_probability"]),
        last_stats={
            "win": 1 if payload["result"] == "Win" else 0,
            "champion": payload["champion"],
            "role": payload["role"],
            "kills": payload["key_stats"]["kills"],
            "deaths": payload["key_stats"]["deaths"],
            "assists": payload["key_stats"]["assists"],
            "cs_per_min": payload["key_stats"]["cs_per_min"],
            "damage_per_min": payload["key_stats"]["damage_per_min"],
            "gold_per_min": payload["key_stats"]["gold_per_min"],
            "vision_per_min": payload["key_stats"]["vision_per_min"],
        },
        percentiles=payload["percentiles"],
        top_drivers=payload["top_drivers"],
        goals=payload["improvement_targets"],
        focus_goal=payload["focus_goal"],
    )
    LAST_GAME_REPORT_PATH.write_text(markdown, encoding="utf-8")
    return str(LAST_GAME_REPORT_PATH)


def weekly_summary_payload() -> dict[str, Any]:
    ensure_directories()
    df, metrics = _load_data()
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column missing from dataset.")

    # Riot timestamps are in milliseconds.
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    week_df = df.dropna(subset=["dt"]).copy()
    if week_df.empty:
        week_df = df.copy()
        week_df["week"] = "unknown"
    else:
        week_df["week"] = week_df["dt"].dt.to_period("W").astype(str)

    weekly_winrate = week_df.groupby("week")["win"].mean().tail(4).to_dict()
    champ_perf = df.groupby("champion")["win"].agg(["count", "mean"]).sort_values("count", ascending=False).head(8)
    leaks = _compute_leaks(df)

    champion_rows = [
        {"champion": row["champion"], "count": int(row["count"]), "mean": float(row["mean"])}
        for row in champ_perf.reset_index().to_dict(orient="records")
    ]
    return {
        "weekly_winrate": {str(k): float(v) for k, v in weekly_winrate.items()},
        "champion_performance": champion_rows,
        "top_drivers": metrics.get("feature_importance", [])[:3],
        "leaks": [_goal_text(leak) for leak in leaks],
    }


def build_weekly_summary() -> str:
    payload = weekly_summary_payload()
    markdown = render_weekly_markdown(
        weekly_winrate=payload["weekly_winrate"],
        champion_performance=payload["champion_performance"],
        top_drivers=payload["top_drivers"],
        leaks=payload["leaks"],
    )
    WEEKLY_REPORT_PATH.write_text(markdown, encoding="utf-8")
    return str(WEEKLY_REPORT_PATH)
