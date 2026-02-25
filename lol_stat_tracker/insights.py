"""Insight generation and report orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass

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


def build_last_game_report() -> str:
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

    markdown = render_last_game_markdown(
        match_id=str(last["match_id"]),
        predicted_win_probability=win_prob,
        last_stats={k: float(last[k]) if isinstance(last[k], (int, float)) else last[k] for k in last.index},
        percentiles=percentile_stats,
        top_drivers=metrics.get("feature_importance", [])[:3],
        goals=goals,
        focus_goal=focus_goal,
    )
    LAST_GAME_REPORT_PATH.write_text(markdown, encoding="utf-8")
    return str(LAST_GAME_REPORT_PATH)


def build_weekly_summary() -> str:
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

    markdown = render_weekly_markdown(
        weekly_winrate=weekly_winrate,
        champion_performance=champ_perf.reset_index().to_dict(orient="records"),
        top_drivers=metrics.get("feature_importance", [])[:3],
        leaks=[_goal_text(leak) for leak in leaks],
    )
    WEEKLY_REPORT_PATH.write_text(markdown, encoding="utf-8")
    return str(WEEKLY_REPORT_PATH)

