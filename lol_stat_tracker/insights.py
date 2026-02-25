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


@dataclass
class WinLever:
    feature: str
    direction: str
    target: float
    uplift: float


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


def _feature_importance_map(metrics: dict[str, Any]) -> dict[str, float]:
    importance: dict[str, float] = {}
    for item in metrics.get("feature_importance", []) or []:
        feature_name = str(item.get("feature", ""))
        score = float(item.get("importance", 0.0))
        normalized_name = feature_name.replace("num__", "").replace("cat__", "")
        if score > importance.get(normalized_name, 0.0):
            importance[normalized_name] = score
    return importance


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _ratio_toward_win(value: float, win_avg: float, loss_avg: float, direction: str) -> float:
    gap = abs(win_avg - loss_avg)
    if gap <= 1e-6:
        return 0.5

    if direction == "decrease":
        score = (loss_avg - value) / gap
    else:
        score = (value - loss_avg) / gap
    return _clamp(score, 0.0, 1.0)


def _format_target(feature: str, target: float, direction: str) -> str:
    if feature == "deaths":
        return f"Keep deaths ≤ {max(1, round(target))}"
    if feature in {"gold_per_min", "damage_per_min", "cs_per_min", "vision_per_min"}:
        comparator = "≥" if direction == "increase" else "≤"
        return f"Target {feature.replace('_', '/')} {comparator} {target:.0f}"
    comparator = "≥" if direction == "increase" else "≤"
    return f"Target {feature.replace('_', ' ')} {comparator} {target:.2f}"


def _confidence_label(metrics: dict[str, Any]) -> str:
    auc = metrics.get("roc_auc")
    auc_score = float(auc) if auc is not None else 0.5
    num_matches = int(metrics.get("num_matches", 0))

    if auc_score >= 0.74 and num_matches >= 80:
        return "High"
    if auc_score >= 0.62 and num_matches >= 40:
        return "Moderate"
    return "Low"


def _weekly_trend_text(df: pd.DataFrame, performance_series: pd.Series) -> str:
    trend_df = df.copy()
    trend_df["performance_index"] = performance_series
    trend_df["dt"] = pd.to_datetime(trend_df["timestamp"], unit="ms", errors="coerce")
    trend_df = trend_df.dropna(subset=["dt"]).copy()
    if len(trend_df) < 8:
        return "Not enough weekly history yet."

    trend_df["week"] = trend_df["dt"].dt.to_period("W").astype(str)
    weekly = trend_df.groupby("week").agg(win_rate=("win", "mean"), perf=("performance_index", "mean"))
    if len(weekly) < 2:
        return "Not enough weekly history yet."

    last = weekly.iloc[-1]
    prev = weekly.iloc[-2]
    win_delta = float((last["win_rate"] - prev["win_rate"]) * 100.0)
    perf_delta = float(last["perf"] - prev["perf"])

    if win_delta >= 3 or perf_delta >= 4:
        return f"Your performance improved {max(perf_delta, 0.0):.0f}% this week."
    if win_delta <= -3 or perf_delta <= -4:
        return "Win trend is declining over the last 7 days."
    return "Your weekly performance is stable."


def _ai_feedback(
    performance_index: int,
    confidence: str,
    win_probability_last_game: float,
    focus_goal: str,
    top_improvements: list[str],
    weekly_trend: str,
) -> str:
    support_one = top_improvements[0] if top_improvements else "Maintain consistent laning fundamentals"
    support_two = (
        top_improvements[1]
        if len(top_improvements) > 1
        else "Prioritize safer fights before major objectives"
    )
    return (
        f"Your current performance index is {performance_index}/100 with {confidence.lower()} model confidence. "
        f"Last game win probability was {win_probability_last_game:.0%}, which points to clear improvement leverage. "
        f"Primary focus: {focus_goal}. "
        f"Supporting priorities are {support_one.lower()} and {support_two.lower()}. "
        f"{weekly_trend} "
        "Keep execution simple and repeat these habits across your next games."
    )


def intelligence_report_payload() -> dict[str, Any]:
    ensure_directories()
    df, metrics = _load_data()
    model = joblib.load(MODEL_PATH)

    candidate_metrics = {
        "deaths": "decrease",
        "gold_per_min": "increase",
        "damage_per_min": "increase",
        "cs_per_min": "increase",
        "kill_participation": "increase",
        "vision_per_min": "increase",
    }

    wins = df[df["win"] == 1]
    losses = df[df["win"] == 0]
    if wins.empty or losses.empty:
        raise ValueError("Need both wins and losses to generate intelligence report.")

    importance = _feature_importance_map(metrics)
    weighted_components: list[tuple[float, float]] = []
    levers: list[WinLever] = []

    for feature, direction in candidate_metrics.items():
        win_avg = float(wins[feature].mean())
        loss_avg = float(losses[feature].mean())
        last_value = float(df.iloc[-1][feature])
        ratio = _ratio_toward_win(last_value, win_avg, loss_avg, direction)
        weight = float(importance.get(feature, 0.03))
        weighted_components.append((ratio, weight))

        gap = abs(win_avg - loss_avg)
        scale = max(abs(loss_avg), 1.0)
        uplift = _clamp((gap / scale) * 0.25 + weight * 0.4, 0.03, 0.25)
        target = win_avg
        levers.append(WinLever(feature=feature, direction=direction, target=target, uplift=uplift))

    total_weight = sum(weight for _, weight in weighted_components) or 1.0
    normalized_score = sum(score * weight for score, weight in weighted_components) / total_weight
    performance_index = int(round(_clamp(normalized_score, 0.0, 1.0) * 100.0))

    last = df.iloc[-1].copy()
    win_probability_last_game = float(model.predict_proba(pd.DataFrame([last[FEATURE_COLS]]))[:, 1][0])
    confidence = _confidence_label(metrics)

    ranked_levers = sorted(levers, key=lambda lever: lever.uplift, reverse=True)[:3]
    focus_lever = ranked_levers[0]
    focus_goal = _format_target(focus_lever.feature, focus_lever.target, focus_lever.direction)
    top_improvements = [
        f"{_format_target(lever.feature, lever.target, lever.direction)} (Estimated +{int(round(lever.uplift * 100))}% win chance)"
        for lever in ranked_levers
    ]

    row_scores = []
    for _, row in df.iterrows():
        row_total = 0.0
        for feature, direction in candidate_metrics.items():
            win_avg = float(wins[feature].mean())
            loss_avg = float(losses[feature].mean())
            ratio = _ratio_toward_win(float(row[feature]), win_avg, loss_avg, direction)
            weight = float(importance.get(feature, 0.03))
            row_total += ratio * weight
        row_scores.append((row_total / total_weight) * 100.0)

    weekly_trend = _weekly_trend_text(df, pd.Series(row_scores, index=df.index))
    ai_feedback = _ai_feedback(
        performance_index=performance_index,
        confidence=confidence,
        win_probability_last_game=win_probability_last_game,
        focus_goal=focus_goal,
        top_improvements=top_improvements,
        weekly_trend=weekly_trend,
    )

    return {
        "performance_index": performance_index,
        "confidence": confidence,
        "win_probability_last_game": round(win_probability_last_game, 4),
        "focus_goal": focus_goal,
        "top_improvements": top_improvements,
        "weekly_trend": weekly_trend,
        "ai_feedback": ai_feedback,
    }


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
