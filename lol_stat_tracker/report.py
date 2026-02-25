"""Markdown report rendering."""

from __future__ import annotations

from typing import Any


def render_last_game_markdown(
    match_id: str,
    predicted_win_probability: float,
    last_stats: dict[str, Any],
    percentiles: dict[str, float],
    top_drivers: list[dict[str, Any]],
    goals: list[str],
    focus_goal: str,
) -> str:
    driver_lines = "\n".join(
        f"- `{d.get('feature', 'unknown')}`: {float(d.get('importance', 0.0)):.3f}" for d in top_drivers
    )
    percentile_lines = "\n".join(f"- `{k}` percentile vs last games: {v:.1f}%" for k, v in percentiles.items())
    goal_lines = "\n".join(f"- {goal}" for goal in goals) if goals else "- No clear leaks yet."

    return f"""# Last Game Report

## Match
- Match ID: `{match_id}`
- Predicted win probability: **{predicted_win_probability:.1%}**
- Result: **{"Win" if int(last_stats.get("win", 0)) == 1 else "Loss"}**
- Champion/Role: `{last_stats.get("champion", "unknown")} / {last_stats.get("role", "unknown")}`

## Key Stats
- K/D/A: `{last_stats.get("kills", 0):.0f}/{last_stats.get("deaths", 0):.0f}/{last_stats.get("assists", 0):.0f}`
- CS/min: `{float(last_stats.get("cs_per_min", 0.0)):.2f}`
- Damage/min: `{float(last_stats.get("damage_per_min", 0.0)):.1f}`
- Gold/min: `{float(last_stats.get("gold_per_min", 0.0)):.1f}`
- Vision/min: `{float(last_stats.get("vision_per_min", 0.0)):.2f}`

## Percentiles
{percentile_lines}

## Top Win-Rate Drivers
{driver_lines}

## Improvement Targets
{goal_lines}

## Focus Goal For Next Match
{focus_goal}
"""


def render_weekly_markdown(
    weekly_winrate: dict[str, float],
    champion_performance: list[dict[str, Any]],
    top_drivers: list[dict[str, Any]],
    leaks: list[str],
) -> str:
    winrate_lines = "\n".join(f"- `{week}`: **{rate:.1%}**" for week, rate in weekly_winrate.items())
    champ_lines = "\n".join(
        f"- `{row.get('champion', 'unknown')}`: {int(row.get('count', 0))} games, {float(row.get('mean', 0.0)):.1%} win rate"
        for row in champion_performance
    )
    driver_lines = "\n".join(
        f"- `{d.get('feature', 'unknown')}`: {float(d.get('importance', 0.0)):.3f}" for d in top_drivers
    )
    leak_lines = "\n".join(f"- {leak}" for leak in leaks) if leaks else "- Insufficient separation in wins/losses."

    return f"""# Weekly Summary

## Win Rate Trend (Recent Weeks)
{winrate_lines}

## Champion Pool Performance
{champ_lines}

## Top Win-Rate Drivers
{driver_lines}

## Progress Targets
{leak_lines}
"""

