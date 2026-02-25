"""Context-aware feature engineering helpers for coaching v2."""

from __future__ import annotations

from typing import Any

import pandas as pd


def select_context(df: pd.DataFrame, champion: str, role: str) -> tuple[pd.DataFrame, str]:
    champ_role = df[(df["champion"] == champion) & (df["role"] == role)]
    if len(champ_role) >= 25:
        return champ_role, "champion_role"

    role_only = df[df["role"] == role]
    if len(role_only) >= 40:
        return role_only, "role"

    return df, "global"


def rolling_value(series: pd.Series, window: int = 20) -> float:
    recent = series.tail(window)
    if recent.empty:
        return 0.0
    return float(recent.mean())


def percentile_target(win_series: pd.Series, direction: str) -> float:
    if win_series.empty:
        return 0.0
    quantile = 0.6 if direction == "increase" else 0.4
    return float(win_series.quantile(quantile))


def importance_map(metrics: dict[str, Any]) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for item in metrics.get("feature_importance", []) or []:
        name = str(item.get("feature", "")).replace("num__", "").replace("cat__", "")
        value = float(item.get("importance", 0.0))
        if value > mapping.get(name, 0.0):
            mapping[name] = value
    total = sum(mapping.values())
    if total <= 1e-9:
        return mapping
    return {name: (value / total) for name, value in mapping.items()}
