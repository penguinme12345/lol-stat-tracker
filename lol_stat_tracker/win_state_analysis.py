"""Win-state analytics for lead conversion and comeback behavior."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_win_state_analytics(df: pd.DataFrame) -> dict[str, Any]:
    ahead15 = df[df.get("ahead_at_15", 0) == 1]
    behind15 = df[df.get("behind_at_15", 0) == 1]

    lead_conversion_rate = _safe_rate(int(ahead15["win"].sum()), len(ahead15))
    comeback_rate = _safe_rate(int(behind15["win"].sum()), len(behind15))
    throw_rate = _safe_rate(int((ahead15["win"] == 0).sum()), len(ahead15))

    snowball_source = df[df.get("gold_diff_10", 0) > 400]
    snowball_strength = _safe_rate(int(snowball_source["win"].sum()), len(snowball_source))

    return {
        "lead_conversion_rate": lead_conversion_rate,
        "comeback_rate": comeback_rate,
        "throw_rate": throw_rate,
        "snowball_strength": snowball_strength,
    }
