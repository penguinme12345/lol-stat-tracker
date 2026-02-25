"""Model training and artifact persistence (v4 architecture)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lol_stat_tracker.config import (
    EARLY_METRICS_PATH,
    EARLY_MODEL_PATH,
    MATCHES_CSV_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RF_BASELINE_MODEL_PATH,
    ensure_directories,
)

TARGET_COL = "win"
FEATURE_COLS = [
    "duration_min",
    "champion",
    "role",
    "diff_reference",
    "opponent_resolution_quality",
    "kills",
    "deaths",
    "assists",
    "kda",
    "cs_total",
    "cs_per_min",
    "damage_to_champions",
    "damage_per_min",
    "gold_earned",
    "gold_per_min",
    "vision_score",
    "vision_per_min",
    "kill_participation",
    "team_damage_share",
    "gold_5",
    "xp_5",
    "cs_5",
    "gold_10",
    "xp_10",
    "cs_10",
    "gold_15",
    "xp_15",
    "cs_15",
    "gold_diff_5",
    "xp_diff_5",
    "cs_diff_5",
    "gold_diff_10",
    "xp_diff_10",
    "cs_diff_10",
    "gold_diff_15",
    "xp_diff_15",
    "cs_diff_15",
    "level_diff_15",
    "gold_diff_5_to_10",
    "gold_diff_10_to_15",
    "xp_diff_5_to_10",
    "xp_diff_10_to_15",
    "cs_diff_5_to_10",
    "cs_diff_10_to_15",
    "level_diff_10",
    "ahead_at_10",
    "ahead_at_15",
    "behind_at_10",
    "behind_at_15",
    "first_blood_participation",
    "first_death_time_min",
    "deaths_before_15",
    "deaths_before_dragon_60s",
    "deaths_before_herald_60s",
    "deaths_before_baron_60s",
    "late_game_deaths_post20",
    "death_rate_after_15",
    "time_to_first_tower_min",
    "damage_per_gold",
    "damage_share",
    "gold_share",
    "kill_share",
    "solo_kills",
    "kills_near_enemy_turret",
    "multikill_score",
    "vision_control_ratio",
    "control_wards_per_game",
    "dragon_participation_rate",
    "baron_participation_rate",
    "turret_participation_rate",
    "plate_control_pct",
    "aggression_index",
    "farming_discipline_index",
    "objective_discipline_index",
    "clutch_index",
]

EARLY_FEATURE_COLS = [
    "champion",
    "role",
    "diff_reference",
    "opponent_resolution_quality",
    "gold_5",
    "xp_5",
    "cs_5",
    "gold_10",
    "xp_10",
    "cs_10",
    "gold_15",
    "xp_15",
    "cs_15",
    "gold_diff_5",
    "xp_diff_5",
    "cs_diff_5",
    "gold_diff_10",
    "xp_diff_10",
    "cs_diff_10",
    "level_diff_10",
    "ahead_at_10",
    "behind_at_10",
    "first_blood_participation",
    "first_death_time_min",
    "deaths_before_15",
    "deaths_before_dragon_60s",
    "deaths_before_herald_60s",
    "time_to_first_tower_min",
]


def _load_matches(path: Path = MATCHES_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        raise ValueError("Dataset not found. Run build-dataset first.")
    df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
    for col in set(FEATURE_COLS + EARLY_FEATURE_COLS):
        if col not in df.columns:
            df[col] = "UNKNOWN" if col in {"champion", "role", "diff_reference", "opponent_resolution_quality"} else 0.0
    return df


def _permutation_importance_pairs(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> list[dict[str, float]]:
    if x_test.empty:
        return []

    scoring = "roc_auc" if y_test.nunique() > 1 else "accuracy"
    try:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=12,
            random_state=42,
            n_jobs=-1,
            scoring=scoring,
        )
    except Exception:
        return []

    pairs: list[dict[str, float]] = []
    for feature, score in zip(FEATURE_COLS, result.importances_mean):
        pairs.append({"feature": feature, "importance": max(float(score), 0.0)})

    pairs = sorted(pairs, key=lambda x: x["importance"], reverse=True)
    if not pairs or pairs[0]["importance"] <= 0:
        return []
    return pairs


def train_win_model() -> str:
    ensure_directories()
    df = _load_matches()
    if len(df) < 20:
        raise ValueError("Need at least 20 matches to train a stable model.")

    split_idx = int(len(df) * 0.8)
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Invalid split for dataset length.")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    x_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    x_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    categorical = ["champion", "role", "diff_reference", "opponent_resolution_quality"]
    numeric = [col for col in FEATURE_COLS if col not in categorical]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=250,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                    objective="binary",
                    verbose=-1,
                ),
            ),
        ]
    )
    primary_model_name = "lightgbm"
    importance_pairs: list[dict[str, float]] = []
    try:
        model.fit(x_train, y_train)
    except TypeError:
        fallback = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("hgb", HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, random_state=42)),
            ]
        )
        fallback.fit(x_train, y_train)
        model = fallback
        primary_model_name = "sklearn_hgb_fallback"
        importance_pairs = []

    proba = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)

    auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else None
    acc = accuracy_score(y_test, pred)

    importance_pairs = _permutation_importance_pairs(model, x_test, y_test)

    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    if not importance_pairs:
        default_importance = 1.0 / max(len(feature_names), 1)
        importance_pairs = [{"feature": name, "importance": default_importance} for name in feature_names[:25]]

    joblib.dump(model, MODEL_PATH)

    rf_baseline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    rf_baseline.fit(x_train, y_train)
    rf_pred = rf_baseline.predict(x_test)
    rf_proba = rf_baseline.predict_proba(x_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba) if len(set(y_test)) > 1 else None
    rf_acc = accuracy_score(y_test, rf_pred)
    model_agreement = float((pred == rf_pred).mean()) if len(pred) else 0.0
    joblib.dump(rf_baseline, RF_BASELINE_MODEL_PATH)

    early_categorical = ["champion", "role", "diff_reference", "opponent_resolution_quality"]
    early_numeric = [col for col in EARLY_FEATURE_COLS if col not in early_categorical]
    early_preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", early_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), early_categorical),
        ]
    )
    early_model = Pipeline(
        steps=[
            ("preprocess", early_preprocess),
            ("lr", LogisticRegression(max_iter=1200, class_weight="balanced")),
        ]
    )
    early_model.fit(train_df[EARLY_FEATURE_COLS], y_train)
    early_proba = early_model.predict_proba(test_df[EARLY_FEATURE_COLS])[:, 1]
    early_pred = early_model.predict(test_df[EARLY_FEATURE_COLS])
    early_auc = roc_auc_score(y_test, early_proba) if len(set(y_test)) > 1 else None
    early_acc = accuracy_score(y_test, early_pred)

    joblib.dump(early_model, EARLY_MODEL_PATH)

    metrics = {
        "num_matches": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "primary_model": primary_model_name,
        "accuracy": float(acc),
        "roc_auc": float(auc) if auc is not None else None,
        "rf_baseline_accuracy": float(rf_acc),
        "rf_baseline_roc_auc": float(rf_auc) if rf_auc is not None else None,
        "model_agreement": model_agreement,
        "early_model_accuracy": float(early_acc),
        "early_model_roc_auc": float(early_auc) if early_auc is not None else None,
        "feature_importance_method": "permutation" if importance_pairs and "__" not in importance_pairs[0]["feature"] else "fallback_uniform",
        "feature_importance": importance_pairs[:25],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    EARLY_METRICS_PATH.write_text(
        json.dumps(
            {
                "num_matches": int(len(df)),
                "train_size": int(len(train_df)),
                "test_size": int(len(test_df)),
                "accuracy": float(early_acc),
                "roc_auc": float(early_auc) if early_auc is not None else None,
                "feature_space": EARLY_FEATURE_COLS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(METRICS_PATH)

