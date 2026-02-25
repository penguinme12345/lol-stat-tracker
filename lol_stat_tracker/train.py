"""Random Forest training and artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lol_stat_tracker.config import MATCHES_CSV_PATH, METRICS_PATH, MODEL_PATH, ensure_directories

TARGET_COL = "win"
FEATURE_COLS = [
    "duration_min",
    "champion",
    "role",
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
]


def _load_matches(path: Path = MATCHES_CSV_PATH) -> pd.DataFrame:
    if not path.exists():
        raise ValueError("Dataset not found. Run build-dataset first.")
    return pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)


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

    categorical = ["champion", "role"]
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
            ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)

    auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else None
    acc = accuracy_score(y_test, pred)

    rf = model.named_steps["rf"]
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importance_pairs = sorted(
        [{"feature": name, "importance": float(score)} for name, score in zip(feature_names, rf.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    joblib.dump(model, MODEL_PATH)
    metrics = {
        "num_matches": int(len(df)),
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "accuracy": float(acc),
        "roc_auc": float(auc) if auc is not None else None,
        "feature_importance": importance_pairs[:25],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return str(METRICS_PATH)

