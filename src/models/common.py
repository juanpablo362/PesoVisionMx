"""Utilidades compartidas para entrenamiento y evaluación."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import PROCESSED_DIR, ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"
FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "ma_ratio",
    "volatility_20d",
    "high_low_spread",
    "dxy_return_1d",
    "dxy_return_5d",
]
TARGET_COL = "direction_next_day"

METRICS_PATH = MODELS_DIR / "metrics.json"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"
ROC_PATH = MODELS_DIR / "roc_curve.json"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
ALL_MODELS_PATH = MODELS_DIR / "all_models.pkl"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "fx_features.parquet"
    if not path.exists():
        path = PROCESSED_DIR / "fx_features.csv"
    if not path.exists():
        raise FileNotFoundError("Ejecuta primero: python -m src.etl.run_etl")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_test = int(len(df) * test_ratio)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    return train, test


def build_candidates() -> dict:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        ),
    }
