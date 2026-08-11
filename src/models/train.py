"""Entrenamiento de modelos supervisados (Fase 3)."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
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
]
TARGET_COL = "direction_next_day"


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "fx_features.parquet"
    if not path.exists():
        path = PROCESSED_DIR / "fx_features.csv"
    if not path.exists():
        raise FileNotFoundError("Ejecuta primero: python -m src.etl.run_etl")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    n_test = int(len(df) * test_ratio)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]
    return train, test


def train_and_evaluate() -> dict:
    df = load_features().sort_values("date")
    train, test = temporal_split(df)

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    candidates = {
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

    results = {}
    best_name, best_model, best_f1 = None, None, -1.0

    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="f1")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        report = {
            "cv_f1_mean": float(cv_scores.mean()),
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "classification_report": classification_report(y_test, preds),
        }
        results[name] = report

        if report["f1"] > best_f1:
            best_f1, best_name, best_model = report["f1"], name, model

    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump({"name": best_name, "model": best_model, "features": FEATURE_COLS}, model_path)

    print(f"Mejor modelo: {best_name} (F1={best_f1:.3f})")
    for name, rep in results.items():
        print(f"\n--- {name} ---")
        print(f"CV F1: {rep['cv_f1_mean']:.3f} | Test Acc: {rep['accuracy']:.3f} | F1: {rep['f1']:.3f}")
        print(rep["classification_report"])

    return {"results": results, "best_model": best_name, "model_path": str(model_path)}


if __name__ == "__main__":
    train_and_evaluate()
