"""Evaluación de modelos y persistencia de artefactos para el dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.models.common import (
    ALL_MODELS_PATH,
    BEST_MODEL_PATH,
    FEATURE_COLS,
    METRICS_PATH,
    MODELS_DIR,
    PREDICTIONS_PATH,
    ROC_PATH,
    TARGET_COL,
    build_candidates,
    load_features,
    temporal_split,
)


def _split_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _feature_importance(name: str, model) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "named_steps") and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "coef_"):
            values = np.abs(clf.coef_[0])
        elif hasattr(clf, "feature_importances_"):
            values = clf.feature_importances_
        else:
            return {}
    else:
        return {}
    return {feat: float(val) for feat, val in zip(FEATURE_COLS, values)}


def evaluate_fitted_models(
    fitted_models: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv_f1: dict[str, float] | None = None,
) -> dict:
    """Calcula métricas train/test y persiste artefactos."""
    MODELS_DIR.mkdir(exist_ok=True)

    metrics: dict = {"models": {}, "best_model": None, "best_f1_test": -1.0}
    roc_data: dict = {}
    best_name = None
    best_f1 = -1.0
    best_preds_df: pd.DataFrame | None = None

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    for name, model in fitted_models.items():
        train_pred = model.predict(X_train)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]

        test_m = _split_metrics(y_test, test_pred, test_proba)
        model_metrics = {
            "cv_f1_mean": cv_f1.get(name, 0.0) if cv_f1 else 0.0,
            "train": _split_metrics(y_train, train_pred, train_proba),
            "test": test_m,
            "feature_importance": _feature_importance(name, model),
        }
        metrics["models"][name] = model_metrics

        fpr, tpr, _ = roc_curve(y_test, test_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

        if test_m["f1"] > best_f1:
            best_f1 = test_m["f1"]
            best_name = name
            best_preds_df = pd.DataFrame(
                {
                    "date": test["date"].values,
                    "close": test["close"].values,
                    "y_true": y_test.values,
                    "y_pred": test_pred,
                    "y_proba": test_proba,
                    "correct": (test_pred == y_test.values).astype(int),
                    "model": name,
                }
            )

    metrics["best_model"] = best_name
    metrics["best_f1_test"] = best_f1

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ROC_PATH.write_text(json.dumps(roc_data, indent=2), encoding="utf-8")
    if best_preds_df is not None:
        best_preds_df.to_csv(PREDICTIONS_PATH, index=False)

    return metrics


def run_evaluation(fitted_models: dict | None = None, cv_f1: dict[str, float] | None = None) -> dict:
    """Evalúa modelos entrenados o carga desde all_models.pkl."""
    df = load_features().sort_values("date")
    train, test = temporal_split(df)

    if fitted_models is None:
        if not ALL_MODELS_PATH.exists():
            raise FileNotFoundError(
                "No hay modelos entrenados. Ejecuta: python -m src.models.train"
            )
        fitted_models = joblib.load(ALL_MODELS_PATH)

    metrics = evaluate_fitted_models(fitted_models, train, test, cv_f1=cv_f1)
    print(f"Evaluación guardada -> {METRICS_PATH}")
    print(f"Mejor modelo (test F1): {metrics['best_model']} ({metrics['best_f1_test']:.3f})")
    for name, m in metrics["models"].items():
        tr, te = m["train"], m["test"]
        print(
            f"  {name}: train F1={tr['f1']:.3f} | test F1={te['f1']:.3f} | "
            f"test Acc={te['accuracy']:.3f} | ROC-AUC={te['roc_auc']:.3f}"
        )
    return metrics


if __name__ == "__main__":
    run_evaluation()
