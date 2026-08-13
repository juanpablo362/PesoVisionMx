"""Entrenamiento de modelos supervisados (Fase 3)."""

from __future__ import annotations

import joblib
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.models.common import (
    ALL_MODELS_PATH,
    BEST_MODEL_PATH,
    FEATURE_COLS,
    MODELS_DIR,
    TARGET_COL,
    build_candidates,
    load_features,
    temporal_split,
)
from src.models.evaluate import evaluate_fitted_models


def train_and_evaluate() -> dict:
    df = load_features().sort_values("date")
    train, test = temporal_split(df)

    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    candidates = build_candidates()
    fitted_models = {}
    cv_f1: dict[str, float] = {}
    results = {}
    best_name, best_model, best_f1 = None, None, -1.0

    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="f1")
        cv_f1[name] = float(cv_scores.mean())
        model.fit(X_train, y_train)
        fitted_models[name] = model

        preds = model.predict(X_test)
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        f1 = f1_score(y_test, preds)
        report = {
            "cv_f1_mean": cv_f1[name],
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1,
            "classification_report": classification_report(y_test, preds),
        }
        results[name] = report

        if f1 > best_f1:
            best_f1, best_name, best_model = f1, name, model

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted_models, ALL_MODELS_PATH)
    joblib.dump({"name": best_name, "model": best_model, "features": FEATURE_COLS}, BEST_MODEL_PATH)

    metrics = evaluate_fitted_models(fitted_models, train, test, cv_f1=cv_f1)

    print(f"Mejor modelo: {best_name} (F1={best_f1:.3f})")
    for name, rep in results.items():
        print(f"\n--- {name} ---")
        print(f"CV F1: {rep['cv_f1_mean']:.3f} | Test Acc: {rep['accuracy']:.3f} | F1: {rep['f1']:.3f}")
        print(rep["classification_report"])

    return {
        "results": results,
        "best_model": best_name,
        "model_path": str(BEST_MODEL_PATH),
        "metrics": metrics,
    }


if __name__ == "__main__":
    train_and_evaluate()
