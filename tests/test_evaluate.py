"""Tests for evaluation artifacts."""

import json

from src.models.common import METRICS_PATH


def test_metrics_json_structure():
    assert METRICS_PATH.exists(), "Ejecuta python -m src.models.train antes de los tests"
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    assert "models" in metrics
    assert "best_model" in metrics
    for name in ("logistic_regression", "random_forest"):
        assert name in metrics["models"]
        model = metrics["models"][name]
        assert "train" in model and "test" in model
        assert "f1" in model["test"]
        assert "confusion_matrix" in model["test"]
