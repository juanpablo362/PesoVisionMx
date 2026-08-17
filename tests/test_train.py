"""Tests for model training utilities."""

import pandas as pd

from src.models.common import FEATURE_COLS, build_candidates, temporal_split


def test_temporal_split_preserves_order():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100), "x": range(100)})
    train, test = temporal_split(df, test_ratio=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert train["date"].max() < test["date"].min()


def test_feature_cols_count():
    assert len(FEATURE_COLS) == 8
    assert "dxy_return_1d" in FEATURE_COLS
    assert "dxy_return_5d" in FEATURE_COLS


def test_build_candidates_includes_gradient_boosting():
    names = set(build_candidates())
    assert names == {"logistic_regression", "random_forest", "gradient_boosting"}
