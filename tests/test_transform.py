"""Tests for ETL transform functions."""

import pandas as pd

from src.etl.transform import build_features, clean_fx


def _sample_raw() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    close = [20.0 + i * 0.01 for i in range(len(dates))]
    return pd.DataFrame(
        {
            "date": dates.date,
            "open": close,
            "high": [c + 0.05 for c in close],
            "low": [c - 0.05 for c in close],
            "close": close,
            "volume": [0] * len(dates),
            "source": ["test"] * len(dates),
            "extracted_at": ["2024-01-01T00:00:00+00:00"] * len(dates),
        }
    )


def test_clean_fx_removes_duplicates():
    raw = _sample_raw()
    dup = raw.copy()
    dup.iloc[0, dup.columns.get_loc("close")] = 99.0
    combined = pd.concat([raw, dup], ignore_index=True)
    clean, stats = clean_fx(combined)
    assert stats["rows_after_dedup"] == len(raw)
    assert len(clean) <= len(raw)


def test_clean_fx_removes_outliers():
    raw = _sample_raw()
    raw.loc[0, "close"] = raw.loc[1, "close"] * 1.5
    clean, stats = clean_fx(raw)
    assert stats["outliers_removed"] >= 1


def test_build_features_columns():
    clean, _ = clean_fx(_sample_raw())
    features = build_features(clean)
    expected = {
        "return_1d",
        "return_5d",
        "return_20d",
        "ma_ratio",
        "volatility_20d",
        "high_low_spread",
        "direction_next_day",
    }
    assert expected.issubset(set(features.columns))
    assert features["direction_next_day"].isin([0, 1]).all()
