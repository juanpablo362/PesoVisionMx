"""Tests for dashboard insight helpers."""

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.dashboard.insights import (
    build_report_pdf,
    explain_today,
    last_n_paper_trades,
)
from src.models.common import FEATURE_COLS, TARGET_COL


def test_explain_today_ranks_top_feature():
    row = pd.Series({col: 0.01 if "return" in col or col.endswith("d") else 1.0 for col in FEATURE_COLS})
    row["dxy_return_1d"] = 0.012
    medians = pd.Series({col: 0.0 for col in FEATURE_COLS})
    importance = {col: 0.05 for col in FEATURE_COLS}
    importance["dxy_return_1d"] = 0.9
    lines = explain_today(row, medians, importance, n=3)
    assert lines
    assert "DXY" in lines[0]
    assert "+1.20%" in lines[0]


def test_last_n_paper_trades_shape():
    n = 12
    df = pd.DataFrame(
        {col: [0.01] * n for col in FEATURE_COLS}
        | {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            TARGET_COL: [i % 2 for i in range(n)],
        }
    )
    model = LogisticRegression().fit(df[FEATURE_COLS], df[TARGET_COL])
    table = last_n_paper_trades(df, model, n=10)
    assert len(table) == 10
    assert list(table.columns) == ["Fecha", "Predicho", "Real", "Acierto"]
    assert table["Acierto"].isin(["Sí", "No"]).all()


def test_build_report_pdf_is_pdf():
    pdf = build_report_pdf(
        close=17.12,
        last_date="14 Aug 2026",
        label="Sube",
        p_up=0.62,
        p_down=0.38,
        model_name="Random Forest",
        f1=0.563,
        auc=0.747,
        reasons=["Hoy Retorno DXY 1d esta en +0.21%."],
    )
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 200
