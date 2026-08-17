"""Extracción de series diarias desde Yahoo Finance."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import (
    DXY_TICKER,
    DXY_TICKER_FALLBACK,
    FX_START_DATE,
    FX_TICKER,
    RAW_DIR,
)


def extract_yahoo(ticker: str, start: str = FX_START_DATE) -> pd.DataFrame:
    """Descarga OHLCV diario y normaliza columnas."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    data = yf.download(ticker, start=start, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(f"No se obtuvieron datos para {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = data.reset_index()
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"date": "date"})

    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["source"] = "yahoo"
    df["extracted_at"] = datetime.now(timezone.utc).isoformat()

    expected = ["date", "open", "high", "low", "close", "volume", "source", "extracted_at"]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes tras extracción: {missing}")

    return df[expected]


def extract_fx(ticker: str = FX_TICKER, start: str = FX_START_DATE) -> pd.DataFrame:
    """Descarga USD/MXN diario."""
    return extract_yahoo(ticker, start=start)


def extract_dxy(start: str = FX_START_DATE) -> pd.DataFrame:
    """Descarga el índice del dólar (DXY), con fallback a futuros DX=F."""
    try:
        return extract_yahoo(DXY_TICKER, start=start)
    except ValueError:
        return extract_yahoo(DXY_TICKER_FALLBACK, start=start)


def save_raw(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or RAW_DIR / "usdmxn_raw.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    frame = extract_fx()
    out = save_raw(frame)
    print(f"Extraídas {len(frame)} filas → {out}")
    dxy = extract_dxy()
    dxy_out = save_raw(dxy, RAW_DIR / "dxy_raw.csv")
    print(f"Extraídas {len(dxy)} filas DXY → {dxy_out}")
