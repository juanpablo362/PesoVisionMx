"""Limpieza y transformación de la serie FX."""

from __future__ import annotations

import pandas as pd

from src.config import MAX_DAILY_RETURN, MAX_FILL_GAP_DAYS


def _forward_fill_close(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Rellena Close solo en huecos cortos (fin de semana / festivo)."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    missing = out["close"].isna()
    if not missing.any():
        return out

    # Identificar grupos consecutivos de nulos
    gap_id = missing.ne(missing.shift()).cumsum()
    gap_sizes = missing.groupby(gap_id).transform("sum")
    fillable = missing & (gap_sizes <= max_gap)
    out.loc[fillable, "close"] = out["close"].ffill()
    return out


def clean_fx(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpia nulos, duplicados y outliers de retorno diario.

    Returns:
        DataFrame limpio y diccionario con estadísticas del proceso.
    """
    stats: dict = {"rows_in": len(df)}

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")

    # Duplicados: conservar última extracción
    if "extracted_at" in out.columns:
        out = out.sort_values("extracted_at")
    out = out.drop_duplicates(subset=["date"], keep="last")
    stats["rows_after_dedup"] = len(out)

    out = _forward_fill_close(out, MAX_FILL_GAP_DAYS)
    stats["nulls_close_before_drop"] = int(out["close"].isna().sum())
    out = out.dropna(subset=["close"])
    stats["rows_after_null_drop"] = len(out)

    out["return_1d"] = out["close"].pct_change()
    outliers = out["return_1d"].abs() > MAX_DAILY_RETURN
    stats["outliers_removed"] = int(outliers.sum())
    out = out.loc[~outliers].copy()
    stats["rows_out"] = len(out)

    out["date"] = out["date"].dt.date
    return out.reset_index(drop=True), stats


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera features técnicas y target para clasificación."""
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["return_5d"] = out["close"].pct_change(5)
    out["return_20d"] = out["close"].pct_change(20)
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma_ratio"] = out["ma_5"] / out["ma_20"]
    out["volatility_20d"] = out["return_1d"].rolling(20).std()
    out["high_low_spread"] = (out["high"] - out["low"]) / out["close"]

    # Target: dirección del día siguiente
    out["direction_next_day"] = (out["close"].shift(-1) > out["close"]).astype("Int64")

    # Última fila no tiene target
    out = out.iloc[:-1].copy()
    out = out.dropna()
    out["direction_next_day"] = out["direction_next_day"].astype(int)

    return out.reset_index(drop=True)
