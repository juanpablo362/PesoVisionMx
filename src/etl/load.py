"""Carga de datos limpios en SQLite y exportación procesada."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_fx_daily (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            extracted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS raw_dxy_daily (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            extracted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS fx_clean (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            return_1d REAL,
            source TEXT,
            extracted_at TEXT
        );

        CREATE TABLE IF NOT EXISTS fx_features (
            date TEXT PRIMARY KEY,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            return_1d REAL,
            return_5d REAL,
            return_20d REAL,
            ma_5 REAL,
            ma_20 REAL,
            ma_ratio REAL,
            volatility_20d REAL,
            high_low_spread REAL,
            dxy_close REAL,
            dxy_return_1d REAL,
            dxy_return_5d REAL,
            direction_next_day INTEGER
        );
        """
    )


def load_to_sqlite(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    features: pd.DataFrame,
    db_path: Path = DB_PATH,
    raw_dxy: pd.DataFrame | None = None,
) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        _init_schema(conn)
        raw.to_sql("raw_fx_daily", conn, if_exists="replace", index=False)
        if raw_dxy is not None:
            raw_dxy.to_sql("raw_dxy_daily", conn, if_exists="replace", index=False)
        clean.to_sql("fx_clean", conn, if_exists="replace", index=False)
        features.to_sql("fx_features", conn, if_exists="replace", index=False)

    return db_path


def save_processed(features: pd.DataFrame, path: Path | None = None) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = path or PROCESSED_DIR / "fx_features.parquet"
    features.to_parquet(path, index=False)
    csv_path = path.with_suffix(".csv")
    features.to_csv(csv_path, index=False)
    return path
