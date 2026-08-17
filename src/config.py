"""Configuración central de PesoVision."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "pesovision.db"

# Yahoo Finance
FX_TICKER = "USDMXN=X"
DXY_TICKER = "DX-Y.NYB"
DXY_TICKER_FALLBACK = "DX=F"
FX_START_DATE = "2019-01-01"

# Limpieza
MAX_DAILY_RETURN = 0.15  # 15% — descartar outliers extremos
MAX_FILL_GAP_DAYS = 3
