"""Orquestador ETL: extracción → transformación → carga."""

from __future__ import annotations

from src.config import RAW_DIR
from src.etl.extract import extract_dxy, extract_fx, save_raw
from src.etl.load import load_to_sqlite, save_processed
from src.etl.transform import build_features, clean_fx


def run() -> None:
    print("=== PesoVision ETL ===")

    print("[1/4] Extracción...")
    raw = extract_fx()
    raw_path = save_raw(raw)
    print(f"      USD/MXN {len(raw)} filas -> {raw_path}")
    raw_dxy = extract_dxy()
    dxy_path = save_raw(raw_dxy, RAW_DIR / "dxy_raw.csv")
    print(f"      DXY {len(raw_dxy)} filas -> {dxy_path}")

    print("[2/4] Limpieza...")
    clean, stats = clean_fx(raw)
    print("      USD/MXN")
    for key, value in stats.items():
        print(f"        {key}: {value}")
    clean_dxy, dxy_stats = clean_fx(raw_dxy)
    print("      DXY")
    for key, value in dxy_stats.items():
        print(f"        {key}: {value}")

    print("[3/4] Feature engineering...")
    features = build_features(clean, clean_dxy)
    n_labeled = int(features["direction_next_day"].notna().sum())
    n_infer = len(features) - n_labeled
    print(f"      {len(features)} filas ({n_labeled} con target, {n_infer} para inferencia)")

    print("[4/4] Carga...")
    db = load_to_sqlite(raw, clean, features, raw_dxy=raw_dxy)
    processed = save_processed(features)
    print(f"      SQLite -> {db}")
    print(f"      Parquet -> {processed}")
    print("=== ETL completado ===")


if __name__ == "__main__":
    run()
