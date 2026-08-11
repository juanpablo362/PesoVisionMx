"""Orquestador ETL: extracción → transformación → carga."""

from __future__ import annotations

from src.etl.extract import extract_fx, save_raw
from src.etl.load import load_to_sqlite, save_processed
from src.etl.transform import build_features, clean_fx


def run() -> None:
    print("=== PesoVision ETL ===")

    print("[1/4] Extracción...")
    raw = extract_fx()
    raw_path = save_raw(raw)
    print(f"      {len(raw)} filas -> {raw_path}")

    print("[2/4] Limpieza...")
    clean, stats = clean_fx(raw)
    for key, value in stats.items():
        print(f"      {key}: {value}")

    print("[3/4] Feature engineering...")
    features = build_features(clean)
    print(f"      {len(features)} filas con target")

    print("[4/4] Carga...")
    db = load_to_sqlite(raw, clean, features)
    processed = save_processed(features)
    print(f"      SQLite -> {db}")
    print(f"      Parquet -> {processed}")
    print("=== ETL completado ===")


if __name__ == "__main__":
    run()
