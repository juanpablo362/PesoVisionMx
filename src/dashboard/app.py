"""Dashboard Streamlit — Fase 4 (esqueleto)."""

import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Streamlit ejecuta este archivo sin la raíz del proyecto en sys.path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DB_PATH, ROOT_DIR

MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"


@st.cache_data
def load_fx_clean() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM fx_clean ORDER BY date", conn)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="PesoVision", layout="wide")
    st.title("PesoVision — USD/MXN")
    st.caption("Predicción educativa de dirección del tipo de cambio (no es asesoría financiera).")

    if not DB_PATH.exists():
        st.error("Base de datos no encontrada. Ejecuta: `python -m src.etl.run_etl`")
        st.stop()

    df = load_fx_clean()
    df["date"] = pd.to_datetime(df["date"])

    col1, col2, col3 = st.columns(3)
    last = df.iloc[-1]
    col1.metric("Último cierre", f"{last['close']:.4f}")
    col2.metric("Cambio 1d", f"{last.get('return_1d', 0) * 100:.2f}%")
    col3.metric("Observaciones", len(df))

    fig = px.line(df, x="date", y="close", title="Tipo de cambio USD/MXN")
    st.plotly_chart(fig, use_container_width=True)

    bundle = load_model()
    if bundle is None:
        st.info("Entrena el modelo con: `python -m src.models.train`")
    else:
        st.subheader("Modelo")
        st.write(f"Algoritmo cargado: **{bundle['name']}**")


if __name__ == "__main__":
    main()

    st.sidebar.caption("v0.1.0-alpha | Status: Skeleton Active")
