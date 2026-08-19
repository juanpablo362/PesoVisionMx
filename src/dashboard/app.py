"""Dashboard Streamlit — PesoVision."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DB_PATH
from src.dashboard.insights import (
    FEATURE_LABELS,
    build_report_pdf,
    explain_today,
    last_n_paper_trades,
)
from src.models.common import (
    BEST_MODEL_PATH,
    FEATURE_COLS,
    METRICS_PATH,
    PREDICTIONS_PATH,
    ROC_PATH,
    TARGET_COL,
)

MODEL_PATH = BEST_MODEL_PATH


def _make_theme(**kwargs) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


THEMES = {
    "dark": _make_theme(
        bg="#1c1c1c",
        line="#3a3a3a",
        text="#ececec",
        muted="#9a9a9a",
        up="#3d8f6a",
        down="#c45c5c",
        primary="#5B8FA8",
        split_bg="#4a4a4a",
        usdmxn="#b8b8b8",
        dxy="#c9a227",
        highlight="#24333c",
        cm_low="#1c1c1c",
        cm_high="#6a6a6a",
        bar="#8a8a8a",
    ),
    "light": _make_theme(
        bg="#f6f5f2",
        line="#d8d4cc",
        text="#1c1c1c",
        muted="#6a6a6a",
        up="#2e7a58",
        down="#b44545",
        primary="#3d6f8a",
        split_bg="#e4e0d8",
        usdmxn="#4a4a4a",
        dxy="#9a7209",
        highlight="#e8eef2",
        cm_low="#f6f5f2",
        cm_high="#8a8a8a",
        bar="#6a6a6a",
    ),
}


def _css(theme: SimpleNamespace) -> str:
    return f"""
<style>
html, body, [class*="css"] {{
    font-family: "Segoe UI", system-ui, sans-serif;
}}

.stApp {{
    background: {theme.bg};
    color: {theme.text};
}}

header[data-testid="stHeader"] {{
    background: {theme.bg};
    height: 0;
}}

.block-container {{
    padding-top: 2.6rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}}

div[data-testid="stVerticalBlock"] > div {{
    gap: 0.35rem;
}}

div[data-testid="stTabs"] {{
    margin-top: 0.35rem;
}}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 0.4rem;
    margin-bottom: 0.4rem;
}}
div[data-testid="stTabs"] [data-baseweb="tab-panel"] {{
    padding-top: 1.15rem;
}}
div[data-testid="stHorizontalBlock"] {{
    gap: 1rem !important;
    align-items: center !important;
}}
[data-testid="stCaptionContainer"] {{
    margin-top: 0.6rem;
}}

h1.brand {{
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
    line-height: 1.25;
    color: {theme.text};
}}
.header-meta {{
    color: {theme.muted};
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.25;
}}
.header-rule {{
    border-bottom: 1px solid {theme.line};
    margin: 0.55rem 0 1.05rem;
}}
div[data-testid="stToggle"] {{
    display: flex;
    align-items: center;
    min-height: 1.6rem;
}}

.lead {{
    color: {theme.muted};
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0 0 1.15rem;
}}

.kpis {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border: 1px solid {theme.line};
    margin-bottom: 1.5rem;
}}
.kpis > div {{
    padding: 1rem 1.05rem;
    border-right: 1px solid {theme.line};
}}
.kpis > div:last-child {{ border-right: none; }}
.kpis .k {{
    color: {theme.muted};
    font-size: 0.75rem;
    margin-bottom: 0.35rem;
}}
.kpis .v {{
    font-family: ui-monospace, Consolas, monospace;
    font-size: 1.2rem;
    margin-top: 0.2rem;
}}
.up {{ color: {theme.up}; }}
.down {{ color: {theme.down}; }}

.split {{
    display: flex;
    height: 10px;
    background: {theme.split_bg};
}}
.split .a {{ background: {theme.down}; }}
.split .b {{ background: {theme.up}; }}
.split-cap {{
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: {theme.text};
    margin-top: 0.5rem;
}}

h3.sec {{
    font-size: 0.95rem;
    font-weight: 600;
    margin: 1.6rem 0 0.65rem;
}}

section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

#MainMenu, footer {{ visibility: hidden; }}
.stAppDeployButton, [data-testid="stToolbar"], button[kind="header"] {{
    display: none !important;
}}
div[data-testid="stStatusWidget"] {{ display: none !important; }}

div[data-testid="stTabs"] button {{
    color: {theme.muted};
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {theme.primary} !important;
}}

@media (max-width: 900px) {{
    .kpis {{ grid-template-columns: 1fr 1fr; }}
    .kpis > div:nth-child(2) {{ border-right: none; }}
}}
</style>
"""


def _theme() -> SimpleNamespace:
    return THEMES[st.session_state.get("theme", "dark")]


def _plotly_layout(fig: go.Figure, title: str = "", theme: SimpleNamespace | None = None) -> go.Figure:
    theme = theme or _theme()
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=theme.text, family="Segoe UI")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.muted, family="Segoe UI", size=13),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=theme.text)),
        margin=dict(l=16, r=16, t=48, b=24),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=theme.line, zeroline=False, linecolor=theme.line, tickfont=dict(size=12))
    fig.update_yaxes(gridcolor=theme.line, zeroline=False, linecolor=theme.line, tickfont=dict(size=12))
    return fig


@st.cache_data
def load_fx_clean() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM fx_clean ORDER BY date", conn)


@st.cache_data
def load_fx_features() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM fx_features ORDER BY date", conn)


@st.cache_data
def load_table_counts() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        rows = []
        for table in ("raw_fx_daily", "raw_dxy_daily", "fx_clean", "fx_features"):
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                continue
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            date_range = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table}").fetchone()
            rows.append(
                {"tabla": table, "filas": count, "desde": date_range[0], "hasta": date_range[1]}
            )
        return pd.DataFrame(rows)


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    if not PREDICTIONS_PATH.exists():
        return None
    df = pd.read_csv(PREDICTIONS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_roc() -> dict | None:
    if not ROC_PATH.exists():
        return None
    return json.loads(ROC_PATH.read_text(encoding="utf-8"))


def predict_next_day(features: pd.DataFrame, bundle: dict) -> tuple[str, float, float]:
    row = features.iloc[-1]
    X = row[FEATURE_COLS].values.reshape(1, -1)
    p_up = float(bundle["model"].predict_proba(X)[0, 1])
    p_down = 1.0 - p_up
    label = "Sube" if p_up >= 0.5 else "Baja"
    return label, p_up, p_down


def confidence_level(p_up: float) -> tuple[str, str]:
    strength = abs(p_up - 0.5)
    if strength >= 0.15:
        return (
            "alta",
            "La clase ganadora tiene al menos 15 puntos de ventaja sobre la otra. "
            "Sigue siendo una estimación estadística, no un hecho.",
        )
    if strength >= 0.05:
        return (
            "moderada",
            "Hay poca ventaja sobre el escenario contrario. El modelo se inclina a un lado, "
            "pero no de forma tajante. Conviene contrastar con otras fuentes.",
        )
    return (
        "débil",
        "Está muy cerca del 50/50: el modelo casi no distingue subida de bajada. "
        "No usar como única base para una decisión.",
    )


def interpret_prediction(label: str, p_up: float, volatility: float) -> dict:
    if label == "Sube":
        meaning = (
            "Estima que el cierre de mañana será mayor que el de hoy "
            "(el peso se deprecia frente al dólar)."
        )
    else:
        meaning = (
            "Estima que el cierre de mañana será menor o igual al de hoy "
            "(el peso se aprecia frente al dólar)."
        )
    conf_label, conf_explain = confidence_level(p_up)
    vol_note = (
        "La volatilidad de 20 días está alta; el movimiento diario puede ser más ruidoso."
        if volatility > 0.006
        else "La volatilidad de 20 días está en un rango habitual."
    )
    return {
        "meaning": meaning,
        "conf_label": conf_label,
        "conf_explain": conf_explain,
        "vol_note": vol_note,
    }


def _delta_class(value: float) -> str:
    if value > 0:
        return "up"
    if value < 0:
        return "down"
    return ""


def _fmt_pct(value: float) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _model_label(name: str) -> str:
    return {
        "logistic_regression": "Regresión logística",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
    }.get(name, name.replace("_", " ").title())


def _run_refresh(retrain: bool) -> tuple[bool, str]:
    commands = [[sys.executable, "-m", "src.etl.run_etl"]]
    if retrain:
        commands.append([sys.executable, "-m", "src.models.train"])
    chunks: list[str] = []
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        chunks.append((proc.stdout or "") + (proc.stderr or ""))
        if proc.returncode != 0:
            return False, "\n".join(chunks)
    return True, "\n".join(chunks)


def render_header(df_clean: pd.DataFrame, theme: SimpleNamespace) -> None:
    last_date = pd.to_datetime(df_clean.iloc[-1]["date"]).strftime("%d %b %Y")
    try:
        left, mid, right = st.columns([2.6, 3.6, 1.05], vertical_alignment="center")
    except TypeError:
        left, mid, right = st.columns([2.6, 3.6, 1.05])
    with left:
        st.markdown('<h1 class="brand">PesoVision — USD/MXN</h1>', unsafe_allow_html=True)
    with mid:
        st.markdown(
            f'<p class="header-meta">Último cierre {last_date} · horizonte 1 día hábil</p>',
            unsafe_allow_html=True,
        )
    with right:
        is_dark = st.toggle("Oscuro", value=st.session_state.theme == "dark")
        wanted = "dark" if is_dark else "light"
        if wanted != st.session_state.theme:
            st.session_state.theme = wanted
            st.rerun()
    st.markdown('<div class="header-rule"></div>', unsafe_allow_html=True)


def render_summary(
    df_clean: pd.DataFrame,
    df_features: pd.DataFrame,
    bundle: dict | None,
    metrics: dict | None,
    theme: SimpleNamespace,
) -> None:
    st.markdown(
        '<p class="lead"><b>Sube</b> = el USD/MXN cierra más alto (el peso se deprecia). '
        "<b>Baja</b> = cierra más bajo o igual. Las dos probabilidades suman 100%.</p>",
        unsafe_allow_html=True,
    )
    last = df_clean.iloc[-1]
    last_feat = df_features.iloc[-1]
    ret_1d = float(last.get("return_1d", 0) or 0)
    ret_5d = float(last_feat.get("return_5d", 0) or 0)
    vol = float(last_feat.get("volatility_20d", 0) or 0)
    last_date = pd.to_datetime(last["date"]).strftime("%d %b %Y")

    st.markdown(
        f"""
        <div class="kpis">
          <div>
            <div class="k">Cierre USD/MXN</div>
            <div class="v">{last["close"]:.4f}</div>
          </div>
          <div>
            <div class="k">Cambio 1 día</div>
            <div class="v {_delta_class(ret_1d)}">{_fmt_pct(ret_1d)}</div>
          </div>
          <div>
            <div class="k">Cambio 5 días</div>
            <div class="v {_delta_class(ret_5d)}">{_fmt_pct(ret_5d)}</div>
          </div>
          <div>
            <div class="k">Volatilidad 20d</div>
            <div class="v">{vol * 100:.2f}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if bundle is None:
        st.info("La predicción no está disponible en este momento.")
        return

    label, p_up, p_down = predict_next_day(df_features, bundle)
    info = interpret_prediction(label, p_up, vol)
    color = theme.up if label == "Sube" else theme.down
    down_w = max(1.0, p_down * 100)
    up_w = max(1.0, p_up * 100)

    best_name = (metrics or {}).get("best_model") or bundle.get("name", "")
    best_metrics = (metrics or {}).get("models", {}).get(best_name, {})
    importance = best_metrics.get("feature_importance", {})
    medians = df_features[FEATURE_COLS].median()
    reasons = explain_today(last_feat, medians, importance) if importance else []

    col_pred, col_mean = st.columns(2, gap="large")
    with col_pred:
        with st.container(border=True):
            feat_date = pd.to_datetime(last_feat["date"]).strftime("%d %b %Y")
            if pd.isna(last_feat.get(TARGET_COL)):
                st.caption(f"Siguiente día hábil · con el cierre del {feat_date}")
            else:
                st.caption(
                    f"Siguiente día hábil · última fila etiquetada ({feat_date}). "
                    "Actualiza datos para predecir a partir del último cierre."
                )
            st.markdown(
                f'<p class="dir" style="color:{color};font-size:3.4rem;font-weight:700;'
                f'letter-spacing:-0.03em;line-height:1.1;margin:0.1rem 0 0.9rem 0">{label}</p>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Prob. de caída", f"{p_down:.1%}")
            m2.metric("Prob. de subida", f"{p_up:.1%}")
            st.markdown(
                f"""
                <div class="split">
                  <div class="a" style="width:{down_w}%"></div>
                  <div class="b" style="width:{up_w}%"></div>
                </div>
                <div class="split-cap">
                  <span>Caída {p_down:.1%}</span>
                  <span>Subida {p_up:.1%}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Confianza:** {info['conf_label']}")

    with col_mean:
        with st.container(border=True):
            st.caption("Qué significa")
            st.write(info["meaning"])
            st.caption(f"{info['conf_explain']} {info['vol_note']}")
            st.caption("Proyecto educativo. No es asesoría financiera.")

    if reasons:
        st.markdown('<h3 class="sec">Por qué hoy</h3>', unsafe_allow_html=True)
        for line in reasons:
            st.markdown(f"- {line}")

    test_m = best_metrics.get("test", {})
    pdf_bytes = build_report_pdf(
        close=float(last["close"]),
        last_date=last_date,
        label=label,
        p_up=p_up,
        p_down=p_down,
        model_name=_model_label(best_name),
        f1=test_m.get("f1"),
        auc=test_m.get("roc_auc"),
        reasons=reasons,
    )
    st.download_button(
        "Descargar reporte (PDF)",
        data=pdf_bytes,
        file_name="pesovision_reporte.pdf",
        mime="application/pdf",
    )

    st.markdown('<h3 class="sec">Últimos 30 días</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lead">Paper trading educativo: predicción del modelo vs cierre real. '
        "El recorte suele caer en el periodo de prueba. No es una estrategia de inversión.</p>",
        unsafe_allow_html=True,
    )
    paper = last_n_paper_trades(df_features, bundle["model"], n=30)
    hits = int((paper["Acierto"] == "Sí").sum())
    st.caption(f"Aciertos: {hits}/{len(paper)} ({hits / len(paper):.0%}).")
    st.dataframe(paper, use_container_width=True, hide_index=True)


def render_history(
    df_clean: pd.DataFrame,
    df_features: pd.DataFrame,
    preds: pd.DataFrame | None,
    theme: SimpleNamespace,
) -> None:
    st.markdown(
        '<p class="lead">Puntos en la serie: acierto o error del modelo en el <b>periodo de prueba</b> '
        "(20% más reciente; no se usó para entrenar). El DXY va en el eje derecho.</p>",
        unsafe_allow_html=True,
    )
    df = df_clean.copy()
    df["date"] = pd.to_datetime(df["date"])
    feats = df_features.copy()
    feats["date"] = pd.to_datetime(feats["date"])
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    presets = {
        "Último año": max_date - timedelta(days=365),
        "Últimos 3 años": max_date - timedelta(days=365 * 3),
        "Todo el histórico": min_date,
        "Solo periodo de prueba": None,
    }
    col_a, col_b, col_c, col_d = st.columns([1.4, 1.1, 1.1, 1.1])
    preset = col_a.selectbox("Periodo", list(presets.keys()), index=0)
    show_dxy = col_d.checkbox("Mostrar DXY", value=True)

    if preset == "Solo periodo de prueba" and preds is not None and not preds.empty:
        default_start = preds["date"].min().date()
        default_end = preds["date"].max().date()
    else:
        default_start = max(min_date, presets[preset] or min_date)
        default_end = max_date

    start = col_b.date_input("Desde", default_start, min_value=min_date, max_value=max_date)
    end = col_c.date_input("Hasta", default_end, min_value=min_date, max_value=max_date)
    if start > end:
        st.warning("La fecha inicial no puede ser posterior a la final.")
        return

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    filtered = df.loc[mask]
    if filtered.empty:
        st.warning("No hay observaciones en ese rango.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered["date"],
            y=filtered["close"],
            mode="lines",
            name="USD/MXN",
            line=dict(color=theme.usdmxn, width=1.6),
            hovertemplate="%{x|%d %b %Y}<br>%{y:.4f}<extra></extra>",
        )
    )

    n_ok = n_bad = 0
    if preds is not None:
        preds_f = preds[(preds["date"].dt.date >= start) & (preds["date"].dt.date <= end)]
        if not preds_f.empty:
            correct = preds_f[preds_f["correct"] == 1]
            wrong = preds_f[preds_f["correct"] == 0]
            n_ok, n_bad = len(correct), len(wrong)
            step = max(1, len(preds_f) // 120)
            fig.add_trace(
                go.Scatter(
                    x=correct["date"].iloc[::step],
                    y=correct["close"].iloc[::step],
                    mode="markers",
                    name="Acierto",
                    marker=dict(color=theme.up, size=5, opacity=0.7, symbol="circle"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=wrong["date"].iloc[::step],
                    y=wrong["close"].iloc[::step],
                    mode="markers",
                    name="Error",
                    marker=dict(color=theme.down, size=6, opacity=0.75, symbol="x"),
                )
            )

    if show_dxy and "dxy_close" in feats.columns:
        dxy_mask = (feats["date"].dt.date >= start) & (feats["date"].dt.date <= end)
        dxy = feats.loc[dxy_mask, ["date", "dxy_close"]].dropna()
        if not dxy.empty:
            fig.add_trace(
                go.Scatter(
                    x=dxy["date"],
                    y=dxy["dxy_close"],
                    mode="lines",
                    name="DXY",
                    yaxis="y2",
                    line=dict(color=theme.dxy, width=1.4, dash="dot"),
                    hovertemplate="%{x|%d %b %Y}<br>DXY %{y:.2f}<extra></extra>",
                )
            )
            fig.update_layout(
                yaxis2=dict(
                    title=dict(text="DXY", font=dict(color=theme.dxy)),
                    overlaying="y",
                    side="right",
                    showgrid=False,
                    tickfont=dict(color=theme.dxy, size=12),
                )
            )

    _plotly_layout(fig, "Cierre USD/MXN", theme)
    fig.update_layout(
        yaxis_title="MXN por USD",
        xaxis_title=None,
        height=440,
        dragmode="zoom",
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

    acc_txt = (
        f"Aciertos en el recorte: {n_ok}/{n_ok + n_bad} ({n_ok / (n_ok + n_bad):.0%})."
        if n_ok + n_bad
        else "Este recorte no incluye el periodo de prueba; no hay puntos de acierto/error."
    )
    st.caption(
        f"{len(filtered):,} observaciones. "
        f"Mín {filtered['close'].min():.2f} / máx {filtered['close'].max():.2f}. {acc_txt}"
    )


def render_model_tab(metrics: dict | None, roc: dict | None, bundle: dict | None, theme: SimpleNamespace) -> None:
    if metrics is None:
        st.info("Las métricas del modelo no están disponibles en este momento.")
        return

    best = metrics.get("best_model", "N/A")
    best_f1 = metrics["models"].get(best, {}).get("test", {}).get("f1")
    if best_f1 is not None:
        st.success(
            f"Ganador por F1 en test: {_model_label(best)} ({best_f1:.3f}). "
            "El resto de la tabla es la comparación completa."
        )
    else:
        st.markdown(
            f'<p class="lead">Ganador por F1 en test: <b>{_model_label(best)}</b>.</p>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<p class="lead">Tres clasificadores con split temporal 80/20. '
        "Un accuracy de ~50% equivale a adivinar al azar; el FX diario es ruidoso.</p>",
        unsafe_allow_html=True,
    )

    rows = []
    for name, m in metrics["models"].items():
        rows.append(
            {
                "Modelo": _model_label(name),
                "CV F1": m["cv_f1_mean"],
                "Train Acc": m["train"]["accuracy"],
                "Train F1": m["train"]["f1"],
                "Test Acc": m["test"]["accuracy"],
                "Test F1": m["test"]["f1"],
                "Test ROC-AUC": m["test"]["roc_auc"],
            }
        )
    table = pd.DataFrame(rows)

    def _highlight_winner(row):
        if row["Modelo"] == _model_label(best):
            return [f"background-color: {theme.highlight}"] * len(row)
        return [""] * len(row)

    st.dataframe(
        table.style.format(
            {
                "CV F1": "{:.3f}",
                "Train Acc": "{:.3f}",
                "Train F1": "{:.3f}",
                "Test Acc": "{:.3f}",
                "Test F1": "{:.3f}",
                "Test ROC-AUC": "{:.3f}",
            }
        ).apply(_highlight_winner, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    if best and best in metrics["models"]:
        cm = metrics["models"][best]["test"]["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Real baja", "Real sube"], columns=["Pred baja", "Pred sube"])
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale=[theme.cm_low, theme.cm_high],
            title=f"Matriz de confusión (test) — {_model_label(best)}",
        )
        fig_cm.update_coloraxes(showscale=False)
        _plotly_layout(fig_cm, fig_cm.layout.title.text, theme)
        fig_cm.update_layout(height=400, autosize=True)
        left.plotly_chart(fig_cm, use_container_width=True)

    if roc:
        fig_roc = go.Figure()
        colors = {
            "logistic_regression": "#c8c8c8" if theme.bg == THEMES["dark"].bg else "#5a5a5a",
            "random_forest": theme.primary,
            "gradient_boosting": theme.dxy,
        }
        for name, curve in roc.items():
            fig_roc.add_trace(
                go.Scatter(
                    x=curve["fpr"],
                    y=curve["tpr"],
                    mode="lines",
                    name=_model_label(name),
                    line=dict(color=colors.get(name, theme.text), width=2),
                )
            )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Azar",
                line=dict(dash="dash", color=theme.muted, width=1),
            )
        )
        _plotly_layout(fig_roc, "Curva ROC (test)", theme)
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=400, autosize=True)
        right.plotly_chart(fig_roc, use_container_width=True)

    if bundle and metrics.get("best_model"):
        imp = metrics["models"][metrics["best_model"]].get("feature_importance", {})
        if imp:
            imp_df = pd.DataFrame(
                {
                    "feature": [FEATURE_LABELS.get(k, k) for k in imp],
                    "importance": list(imp.values()),
                }
            ).sort_values("importance", ascending=True)
            fig_imp = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Peso de variables — {_model_label(best)}",
            )
            fig_imp.update_traces(marker_color=theme.bar)
            _plotly_layout(fig_imp, fig_imp.layout.title.text, theme)
            fig_imp.update_layout(xaxis_title=None, yaxis_title=None, height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<h3 class="sec">Train vs test</h3>', unsafe_allow_html=True)
    st.markdown(
        '<p class="lead">Si F1 de train es mucho mayor que F1 de test, el modelo memorizó ruido. '
        "Si ambos rondan 0.5, no hay señal útil o el problema es difícil.</p>",
        unsafe_allow_html=True,
    )
    for name, m in metrics["models"].items():
        gap = m["train"]["f1"] - m["test"]["f1"]
        if gap > 0.08:
            note = "posible overfitting (F1 train ≫ test)"
        elif m["test"]["f1"] < 0.52:
            note = "rendimiento cercano al azar en test"
        else:
            note = "generalización razonable"
        st.markdown(f"- **{_model_label(name)}**: gap F1 train–test = `{gap:.3f}` → {note}")


def render_etl_tab(df_clean: pd.DataFrame, metrics: dict | None) -> None:
    n = len(df_clean)
    start = pd.to_datetime(df_clean.iloc[0]["date"]).date()
    end = pd.to_datetime(df_clean.iloc[-1]["date"]).date()
    best = metrics.get("best_model", "sin entrenar") if metrics else "sin entrenar"

    st.markdown(
        '<p class="lead">Origen de la serie, reglas de limpieza y almacenamiento en SQLite. '
        "Esta pestaña documenta el ETL (fases 1 y 2).</p>",
        unsafe_allow_html=True,
    )

    st.markdown('<h3 class="sec">Actualizar datos</h3>', unsafe_allow_html=True)
    st.caption(
        "Vuelve a descargar USD/MXN y DXY desde Yahoo Finance. Tarda alrededor de un minuto "
        "y necesita internet. Marca reentrenar si también quieres ajustar los modelos."
    )
    retrain = st.checkbox("Reentrenar modelo después de extraer", value=False)
    if st.button("Actualizar datos"):
        with st.spinner("Actualizando datos..."):
            ok, log = _run_refresh(retrain)
        if ok:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Datos actualizados.")
            st.rerun()
        else:
            st.error("No se pudo actualizar.")
            st.code(log[-4000:] if log else "Sin salida.")

    st.markdown('<h3 class="sec">Fuente de datos</h3>', unsafe_allow_html=True)
    fuentes = pd.DataFrame(
        [
            {
                "Fuente": "Yahoo Finance",
                "Serie": "USDMXN=X",
                "Variables": "Open, High, Low, Close, Volume",
                "Frecuencia": "Diaria",
                "Uso": "Serie principal del tipo de cambio",
            },
            {
                "Fuente": "Yahoo Finance",
                "Serie": "DX-Y.NYB (fallback DX=F)",
                "Variables": "Close → retornos 1d y 5d",
                "Frecuencia": "Diaria",
                "Uso": "Índice del dólar (DXY) como contexto global",
            },
        ]
    )
    st.dataframe(fuentes, use_container_width=True, hide_index=True)
    st.caption(
        "Se eligió Yahoo Finance porque ofrece histórico diario gratuito y reproducible. "
        "FRED y Banxico siguen como fuentes opcionales; DXY se alinea por fecha al USD/MXN."
    )

    origen = pd.DataFrame(
        [
            {"Campo": "Observaciones", "Valor": f"{n:,} días hábiles"},
            {"Campo": "Rango", "Valor": f"{start} a {end}"},
            {"Campo": "Modelo en uso", "Valor": best.replace("_", " ")},
            {"Campo": "Almacén", "Valor": "SQLite · data/pesovision.db"},
        ]
    )
    st.dataframe(origen, use_container_width=True, hide_index=True)

    st.markdown('<h3 class="sec">Limpieza</h3>', unsafe_allow_html=True)
    st.markdown(
        """
- **Duplicados:** una fila por fecha; se conserva la última extracción.
- **Nulos en Close:** se rellenan solo si el hueco es de 3 días o menos; el resto se elimina.
- **Outliers:** se descartan retornos diarios mayores a 15% (error de dato o evento anómalo).
- **Orden:** serie ascendente por fecha; el split del modelo es temporal, no aleatorio.
        """
    )

    st.markdown('<h3 class="sec">Tablas SQLite</h3>', unsafe_allow_html=True)
    stats = load_table_counts().rename(
        columns={"tabla": "Tabla", "filas": "Filas", "desde": "Desde", "hasta": "Hasta"}
    )
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.caption(
        "`raw_fx_daily` y `raw_dxy_daily` son extracciones crudas; `fx_clean` una fila por día hábil; "
        "`fx_features` las variables (incl. DXY) y el target `direction_next_day`. "
        "La última fila no tiene target: sirve para predecir el día siguiente "
        "sin reentrenar (Actualizar datos)."
    )

    st.markdown('<h3 class="sec">Pipeline</h3>', unsafe_allow_html=True)
    st.markdown(
        """
1. Extraer `USDMXN=X` y DXY a `data/raw/`
2. Limpiar nulos, duplicados y outliers
3. Unir DXY por fecha, calcular retornos, medias, volatilidad y target (la última fila se guarda sin etiqueta para inferencia)
4. Cargar en SQLite
        """
    )
    st.code(
        "python -m src.etl.run_etl\n"
        "python -m src.models.train\n"
        "streamlit run src/dashboard/app.py",
        language="bash",
    )


def main() -> None:
    st.set_page_config(
        page_title="PesoVision",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get help": None, "Report a bug": None, "About": None},
    )
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    theme = _theme()
    st.markdown(_css(theme), unsafe_allow_html=True)

    if not DB_PATH.exists():
        st.error("No hay datos disponibles. Intenta más tarde.")
        st.stop()

    df_clean = load_fx_clean()
    df_features = load_fx_features()
    bundle = load_model_bundle()
    metrics = load_metrics()
    preds = load_predictions()
    roc = load_roc()

    render_header(df_clean, theme)

    tab_resumen, tab_hist, tab_modelo, tab_etl = st.tabs(
        [
            "Predicción del día",
            "Serie USD/MXN",
            "Métricas del clasificador",
            "Datos y ETL",
        ]
    )
    with tab_resumen:
        render_summary(df_clean, df_features, bundle, metrics, theme)
    with tab_hist:
        render_history(df_clean, df_features, preds, theme)
    with tab_modelo:
        render_model_tab(metrics, roc, bundle, theme)
    with tab_etl:
        render_etl_tab(df_clean, metrics)


if __name__ == "__main__":
    main()
