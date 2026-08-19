"""Helpers puros del dashboard: explicación del día, paper trading y PDF."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from fpdf import FPDF

from src.models.common import FEATURE_COLS, TARGET_COL

FEATURE_LABELS = {
    "return_1d": "Retorno 1d",
    "return_5d": "Retorno 5d",
    "return_20d": "Retorno 20d",
    "ma_ratio": "Ratio MA 5/20",
    "volatility_20d": "Volatilidad 20d",
    "high_low_spread": "Rango high-low",
    "dxy_return_1d": "Retorno DXY 1d",
    "dxy_return_5d": "Retorno DXY 5d",
}

_PCT_FEATURES = {
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "high_low_spread",
    "dxy_return_1d",
    "dxy_return_5d",
}


def _fmt_feature(feat: str, value: float) -> str:
    if feat in _PCT_FEATURES:
        return f"{value * 100:+.2f}%"
    return f"{value:.4f}"


def explain_today(
    row: pd.Series,
    medians: pd.Series,
    importance: dict[str, float],
    n: int = 3,
) -> list[str]:
    ranked = sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)
    lines: list[str] = []
    for i, (feat, _) in enumerate(ranked[:n]):
        if feat not in row.index:
            continue
        val = float(row[feat])
        med = float(medians[feat]) if feat in medians.index else 0.0
        label = FEATURE_LABELS.get(feat, feat)
        val_txt = _fmt_feature(feat, val)
        med_txt = _fmt_feature(feat, med)
        if val > med:
            vs = "por encima"
        elif val < med:
            vs = "por debajo"
        else:
            vs = "en línea"
        if i == 0:
            lines.append(
                f"Hoy {label} está en {val_txt} y es la variable con más peso "
                f"({vs} de la mediana {med_txt})."
            )
        else:
            lines.append(f"{label}: {val_txt} ({vs} de la mediana {med_txt}).")
    return lines


def last_n_paper_trades(features: pd.DataFrame, model, n: int = 30) -> pd.DataFrame:
    subset = features.dropna(subset=[TARGET_COL]).tail(n).copy()
    y_pred = model.predict(subset[FEATURE_COLS])
    y_true = subset[TARGET_COL].astype(int).to_numpy()
    return pd.DataFrame(
        {
            "Fecha": pd.to_datetime(subset["date"]).dt.strftime("%Y-%m-%d"),
            "Predicho": np.where(y_pred == 1, "Sube", "Baja"),
            "Real": np.where(y_true == 1, "Sube", "Baja"),
            "Acierto": np.where(y_pred == y_true, "Sí", "No"),
        }
    )


def _pdf_text(text: str) -> str:
    mapping = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "Á": "A",
            "É": "E",
            "Í": "I",
            "Ó": "O",
            "Ú": "U",
            "ñ": "n",
            "Ñ": "N",
            "ü": "u",
            "¿": "",
            "¡": "",
        }
    )
    cleaned = text.translate(mapping).replace("—", "-").replace("–", "-")
    return cleaned.encode("latin-1", "replace").decode("latin-1")


def build_report_pdf(
    *,
    close: float,
    last_date: str,
    label: str,
    p_up: float,
    p_down: float,
    model_name: str,
    f1: float | None,
    auc: float | None,
    reasons: list[str],
    **kwargs,
) -> bytes:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    usable = 210 - 36

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(usable, 10, _pdf_text("PesoVision - reporte diario"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(
        usable,
        7,
        _pdf_text(f"USD/MXN  |  ultimo cierre {last_date}  |  horizonte 1 dia habil"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(usable, 8, "Senal", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(usable, 12, _pdf_text(label.upper()), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(usable, 7, f"Cierre: {close:.4f} MXN por USD", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(
        usable,
        7,
        f"Prob. de caida {p_down:.1%}   |   Prob. de subida {p_up:.1%}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    f1_txt = f"{f1:.3f}" if f1 is not None else "n/d"
    auc_txt = f"{auc:.3f}" if auc is not None else "n/d"
    pdf.cell(
        usable,
        7,
        _pdf_text(f"Modelo: {model_name}   |   F1 test {f1_txt}   |   ROC-AUC {auc_txt}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(usable, 8, "Por que hoy", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    if reasons:
        for line in reasons:
            pdf.multi_cell(usable, 6, _pdf_text(f"- {line}"))
    else:
        pdf.multi_cell(usable, 6, "No hay explicacion de variables disponible.")

    pdf.ln(8)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(90, 90, 90)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.multi_cell(
        usable,
        5,
        "Proyecto educativo. No es asesoria financiera ni recomendacion de inversion. "
        f"Generado {generated}.",
    )

    return bytes(pdf.output())
