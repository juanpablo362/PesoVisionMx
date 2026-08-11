# Documento de funcionalidad — PesoVision

> Exportar este archivo a PDF/DOCX para la entrega académica.
> Completar las secciones marcadas con [COMPLETAR] tras Fase 3 y 4.

---

## 1. Nombre del proyecto

**PesoVision** — Sistema de predicción de la dirección del tipo de cambio USD/MXN.

## 2. Integrantes

- [COMPLETAR: Nombre 1]
- [COMPLETAR: Nombre 2]

## 3. Descripción general

PesoVision es una aplicación de ciencia de datos que procesa datos históricos del tipo de cambio dólar–peso mexicano (USD/MXN), los limpia mediante un pipeline ETL, entrena un modelo de aprendizaje supervisado para clasificar si el peso **subirá** o **bajará** al día siguiente, y presenta los resultados en un dashboard interactivo para apoyar la interpretación y la toma de decisiones informadas.

## 4. Problema de negocio

Las PYMES, importadores y personas físicas en México enfrentan incertidumbre cambiaria diaria. PesoVision no sustituye asesoría financiera profesional, pero demuestra cómo convertir datos públicos en **señales probabilísticas** que complementan el criterio humano.

**Pregunta clave:** ¿Mañana el tipo de cambio cerrará por encima o por debajo del cierre de hoy?

## 5. Funcionalidades principales

### 5.1 Extracción y almacenamiento (ETL)

- Descarga automática de USD/MXN desde Yahoo Finance.
- Limpieza de nulos, duplicados y outliers extremos.
- Almacenamiento en SQLite (`raw_fx_daily`, `fx_clean`, `fx_features`).

### 5.2 Modelado supervisado

- Clasificación binaria: subida (1) vs bajada (0).
- Comparación de Regresión Logística y Random Forest.
- Evaluación con split temporal y métricas F1, accuracy, ROC-AUC.

### 5.3 Dashboard

- Visualización histórica del tipo de cambio.
- Métricas del modelo e interpretación orientada a decisiones.

## 6. Tecnologías utilizadas

Python, pandas, yfinance, SQLite, scikit-learn, Streamlit, Plotly.

## 7. Cómo ejecutar el proyecto

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.etl.run_etl
python -m src.models.train
streamlit run src/dashboard/app.py
```

## 8. Resultados del modelo [COMPLETAR]

| Modelo | Accuracy | F1 | ROC-AUC |
|--------|----------|-----|---------|
| Logistic Regression | | | |
| Random Forest | | | |

**Modelo seleccionado:** [COMPLETAR + justificación]

## 9. Conclusiones [COMPLETAR]

- ¿El modelo supera el baseline (~50% accuracy)?
- ¿Qué variables fueron más relevantes?
- ¿Qué limitaciones tiene el enfoque?

## 10. Advertencia

Este proyecto es con fines educativos. Las predicciones no garantizan resultados en mercados reales.
