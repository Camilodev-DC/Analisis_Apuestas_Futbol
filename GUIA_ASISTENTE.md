# Guía de Implementación: ML Premier League 2025-26

Este documento sirve como referencia técnica para el Asistente de IA (Arpa) en el desarrollo del Taller 2 de Machine Learning I.

## 1. Configuración y Conexión
- **API Base:** `https://premier.72-60-245-2.sslip.io`
- **Librerías principales:** `pandas`, `requests`, `scikit-learn`, `matplotlib`.
- **Endpoint principal:** `/matches?limit=500` para obtener el histórico de la temporada.

## 2. Modelado de Regresión Lineal (Goles Locales)
- **Objetivo:** Predecir `fthg` (Full Time Home Goals).
- **Features recomendadas:** 
  - `hs` (Home Shots)
  - `hst` (Home Shots on Target)
  - `hc` (Home Corners)
  - `hf` (Home Fouls)
- **Métricas de evaluación:** $R^2$, RMSE y análisis de residuos.

## 3. Modelado de Regresión Logística (Resultados H/D/A)
- **Objetivo:** Predecir `ftr` (Full Time Result: H=Home, D=Draw, A=Away).
- **Features de apuestas (Probabilidades implícitas):**
  - Convertir cuotas Bet365 (`b365h`, `b365d`, `b365a`) a probabilidades: $P = 1/odd$.
  - Normalizar para eliminar el margen de la casa.
- **Features de juego adicionales:** `SOTDiff` (`hst - ast`).
- **Métricas:** Accuracy, Matriz de Confusión y Odds Ratio.

## 4. Benchmarking
- El modelo debe compararse con el "baseline" de Bet365 (predecir el resultado con la cuota más baja).
- **Insight:** Los empates (D) suelen ser los más difíciles de clasificar debido a su baja frecuencia y naturaleza estadística.

## 5. Mejoras Propuestas
- Derivar features como `shots_ratio` o `expected_goals` si están disponibles.
- Evaluar el impacto del arbitraje y la ventaja de localía (Local win ~42.3%).
- Probar regularización (Ridge/Lasso) para mejorar la generalización.
