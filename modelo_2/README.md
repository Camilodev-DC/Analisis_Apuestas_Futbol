# Modelo 2

Esta carpeta contiene el pipeline completo del Modelo 2 del proyecto, separado en dos tareas:

- Parte A: `Regresion Lineal` para predecir `total_goals`
- Parte B: `Regresion Logistica Multiclase` para predecir `ftr` (`H`, `D`, `A`)

## Principio metodologico clave

Aunque `matches.csv` trae tiros, corners, faltas y tarjetas del mismo partido, **esas variables no deben usarse directamente como features del partido a predecir** si queremos una validacion honesta.

Por eso este Modelo 2 usa:

- `odds` pre-partido
- `referee` transformado a sesgos numericos pre-match
- `rolling averages` de ultimos 5 partidos
- agregados de `events` por equipo y partido
- `team strength ratings` derivados de la tabla acumulada antes de cada partido
- `xG` agregado del Modelo 1

## Archivos principales

- `build_features_modelo_2.py`: construye la tabla de features de partido
- `train_modelo_2.py`: entrena las dos partes del modelo y genera artefactos
- `feature_search_modelo_2.py`: compara subsets honestos de features con CV temporal
- `artifacts/`: resultados, modelos, tablas y graficas

## Ejecucion

```bash
.venv/bin/python "modelo_2/train_modelo_2.py"
```

## Features principales

### Mercado

- `implied_prob_h`, `implied_prob_d`, `implied_prob_a`
- `bookmaker_spread_home`, `bookmaker_spread_draw`, `bookmaker_spread_away`
- `market_entropy`

### Arbitro

- `ref_home_win_rate_pre`
- `ref_avg_total_goals_pre`
- `ref_card_bias_pre`

### Fuerza de equipo

- `home_strength_rating`
- `away_strength_rating`
- `strength_rating_diff`

### Forma reciente desde `matches`

- goles a favor / en contra rolling
- tiros a puerta rolling
- corners rolling
- tarjetas rolling
- puntos rolling

### Forma reciente desde `events`

- `xg_avg5`
- `pass_accuracy_avg5`
- `progressive_passes_avg5`
- `big_chances_avg5`
- `high_press_pct_avg5`
- `crosses_avg5`

## Salidas esperadas

- `features_modelo_2.csv`
- `regression_cv_metrics.csv`
- `classification_cv_metrics.csv`
- `report_modelo_2.md`
- `feature_search_summary.md`
- `linear_model.joblib`
- `multiclass_logit_model.joblib`
- `confusion_matrix_multiclass.png`
- `regression_residuals.png`
- `accuracy_vs_bet365.png`
