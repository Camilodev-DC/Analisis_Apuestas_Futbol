# Reporte Modelo 2

## Objetivo

Modelo 2 resuelve dos tareas con validacion temporal honesta:

- Parte A: `Regresion Lineal` para `total_goals`
- Parte B: `Regresion Logistica Multiclase` para `ftr`

## Datos

- Partidos usados: 291
- Features construidas en tabla maestra: 45
- Fuente base: `matches.csv` + agregados de `events_rich.csv` + `xG` del Modelo 1

## Decisiones metodologicas importantes

- No se usan tiros, corners o tarjetas del mismo partido como features directas porque eso seria leakage.
- En su lugar se usan rolling averages pre-match.
- Los datos de eventos se agregan por equipo y partido.
- La fuerza de equipo se estima desde una tabla acumulada pre-match.
- La validacion se hace con `TimeSeriesSplit`, no con split aleatorio.

## Features clave

- Mercado: probabilidades implicitas, spreads de bookmakers, entropia del mercado
- Arbitro: `ref_home_win_rate_pre`, `ref_avg_total_goals_pre`, `ref_card_bias_pre`
- Strength: ratings pre-match del local y visitante
- Forma reciente: goles, tiros a puerta, corners, tarjetas, puntos
- Tactica reciente: `xg`, `pass_accuracy`, `progressive_passes`, `big_chances`, `high_press_pct`, `crosses`

## Seleccion final de variables por tarea

### Regresion lineal

Se quedo con el bloque `referee`, porque fue el mejor subset honesto en CV para `RMSE`.

- Features usadas: 3

### Logistica multiclase

Se quedo con el bloque `odds` solamente, porque fue el mejor subset honesto en CV para accuracy.

- Features usadas: 7

Las demas features si se construyeron y quedan disponibles en la tabla maestra, pero no mejoraron el desempeno CV en esta muestra.

## Parte A - Regresion Lineal (`total_goals`)

- RMSE medio CV: 1.6077 (+/- 0.1644)
- MAE medio CV: 1.2651 (+/- 0.1279)
- R2 medio CV: -0.0335 (+/- 0.0381)

## Parte B - Logistica Multiclase (`ftr`)

- Accuracy media CV: 0.4958 (+/- 0.0539)
- F1 macro media CV: 0.3767 (+/- 0.0379)
- Log Loss media CV: 1.0679 (+/- 0.0743)
- Accuracy media Bet365 en los mismos folds: 0.5083

## Benchmark

El taller fija como referencia aproximadamente `49.8%` de accuracy para Bet365.

Comparacion directa en nuestra muestra:

- Modelo 2: 0.4958
- Bet365: 0.5083

## Lectura

- En esta version honesta del pipeline, el modelo **no supera** a Bet365 en accuracy multiclase.
- Eso no invalida el ejercicio: muestra que el mercado es un baseline muy fuerte y que evitar leakage vuelve el problema realmente dificil.
- La parte de regresion lineal para `total_goals` tambien queda debil, incluso usando el mejor subset honesto encontrado (`referee`), con `R2` negativo.
- El valor tecnico del proyecto esta en tener un pipeline limpio, reproducible y listo para seguir iterando sobre features realmente pre-match.

## Interpretacion metodologica

- Si usaramos estadisticas del mismo partido como tiros, corners o tarjetas, la performance subiria mucho, pero seria leakage.
- Por eso este `Modelo 2` prioriza honestidad metodologica sobre inflar metricas artificialmente.
- La lectura correcta del resultado no es “el modelo falla”, sino “el benchmark del mercado es duro y nuestras features pre-match todavia no capturan suficiente ventaja incremental”.

## Graficas

- `confusion_matrix_multiclass.png`
- `regression_residuals.png`
- `accuracy_vs_bet365.png`
