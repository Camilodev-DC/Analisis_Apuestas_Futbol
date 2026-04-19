# Reporte Modelo 1

## Enfoque

Modelo de Regresion Logistica para estimar `xG` a nivel de tiro. El target es binario (`is_goal`) y la salida del modelo es una probabilidad interpretable en `[0,1]`.

## Features finales

- Obligatorias: `distance_to_goal`, `angle_to_goal`
- Avanzadas: `is_big_chance`, `defensive_pressure`, `buildup_passes`, `buildup_unique_players`, `buildup_decentralized`, `first_touch`

## Metricas

- Train rows: 5758
- Test rows: 1440
- Goal rate test: 0.1062
- AUC-ROC: 0.7806
- Log Loss: 0.5201
- Brier Score: 0.1634
- Accuracy @ 0.5: 0.8653

## VIF

| Feature | VIF |
| --- | ---: |
| `buildup_unique_players` | 4.15 |
| `buildup_passes` | 2.62 |
| `buildup_decentralized` | 2.11 |
| `distance_to_goal` | 1.49 |
| `first_touch` | 1.27 |
| `is_big_chance` | 1.21 |
| `defensive_pressure` | 1.01 |
| `angle_to_goal` | 1.01 |

## Coeficientes positivos mas fuertes

| Feature | Coeficiente | Odds Ratio |
| --- | ---: | ---: |
| `is_big_chance` | 0.9880 | 2.686 |
| `distance_to_goal` | 0.2101 | 1.234 |
| `buildup_decentralized` | 0.0434 | 1.044 |
| `buildup_passes` | 0.0401 | 1.041 |

## Coeficientes negativos mas fuertes

| Feature | Coeficiente | Odds Ratio |
| --- | ---: | ---: |
| `angle_to_goal` | -0.2624 | 0.769 |
| `defensive_pressure` | -0.1481 | 0.862 |
| `buildup_unique_players` | -0.0994 | 0.905 |
| `first_touch` | -0.0146 | 0.986 |

## Lectura futbolistica

- `is_big_chance` aumenta con fuerza la probabilidad de gol, como era esperable para una ocasion clara.
- `defensive_pressure` reduce la probabilidad al capturar incomodidad de ejecucion.
- Las features de `buildup` ayudan a separar tiros creados en secuencias mas limpias frente a posesiones donde la defensa ya alcanzo a cerrar lineas.
- `first_touch` agrega una capa biomecanica y temporal de ejecucion del remate.

## Nota metodologica

- Se uso split temporal por `match_date`.
- Se excluyeron features con riesgo de leakage post-shot como `porteria_zone_*`.
- Se excluyeron derivadas geometricas como `dist_squared` y `dist_angle` para no inflar multicolinealidad.
- Se usa `class_weight="balanced"` por el desbalance natural entre goles y no goles.

## Logit

El resumen completo esta disponible en `logit_summary.txt`.
