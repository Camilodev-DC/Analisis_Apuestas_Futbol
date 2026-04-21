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
- AUC-ROC: 0.7813
- Log Loss: 0.2599
- Brier Score: 0.0733
- Accuracy @ 0.5: 0.9028
- Precision @ 0.5: 0.7826
- Recall @ 0.5: 0.1176
- F1 @ 0.5: 0.2045
- Baseline naive accuracy: 0.8938
- xG medio predicho: 0.1118
- Tasa real de gol: 0.1062

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
| `is_big_chance` | 1.0251 | 2.787 |
| `distance_to_goal` | 0.2855 | 1.330 |
| `buildup_decentralized` | 0.0110 | 1.011 |
| `buildup_passes` | -0.0017 | 0.998 |

## Coeficientes negativos mas fuertes

| Feature | Coeficiente | Odds Ratio |
| --- | ---: | ---: |
| `angle_to_goal` | -0.3189 | 0.727 |
| `defensive_pressure` | -0.1305 | 0.878 |
| `buildup_unique_players` | -0.0806 | 0.923 |
| `first_touch` | -0.0397 | 0.961 |

## Lectura futbolistica

- `is_big_chance` aumenta con fuerza la probabilidad de gol, como era esperable para una ocasion clara.
- `defensive_pressure` reduce la probabilidad al capturar incomodidad de ejecucion.
- Las features de `buildup` ayudan a separar tiros creados en secuencias mas limpias frente a posesiones donde la defensa ya alcanzo a cerrar lineas.
- `first_touch` agrega una capa biomecanica y temporal de ejecucion del remate.

## Matriz de confusion @ 0.5

| Real \ Pred | No gol | Gol |
| --- | ---: | ---: |
| No gol | 1282 | 5 |
| Gol | 135 | 18 |

Interpretacion:

- el baseline naive gana mucha `accuracy` porque casi todos los tiros son `no gol`
- aun asi, el modelo es mucho mas util porque entrega probabilidades y separa mejor los tiros de alta calidad

## Por que no dejamos `RightFoot`, `LeftFoot`, `Head` y zonas de disparo como features finales independientes

- `RightFoot`, `LeftFoot` y `Head` si se construyeron en el feature engineering y existen en la tabla procesada.
- No quedaron en el set final porque entre si forman un bloque muy redundante: casi todos los tiros pertenecen a una de esas categorias y eso introduce colinealidad estructural.
- En pruebas de VIF previas, estas variables elevaban inestabilidad e inflaban la interpretacion lineal del modelo.
- Su informacion no se perdio del todo: parte del contexto del remate queda absorbido por `is_big_chance`, `first_touch`, `defensive_pressure` y la propia geometria del tiro.

- Las zonas de disparo tipo `BoxCentre`, `OutOfBoxCentre` o `SmallBoxCentre` tambien aparecen dentro de los `qualifiers` y conceptualmente ya estaban representadas por las variables geometricas.
- En el feature engineering, esa idea de zona se resume principalmente en `distance_to_goal`, `angle_to_goal` y, en exploracion interna, en variables como `is_in_area` e `is_central`.
- No se dejaron como bloque final separado porque duplicaban la informacion espacial ya contenida en la geometria y empeoraban la parsimonia del modelo.

## Nota metodologica

- Se uso split temporal por `match_date`.
- Se excluyeron features con riesgo de leakage post-shot como `porteria_zone_*`.
- Se excluyeron derivadas geometricas como `dist_squared` y `dist_angle` para no inflar multicolinealidad.
- Se dejo la variante `unweighted` como modelo oficial porque la version `balanced` sobreestimaba fuertemente la probabilidad de gol.

## Logit

El resumen completo esta disponible en `logit_summary.txt`.
