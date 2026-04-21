# Comparacion de variantes logisticas - Modelo 1

## Objetivo

Comparar tres versiones del Modelo 1 para decidir cual sirve mejor como `xG` final:

- Logistica con `class_weight="balanced"`
- Logistica sin `class_weight`
- Logistica calibrada sobre la version sin pesos

## Resultados

| Variante | AUC | Log Loss | Brier | xG medio predicho | Tasa real de gol | Ratio sobreestimacion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logit_unweighted | 0.7813 | 0.2599 | 0.0733 | 0.1118 | 0.1062 | 1.05 |
| logit_calibrated | 0.7814 | 0.2601 | 0.0734 | 0.1121 | 0.1062 | 1.05 |
| logit_balanced | 0.7806 | 0.5201 | 0.1634 | 0.3945 | 0.1062 | 3.71 |

## Lectura

- Mejor AUC: `logit_calibrated`
- Mejor Log Loss: `logit_unweighted`
- Mejor Brier Score: `logit_unweighted`
- Variante mas cercana a la tasa real de gol: `logit_unweighted`

## Conclusion recomendada

Si el objetivo es `xG` como probabilidad creible, la mejor candidata debe priorizar calibracion (`Brier`, `Log Loss` y cercania a la tasa real) por encima de un pequeño beneficio en discriminacion.

En este contexto, la variante recomendada es la que mejor balancee:

1. Probabilidades realistas
2. Buena separacion entre tiros peligrosos y no peligrosos
3. Menor sobreestimacion agregada de goles

## Graficas

- `calibracion_variantes.png`
- `promedio_probabilidades.png`
