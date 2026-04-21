# 03. Modelado Unificado

## 1. Pipeline real del proyecto

```text
data/raw/ -> feature engineering -> Modelo 1 (xG por tiro) -> agregacion por partido -> Modelo 2
```

Version aterrizada al repo:

- raw principal: `matches.csv`, `events_rich.csv`, `players.csv`, `player_history.csv`
- features de tiro: `data/processed/features_modelo1_a_j.csv`
- Modelo 1 implementado: `modelo 1/train_modelo_1.py`

## 2. Decision correcta para Modelo 1

### Que debe ser

- `Regresion Logistica`

### Por que

- el target observado es binario: `is_goal`
- xG es una probabilidad estimada `P(gol = 1 | features)`
- la salida natural debe quedar en `[0, 1]`

### Que no conviene como modelo principal

- regresion lineal para xG final, porque no es la formulacion natural del problema

### Variante elegida finalmente

- `Logistic Regression unweighted`

### Por que no usar `class_weight="balanced"` para el xG final

Hallazgo clave del proyecto:

- la version `balanced` servia para discriminar tiros peligrosos, pero **inflaba brutalmente las probabilidades**
- en test, la tasa real de gol fue cercana a `10.6%`
- la version `balanced` predijo un xG medio cercano a `39.4%`
- eso implica una sobreestimacion agregada de goles de aproximadamente `3.7x`

En cambio, la version `unweighted`:

- mantuvo `AUC` practicamente igual
- mejoro mucho `Log Loss` y `Brier Score`
- dejo el xG medio predicho cerca de `11.2%`, mucho mas pegado a la realidad

Conclusion:

- para `xG` importa mas la **calibracion probabilistica** que forzar balance entre clases
- por eso el `Modelo 1` oficial debe quedarse con `unweighted`

## 3. Set final actual del Modelo 1

Variables obligatorias:

- `distance_to_goal`
- `angle_to_goal`

Variables avanzadas escogidas por utilidad y VIF razonable:

- `is_big_chance`
- `defensive_pressure`
- `buildup_passes`
- `buildup_unique_players`
- `buildup_decentralized`
- `first_touch`

## 4. Estado del Modelo 1 ya construido

Ubicacion:

- `modelo 1/train_modelo_1.py`
- `modelo 1/artifacts/`
- `modelo 1/comparacion_variantes/`

Metricas del modelo logistico inicial:

- `AUC-ROC`: ~0.78
- `Log Loss`: ~0.52
- `Brier Score`: ~0.16

Comparacion de variantes encontrada despues:

- `logit_balanced`: AUC ~0.7806 | xG medio ~0.3945 | sobreestimacion ~3.71x
- `logit_unweighted`: AUC ~0.7813 | xG medio ~0.1118 | sobreestimacion ~1.05x
- `logit_calibrated`: AUC ~0.7814 | xG medio ~0.1121 | sobreestimacion ~1.05x

Interpretacion final:

- el problema principal no era la logistica, sino el uso de `balanced`
- la variante `unweighted` conserva la discriminacion y mejora mucho el realismo probabilistico
- la variante calibrada casi no mejora sobre `unweighted`, asi que no vale la pena complejizar el pipeline por ahora
- el Modelo 1 queda defendible para el taller y para mostrar en una pagina web

## 5. Feature engineering importante ya implementado

En `scripts/feature_engineering.py` ya quedaron construidas:

- base geometrica del tiro
- booleanas desde `qualifiers`
- `defensive_pressure`
- `buildup_passes`
- `buildup_unique_players`
- `buildup_decentralized`
- `porteria_zone`
- `home_xg_debt_5`
- `ppda`
- `pass_decentralization`
- `momentum`
- `home_win_rate`
- `home_bias`
- `altitude_of_play`
- `clutch_ratio`

## 6. Que debe alimentar el Modelo 2

Minimo razonable:

- cuotas y probabilidades implicitas
- xG agregado del Modelo 1 por equipo-partido, usando la variante `unweighted`
- forma reciente
- localia
- algunas features tacticas de partido si no generan leakage

## 7. Reglas para no enredar mas el proyecto

### Mantener

- una sola tabla principal de features por nivel de modelado
- una sola version “oficial” del modelo vigente
- documentacion centralizada

### Evitar

- duplicar EDA en varios archivos
- mantener simultaneamente versiones desactualizadas sin marcar
- crear scripts paralelos que hagan casi lo mismo

## 8. Prioridad practica

Orden recomendado a partir de ahora:

1. estabilizar `Modelo 1` con la variante `unweighted` como version oficial
2. generar `xg_per_match`
3. construir `matches_enriched`
4. entrenar `Modelo 2`
5. evaluar contra cuotas

## 9. Hallazgos que conviene mostrar en la futura pagina web

Si este contenido se transforma en web, los mensajes principales deberian ser:

1. El xG no depende solo de distancia y angulo; tambien depende del contexto de la jugada.
2. `is_big_chance`, `defensive_pressure` y las variables de `buildup` agregan informacion real al modelo.
3. Un clasificador puede tener buen `AUC` y aun asi producir probabilidades malas.
4. Para un xG creible, la calibracion importa tanto como la discriminacion.
5. En este proyecto, quitar `class_weight="balanced"` mejoro dramaticamente el realismo del xG sin sacrificar rendimiento.

Traduccion web:

- usar tarjetas cortas con estos hallazgos
- mostrar comparacion visual entre `balanced` y `unweighted`
- explicar con lenguaje simple que `xG` es una probabilidad y no solo un score
