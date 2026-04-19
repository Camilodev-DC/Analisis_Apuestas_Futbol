# Modelo 1 - xG logistico

Este folder contiene una version interpretable del Modelo 1 para estimar `xG` a nivel de tiro con Regresion Logistica.

## Objetivo

Construir un modelo simple, explicable y reproducible que combine:

- Variables obligatorias del taller
- Variables avanzadas de contexto tactico que sigan siendo razonables para un modelo logistico interpretable
- Un control explicito de multicolinealidad via VIF

## Lluvia de ideas - features avanzadas recomendadas

### 1. Defensive Pressure Proxy

- Nombre de la variable: `defensive_pressure`
- Logica futbolistica: un remate con dos o tres rivales cerrando espacio suele salir con peor perfil corporal, menos tiempo de ejecucion y menor precision final.
- Transformacion matematica: contar acciones del rival en el mismo `match_id` y `minute` que caigan dentro de una vecindad rectangular alrededor del tiro. La variable entra linealmente como conteo.

### 2. Buildup Pass Volume

- Nombre de la variable: `buildup_passes`
- Logica futbolistica: la calidad de la posesion previa cambia la limpieza de la ocasion. Un ataque muy directo y uno muy elaborado no generan el mismo tiro.
- Transformacion matematica: contar pases exitosos del mismo equipo en los `60` segundos previos al remate. Es una variable discreta y linealizable.

### 3. Buildup Unique Players

- Nombre de la variable: `buildup_unique_players`
- Logica futbolistica: si demasiados jugadores intervienen, la jugada puede ser mas colectiva pero tambien mas predecible y con defensa reorganizada. Si intervienen muy pocos, puede ser una transicion limpia.
- Transformacion matematica: numero de `player_id` unicos del mismo equipo en la ventana previa al tiro.

### 4. Buildup Decentralized Flag

- Nombre de la variable: `buildup_decentralized`
- Logica futbolistica: sirve para capturar si la posesion fue coral o concentrada sin exigir al modelo una no linealidad compleja.
- Transformacion matematica: dummy `1` si `buildup_unique_players > 3`, `0` en caso contrario.

### 5. First-Touch Finish

- Nombre de la variable: `first_touch`
- Logica futbolistica: un remate de primer toque suele venir con ventaja temporal sobre la defensa y el arquero, o al menos evita un control que enfrie la ocasion.
- Transformacion matematica: dummy extraida desde `qualifiers`, con valor `1` cuando el remate lleva etiqueta `FirstTouch`.

### 6. Big Chance Context

- Nombre de la variable: `is_big_chance`
- Logica futbolistica: resume que la ocasion fue catalogada por el feed como una llegada clara. No es geometria pura, sino contexto tactico de superioridad.
- Transformacion matematica: dummy desde `qualifiers`, `1` cuando aparece `BigChance`.

### 7. Altitude of Play

- Nombre de la variable: `altitude_of_play`
- Logica futbolistica: los equipos que viven mas arriba en el campo suelen rematar con la defensa rival mas hundida y con menos distancia media al arco.
- Transformacion matematica: promedio de `x` de eventos de recuperacion, pase e intervencion defensiva del equipo en ese partido.

## Variables que NO use en el modelo final

- `porteria_zone_*`: aporta informacion muy fuerte, pero es post-shot y puede acercarse a leakage si la tarea es xG pre-shot.
- `home_xg_debt_5`, `home_bias`, `home_win_rate`, `momentum`, `clutch_ratio`: son utiles para prediccion de partido, pero aportan menos para calidad intrinseca del tiro.
- `dist_squared`, `dist_angle`: elevan la colinealidad con `distance_to_goal`.

## Set final del modelo

Variables obligatorias:

- `distance_to_goal`
- `angle_to_goal`

Variables avanzadas elegidas:

- `is_big_chance`
- `defensive_pressure`
- `buildup_passes`
- `buildup_unique_players`
- `buildup_decentralized`
- `first_touch`

Este set mantiene VIF aceptable y funciona mejor que combinaciones mas cargadas con variables historicas de equipo en este dataset.

## Ejecucion

```bash
.venv/bin/python "modelo 1/train_modelo_1.py"
```

## Artefactos esperados

El script deja sus salidas en `modelo 1/artifacts/`:

- `modelo_1_xg_logistic.joblib`
- `selected_features.csv`
- `train_dataset_modelo_1.csv`
- `test_predictions_modelo_1.csv`
- `metrics.json`
- `coefficients.csv`
- `vif.csv`
- `logit_summary.txt`
- `reporte_modelo_1.md`
