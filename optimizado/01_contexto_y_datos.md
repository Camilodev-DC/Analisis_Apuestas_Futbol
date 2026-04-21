# 01. Contexto y Datos

## 1. Problema del proyecto

El proyecto busca predecir futbol mejor que un baseline ingenuo, separando dos tareas:

- `Modelo 1`: estimar `xG` a nivel de tiro, es decir, la probabilidad de que un remate termine en gol.
- `Modelo 2`: predecir el resultado del partido usando cuotas, forma, contexto tactico y el output agregado del Modelo 1.

La Premier League tiene:

- 20 equipos
- 380 partidos por temporada
- ventaja local real
- alto ruido e incertidumbre en eventos de gol

## 2. Conceptos minimos que de verdad importan

- `Home / Away`: el local suele ganar mas y producir mas volumen ofensivo.
- `Shots on target`: son mejor señal de peligro que los tiros totales.
- `Odds`: las cuotas resumen informacion de mercado y son baseline obligatorio para el Modelo 2.
- `xG`: no es un dato observado; es una probabilidad estimada a partir de `gol = 1` o `no gol = 0`.
- `Qualifiers`: en eventos son la fuente mas rica para contexto del tiro.

## 3. Inventario util de datos

### `data/raw/matches.csv`

Unidad: partido

Contiene:

- identificacion (`id`, `date`, `home_team`, `away_team`, `referee`)
- resultado (`fthg`, `ftag`, `ftr`, `hthg`, `htag`, `htr`)
- estadisticas agregadas (`hs`, `hst`, `hc`, `hy`, etc.)
- cuotas y probabilidades implicitas (`b365*`, `bw*`, `max*`, `avg*`, `implied_prob_*`)

Uso principal:

- baseline de mercado
- features de partido
- target del Modelo 2

### `data/raw/events.csv`

Unidad: evento

Contiene:

- evento por evento (`event_type`, `outcome`, `team_name`, `player_id`)
- coordenadas (`x`, `y`, `end_x`, `end_y`)
- campos de remate (`goal_mouth_y`, `goal_mouth_z`)
- flags (`is_touch`, `is_shot`, `is_goal`)

Uso principal:

- geometria del tiro
- extraccion de tiros
- EDA espacial

### `data/raw/events_rich.csv`

Unidad: evento enriquecido

Es la version util para modelado porque agrega:

- `qualifiers`
- `home_team`, `away_team`, `match_date`, `referee`

Uso principal:

- feature engineering del Modelo 1
- features tacticas y contextuales

### `data/raw/players.csv`

Unidad: jugador temporada

Contiene:

- estado del jugador
- minutos, goles, asistencias
- expected stats FPL (`xG`, `xA`, `xGI`)
- indices (`influence`, `creativity`, `threat`)

Uso principal:

- calidad de plantilla
- features agregadas de jugadores
- soporte para Modelo 2

### `data/raw/player_history.csv`

Unidad: jugador-jornada

Contiene:

- rendimiento por gameweek
- `was_home`
- `minutes`
- `expected_goals`, `expected_assists`

Uso principal:

- forma reciente
- features temporales por jugador

Limitacion:

- la API esta degradada y trae muchas menos filas de las esperadas

## 4. Datos procesados importantes

- `data/processed/features_modelo1_a_j.csv`: tabla principal de tiros con features base y avanzadas del Modelo 1
- `data/processed/vif_modelo1_a_j.csv`: VIF del conjunto ampliado
- `data/processed/player_id_map.json`: puente entre IDs de eventos y Fantasy

## 5. Incidencias reales que no se deben olvidar

### IDs incompatibles entre fuentes

- `events` usa IDs tipo WhoScored / Opta
- `players` y `player_history` usan IDs de Fantasy Premier League

Solucion actual:

- usar `data/processed/player_id_map.json`

### Dataset historico de jugadores incompleto

- `player_history.csv` esta truncado por degradacion de API
- sirve para insight y algunas features, pero no para confiar ciegamente en ventanas largas

### Dos versiones practicas de eventos

- `events.csv` sirve para analisis base
- `events_rich.csv` es la que conviene usar para features, porque trae `qualifiers`

## 6. Regla operativa para trabajar sin romper consistencia

- Para contexto y targets de partido, usar `matches.csv`
- Para xG y features de tiro, usar `events_rich.csv`
- Para cruces entre eventos y FPL, pasar siempre por `player_id_map.json`
- Para evitar leakage en Modelo 1, no usar variables post-shot como definicion principal del modelo final
