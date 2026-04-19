# Diccionario de Datos — Proyecto ML Premier League

Este documento describe cada archivo en `data/raw/` y el detalle de cada columna.

---

## Inventario de Archivos

| Archivo | Fuente | Registros | Tamaño | Descripción |
|---------|--------|-----------|--------|-------------|
| `matches.json` | football-data.co.uk | 291 partidos | 142 KB | Resultados, estadísticas y cuotas (formato JSON) |
| `matches.csv` | football-data.co.uk | 291 partidos | 48 KB | Misma info que matches.json pero en CSV |
| `players.csv` | Fantasy Premier League | 822 jugadores | 135 KB | Estadísticas acumuladas de cada jugador |
| `player_history.csv` | Fantasy Premier League | 1,499 registros | 129 KB | Rendimiento jornada por jornada |
| `events.csv` | WhoScored (scraping) | 444,252 eventos* | 201 MB | Cada acción en el campo con coordenadas x,y |

> **\*Nota sobre events.csv:** La API reporta 444,252 eventos totales en su metadata. El archivo descargado contiene los datos útiles consolidados partido por partido.

---

## 1. `matches.json` / `matches.csv` — Datos de Partidos

**Fuente:** football-data.co.uk 🇬🇧 | **Registros:** 291 | **Columnas:** 41

### Identificación del Partido
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | int | Identificador único del partido |
| `date` | str | Fecha del partido (DD/MM/YYYY) |
| `time` | str | Hora de inicio (HH:MM) |
| `home_team` | str | Nombre del equipo local |
| `away_team` | str | Nombre del equipo visitante |
| `referee` | str | Nombre del árbitro principal |

### Resultado Final
| Columna | Tipo | Descripción | Rango |
|---------|------|-------------|-------|
| `fthg` | int | **Full Time Home Goals** — Goles del local al final | 0–5, media: 1.52 |
| `ftag` | int | **Full Time Away Goals** — Goles del visitante al final | 0–5, media: 1.26 |
| `ftr` | str | **Full Time Result** — H (Home 42.3%), D (Draw 26.1%), A (Away 31.6%) | H/D/A |
| `hthg` | int | **Half Time Home Goals** — Goles del local al medio tiempo | 0–3, media: 0.70 |
| `htag` | int | **Half Time Away Goals** — Goles del visitante al medio tiempo | 0–3, media: 0.52 |
| `htr` | str | **Half Time Result** | H/D/A |
| `total_goals` | int | Suma total de goles en el partido | 0–9, media: 2.77 |
| `goal_diff` | int | Diferencia de goles (home − away) | −4 a +5 |

### Estadísticas de Juego
| Columna | Tipo | Descripción | Media Local | Media Visitante |
|---------|------|-------------|-------------|-----------------|
| `hs` / `as_` | int | Tiros totales | 13.6 | 11.0 |
| `hst` / `ast` | int | Tiros a puerta | 4.4 | 3.8 |
| `hf` / `af` | int | Faltas cometidas | 10.6 | 11.0 |
| `hc` / `ac` | int | Córners | 5.3 | 4.6 |
| `hy` / `ay` | int | Tarjetas amarillas | 1.7 | 2.1 |
| `hr` / `ar` | int | Tarjetas rojas | 0.06 | 0.05 |

### Cuotas de Apuestas
| Columna | Tipo | Descripción | Media |
|---------|------|-------------|-------|
| `b365h` / `b365d` / `b365a` | float | Cuotas **Bet365** (Home/Draw/Away) | 2.65 / 3.99 / 3.95 |
| `bwh` / `bwd` / `bwa` | float | Cuotas **BetWay** | 2.64 / 3.98 / 3.88 |
| `maxh` / `maxd` / `maxa` | float | Cuota **máxima** del mercado | 2.79 / 4.13 / 4.30 |
| `avgh` / `avgd` / `avga` | float | Cuota **promedio** del mercado | 2.67 / 3.96 / 4.01 |

### Probabilidades Implícitas (Derivadas)
| Columna | Tipo | Descripción | Media |
|---------|------|-------------|-------|
| `implied_prob_h` | float | Probabilidad implícita de victoria local | 0.46 |
| `implied_prob_d` | float | Probabilidad implícita de empate | 0.26 |
| `implied_prob_a` | float | Probabilidad implícita de victoria visitante | 0.34 |

---

## 2. `players.csv` — Datos de Jugadores (Fantasy PL)

**Fuente:** Fantasy Premier League API ⚽ | **Registros:** 822 | **Columnas:** 37

### Identificación
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | int | ID único del jugador en FPL |
| `first_name` | str | Nombre |
| `second_name` | str | Apellido |
| `web_name` | str | Nombre corto (como aparece en Fantasy) |
| `team` | str | Nombre del equipo |
| `team_short` | str | Abreviatura del equipo (ej: MCI, ARS) |
| `position` | str | Posición: GKP (portero), DEF, MID, FWD |
| `price` | float | Precio en Fantasy (en millones, ej: 14.6) |
| `status` | str | Estado: `a` (disponible), `i` (lesionado), `s` (suspendido), `u` (no disponible) |

### Rendimiento Acumulado
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `total_points` | int | Puntos totales en Fantasy |
| `minutes` | int | Minutos jugados en la temporada |
| `starts` | int | Partidos como titular |
| `goals_scored` | int | Goles marcados |
| `assists` | int | Asistencias dadas |
| `clean_sheets` | int | Porterías invictas (solo relevante para DEF/GKP) |
| `goals_conceded` | int | Goles recibidos |
| `yellow_cards` | int | Tarjetas amarillas |
| `red_cards` | int | Tarjetas rojas |
| `saves` | int | Atajadas (solo porteros) |
| `bonus` | int | Puntos bonus FPL |
| `bps` | int | Bonus Points System (sistema interno de FPL) |

### Expected Stats (Métricas Avanzadas)
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `xG` | float | **Expected Goals** — Goles esperados según calidad de los tiros |
| `xA` | float | **Expected Assists** — Asistencias esperadas |
| `xGI` | float | **Expected Goal Involvement** (xG + xA) |
| `xG_per90` | float | xG por cada 90 minutos |
| `xA_per90` | float | xA por cada 90 minutos |

### Índices de Rendimiento FPL
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `influence` | float | Índice de influencia en los partidos |
| `creativity` | float | Índice de creatividad (chances creados) |
| `threat` | float | Índice de amenaza ofensiva |
| `ict_index` | float | ICT combinado (Influence + Creativity + Threat) |
| `form` | float | Forma reciente (puntos promedio últimas jornadas) |
| `points_per_game` | float | Puntos promedio por partido |

### Mercado de Transferencias FPL
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `selected_by_percent` | float | % de managers Fantasy que lo tienen |
| `transfers_in` | int | Fichajes entrantes (managers que lo compraron) |
| `transfers_out` | int | Fichajes salientes (managers que lo vendieron) |
| `chance_of_playing_next_round` | int | Probabilidad de jugar la próxima jornada (0–100) |
| `news` | str | Noticias/lesiones del jugador |

---

## 3. `player_history.csv` — Historial por Jornada

**Fuente:** Fantasy Premier League API ⚽ | **Registros:** 1,499 | **Columnas:** 20

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `player_id` | int | ID del jugador (FK a players.csv) |
| `web_name` | str | Nombre corto del jugador |
| `team` | str | Equipo del jugador |
| `gameweek` | int | Número de jornada (1–38) |
| `opponent` | str | Rival enfrentado (abreviatura) |
| `was_home` | int | 1 = jugó como local, 0 = visitante |
| `minutes` | int | Minutos jugados en esa jornada |
| `goals_scored` | int | Goles en esa jornada |
| `assists` | int | Asistencias en esa jornada |
| `clean_sheets` | int | Portería invicta en esa jornada |
| `total_points` | int | Puntos FPL obtenidos en esa jornada |
| `expected_goals` | float | xG en esa jornada específica |
| `expected_assists` | float | xA en esa jornada específica |
| `influence` | float | Índice de influencia del partido |
| `creativity` | float | Índice de creatividad del partido |
| `threat` | float | Índice de amenaza del partido |
| `value` | int | Precio del jugador en ese momento (en décimas de millón) |
| `selected` | int | Número de managers que lo tenían seleccionado |
| `transfers_in` | int | Transferencias entrantes esa semana |
| `transfers_out` | int | Transferencias salientes esa semana |

---

## 4. `events.csv` — Eventos de Juego

**Fuente:** WhoScored (web scraping) 🕸️ | **Registros:** 444,252 | **Columnas:** 20

> **Actualización 21/03/2026:** `events.csv` ahora se descarga partido por partido via `/matches/{id}/events`, preservando el campo `qualifiers` completo. Columnas: 19 → 20.

### Identificación del Evento
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `id` | int | Identificador único del evento |
| `match_id` | int | ID del partido (FK a matches) |
| `minute` | int | Minuto del partido |
| `second` | int | Segundo exacto dentro del minuto |
| `period` | str | `FirstHalf` (49.1%), `SecondHalf` (50.6%), `PreMatch`, `PostGame` |

### Descripción del Evento
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `event_type` | str | Tipo de evento (39 tipos, ver EDA para distribución) |
| `outcome` | str | `Successful` (77.8%) o `Unsuccessful` (22.2%) |
| `team_name` | str | Equipo que realizó la acción |
| `player_name` | str | Nombre del jugador (puede ser nulo) |
| `player_id` | int | ID del jugador (puede ser nulo) |

### Coordenadas Espaciales (Sistema Opta: 0–100)
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `x` | float | Posición horizontal (0 = defensa propia → 100 = gol rival) |
| `y` | float | Posición vertical (0 = banda izquierda → 100 = banda derecha) |
| `end_x` / `end_y` | float | Coordenadas finales (para pases, tiros) |
| `goal_mouth_y` | float | Posición Y en portería (solo tiros) |
| `goal_mouth_z` | float | Altura Z en portería (solo tiros) |

### Flags Booleanos
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `is_touch` | bool | Si el jugador tocó el balón |
| `is_shot` | bool | Si es un tiro (6,401 totales) |
| `is_goal` | bool | Si fue gol (714 totales, 11.2% efectividad) |

### Campo `qualifiers` — Metadata Rica (columna nueva ✨)
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `qualifiers` | JSON string | Array JSON con **~110 tipos únicos** de metadata por evento. Contiene campos como `BigChance`, `Head`, `RightFoot`, `FastBreak`, `FromCorner`, `Zone`, `Angle`, `Length` y muchos más. |

Estructura de cada elemento del array:
```json
[{"type": {"displayName": "Zone"}, "value": "Back"},
 {"type": {"displayName": "Angle"}, "value": "3.90"},
 {"type": {"displayName": "Head"}, "value": "true"}]
```

#### Qualifiers clave para ML

| Qualifier (Key) | Descripción Técnica | Impacto en Modelos (xG / Match) |
|---|---|---|
| `BigChance` | Oportunidad clara de gol (mantenimiento de posesión + cercanía). | **Crítico** para xG (coeficiente alto). |
| `Penalty` | Tiro desde el punto de pena máxima. | Feature binaria mandatoria (xG fijo ~0.76). |
| `Head` | El remate o pase se realizó con la cabeza. | Indica menor probabilidad de gol vs pie por naturaleza mecánica. |
| `RightFoot` / `LeftFoot` | Pie utilizado para la ejecución. | Calibra precisión según pie dominante del jugador. |
| `FastBreak` | Acción generada en una transición rápida/contraataque. | Indica defensa rival desorganizada (+xG). |
| `FromCorner` | Jugada originada directamente de un tiro de esquina. | Contexto de balón parado táctico. |
| `SetPiece` | Tiro libre o jugada ensayada de balón parado. | Feature de contexto táctico y bloque defensivo estático. |
| `FirstTouch` | Remate o pase realizado sin control previo del balón. | Factor de dificultad técnica; reduce tiempo de reacción del portero. |
| `Volley` | Remate de balón en el aire (volea). | Variable de técnica de disparo (alta varianza en precisión). |
| `ThroughBall` | Pase filtrado que rompe la línea defensiva rival. | Precursor de alta probabilidad de tiro (asistencia clave). |
| `Cross` | Centro desde las bandas al área de penal. | Origen principal de remates de cabeza y "scramble" en el área. |
| `Assisted` | El evento fue precedido por un pase facilitador comprobado. | Atribución de xA (Expected Assists) al asistidor. |
| `IntentionalGoalAssist` | Pase diseñado explícitamente para dejar al compañero solo frente al arco. | Feature de alta calidad creativa. |
| `SavedOffLine` | El tiro fue salvado por un defensa sobre la misma línea de gol. | Caso de xG altísimo que no termina en gol por factor externo. |
| `Blocked` | El tiro fue obstruido por la intervención de un oponente. | Indica presión defensiva cercana y reducción de ángulo de tiro. |
| `Deflected` | El balón cambió de trayectoria por un desvío accidental. | Factor de aleatoriedad en el xG y el resultado. |
| `Zone` | Zona táctica del campo (Back, Left, Right, Center, etc.). | Segmentación espacial para análisis de calor. |
| `Angle` | Ángulo exacto del tiro respecto al centro de la portería. | **Feature geométrica** fundamental para el Modelo 1. |
| `Length` | Distancia total del desplazamiento del balón. | Relevante para calcular peligro de pases largos vs cortos. |
| `Longball` | Pase largo diagonal o vertical que busca profundidad. | Indicador de estilo de juego directo. |

#### Features booleanas extraídas (para ML)
```python
q = df["qualifiers"].astype(str)
df["is_big_chance"]  = q.str.contains("BigChance",  na=False).astype(int)
df["is_header"]      = q.str.contains('"Head"',     na=False).astype(int)
df["is_right_foot"]  = q.str.contains("RightFoot",  na=False).astype(int)
df["is_left_foot"]   = q.str.contains("LeftFoot",   na=False).astype(int)
df["is_counter"]     = q.str.contains("FastBreak",  na=False).astype(int)
df["from_corner"]    = q.str.contains("FromCorner", na=False).astype(int)
df["is_penalty"]     = q.str.contains('"Penalty"',  na=False).astype(int)
df["is_volley"]      = q.str.contains("Volley",     na=False).astype(int)
df["first_touch"]    = q.str.contains("FirstTouch", na=False).astype(int)
```

#### Features geométricas derivadas de x, y (Taller2 ML1 — obligatorias)
```python
shots["distance_to_goal"] = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
shots["angle_to_goal"]    = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))
shots["dist_angle"]       = shots["distance_to_goal"] * shots["angle_to_goal"]
shots["is_in_area"]       = (shots["x"] > 83).astype(int)
shots["is_central"]       = shots["y"].between(33, 67).astype(int)
```


---

*Documento actualizado con análisis de qualifiers y guía de Feature Engineering & Selection del Taller2 ML1-2026.*
