# Exploración Avanzada de Datos (EDA)

Este informe detalla el análisis exploratorio realizado sobre los datos de la Premier League 2025-26.

---

## 1. Resumen General de Datasets

| Dataset | Registros | Columnas | Archivo |
|---------|-----------|----------|---------|
| Partidos | 291 | 41 | `data/raw/matches.json` |
| Eventos | 398,961 | 20 | `data/raw/events.csv` |

> **Nota:** Los 291 partidos corresponden a la temporada en curso. Se esperan 380 partidos al final de la temporada (38 jornadas).

---

## 2. Análisis de Partidos (`matches.json`)

### 2.1 Distribución de Resultados
| Resultado | Cantidad | Porcentaje |
|-----------|----------|------------|
| Home (H) | 123 | 42.3% |
| Away (A) | 92 | 31.6% |
| Draw (D) | 76 | 26.1% |

**Insight:** La ventaja de localía se confirma estadísticamente. El local gana ~10% más que el visitante. Los empates (~26%) serán el mayor reto de clasificación.

### 2.2 Goles por Partido
| Métrica | Local (fthg) | Visitante (ftag) | Total |
|---------|-------------|------------------|-------|
| Media | 1.52 | 1.26 | 2.77 |
| Mínimo | 0 | 0 | 0 |
| Máximo | 5 | 5 | 9 |

**Insight:** Se anotan en promedio ~2.8 goles por partido. Un partido de 0-0 sí ocurre, y el máximo registrado es un partido con 9 goles totales.

### 2.3 Estadísticas de Juego — Promedios
| Estadística | Local | Visitante | Diferencia |
|-------------|-------|-----------|------------|
| Tiros totales | 13.6 | 11.0 | +2.6 |
| Tiros a puerta | 4.4 | 3.8 | +0.6 |
| Córners | 5.3 | 4.6 | +0.7 |
| Faltas | 10.6 | 11.0 | -0.4 |
| Amarillas | 1.7 | 2.1 | -0.4 |
| Rojas | 0.06 | 0.05 | +0.01 |

**Insights:**
- El local dispara más y acorrala más (más tiros y córners).
- El visitante comete más faltas y recibe más amarillas (juega más defensivamente).
- Las rojas son eventos raros (~6% de los partidos tienen una).

### 2.4 Cuotas de Apuestas (Bet365)
| Cuota | Media | Mín | Máx |
|-------|-------|-----|-----|
| Home (b365h) | 2.65 | 1.13 | 11.00 |
| Draw (b365d) | 3.99 | 3.00 | 8.50 |
| Away (b365a) | 3.95 | 1.27 | 21.00 |

**Insight:** Cuando `b365h < 1.5`, el local es un **gran favorito** (como Man City en casa). Cuotas > 5.0 indican grandes sorpresas si ganan.

### 2.5 Probabilidades Implícitas
| Probabilidad | Media | Mín | Máx |
|-------------|-------|-----|-----|
| P(Home) | 46% | 9% | 89% |
| P(Draw) | 26% | 12% | 33% |
| P(Away) | 34% | 5% | 79% |

**Insight:** Las casas de apuestas asignan una probabilidad media del 46% al local, coherente con la distribución real de 42.3%.

### 2.6 Árbitros Más Activos
| Árbitro | Partidos |
|---------|----------|
| A Taylor | 22 |
| C Kavanagh | 21 |
| M Oliver | 21 |
| P Bankes | 21 |
| S Attwell | 18 |

**Insight:** 23 árbitros en total. Los más activos dirigen ~20 partidos cada uno. Podrían ser una feature interesante (algunos árbitros son más estrictos con las tarjetas).

---

## 3. Análisis de Eventos (`events.csv`)

### 3.1 Volumen de Datos
- **398,961 eventos totales** distribuidos en 291 partidos.
- Promedio: **~1,371 eventos por partido**.
- **517 jugadores únicos** registrados.

### 3.2 Distribución por Tipo de Evento (Top 10)
| Tipo | Cantidad | % |
|------|----------|---|
| Pass | 250,420 | 62.8% |
| BallRecovery | 21,228 | 5.3% |
| BallTouch | 18,423 | 4.6% |
| Aerial | 16,746 | 4.2% |
| Clearance | 14,647 | 3.7% |
| Foul | 11,426 | 2.9% |
| TakeOn | 9,312 | 2.3% |
| Tackle | 8,808 | 2.2% |
| CornerAwarded | 5,086 | 1.3% |
| Dispossessed | 4,543 | 1.1% |

**Insight:** Los pases dominan con un 63% de las acciones. Un equipo promedio realiza ~430 pases por partido. La precisión de pases global es del 77.8%.

### 3.3 Tiros y Goles
| Métrica | Valor |
|---------|-------|
| Tiros totales | 6,401 |
| Goles totales | 714 |
| Efectividad | 11.2% |
| Tiros por partido | ~22.0 |
| Goles por partido | ~2.5 |

### 3.4 Goleadores — Top 10
| Jugador | Goles |
|---------|-------|
| Erling Haaland | 20 |
| Igor Thiago | 14 |
| João Pedro | 13 |
| Antoine Semenyo | 12 |
| Hugo Ekitiké | 10 |
| Viktor Gyökeres | 10 |
| Cole Palmer | 9 |
| Bruno Guimarães | 9 |
| Bryan Mbeumo | 9 |
| Harry Wilson | 9 |

### 3.5 Jugadores con Más Participación
| Jugador | Eventos |
|---------|---------|
| Elliot Anderson | 2,784 |
| Lewis Dunk | 2,569 |
| Trevoh Chalobah | 2,552 |
| Virgil van Dijk | 2,434 |
| Jan Paul van Hecke | 2,410 |

**Insight:** Los jugadores con más eventos son defensas y mediocampistas centrales (tocan el balón constantemente). Los delanteros como Haaland tienen menos eventos pero más impacto.

### 3.6 Equipos con Más Eventos
| Equipo | Eventos |
|--------|---------|
| Chelsea | 23,131 |
| Man City | 22,217 |
| Man Utd | 22,032 |
| Brighton | 21,879 |
| Fulham | 21,447 |

**Insight:** Los equipos con más posesión (Chelsea, Man City) generan más eventos. Los equipos más defensivos (Burnley, Sunderland) generan menos.

### 3.7 Distribución Temporal
| Periodo | Eventos | % |
|---------|---------|---|
| Segunda Parte | 202,037 | 50.6% |
| Primera Parte | 195,880 | 49.1% |

**Insight:** Ligeramente más eventos en la segunda parte, posiblemente por sustituciones y equipos que presionan más buscando el resultado.

### 3.8 Coordenadas Espaciales
| Eje | Mín | Máx | Media |
|-----|-----|-----|-------|
| X | 0.1 | 100.0 | 46.4 |
| Y | 0.1 | 100.0 | 50.9 |

**Insight:** La media de X es 46.4 (ligeramente más en campo propio), confirmando que la mayoría de eventos son defensivos o de construcción. La media Y es ~50.9, indicando distribución equilibrada entre ambas bandas.

---

## 4. Features Recomendadas para Modelado

### Para Regresión Lineal (Predecir `fthg`)
1. `hs` — Home Shots
2. `hst` — Home Shots on Target ⭐ (más correlacionado)
3. `hc` — Home Corners
4. `hf` — Home Fouls

### Para Regresión Logística (Predecir `ftr`)
1. `implied_prob_h`, `implied_prob_d`, `implied_prob_a` — Probabilidades del mercado
2. `SOTDiff` = `hst - ast` — Diferencia de tiros a puerta
3. `b365h`, `b365d`, `b365a` — Cuotas crudas

### Features Derivadas Potenciales
- `shots_ratio` = `hs / (hs + as_)` — Dominio de tiros
- `corner_ratio` = `hc / (hc + ac)` — Dominio territorial
- `card_intensity` = `(hy + ay + hr*3 + ar*3)` — Intensidad del partido
