# EDA — `events.csv`

**Registros:** 444,252 eventos | **Columnas:** 20 | **Fuente:** Premier League API (WhoScored / Opta)

> Actualizado: 21/03/2026 — Datos incluyen el campo `qualifiers` completo (~110 tipos únicos).

---

## 1. Calidad de Datos
![Nulos](graficas/01_nulos.png)

Los nulos están en campos estructurales (`player_name`, `player_id`) o específicos de acción (`end_x`, `goal_mouth_z`). Los campos críticos de posición y tipo están completos.

---

## 2. Distribución de Eventos
![Tipos de Evento](graficas/02_tipos_evento.png)

El **Pass** domina el dataset (~60%), lo cual es coherente con el fútbol moderno de posesión. Los eventos de tiro (`MissedShots`, `SavedShot`, `Goal`) son una fracción pequeña pero altamente informativa para el Modelo 1.

---

## 3. Outcome y Desbalance
![Outcome](graficas/03_outcome.png)

La tasa de éxito global es **~78%**. Este desbalance es completamente normal. Para modelos que predigan el éxito de una acción específica (ej. si un pase llega a destino), se recomiendan técnicas de `class_weight` o umbral ajustado.

---

## 4. Distribución Temporal (Minutos)
![Eventos por Minuto](graficas/04_eventos_minuto.png)

Se observa un flujo constante con picos lógicos en los minutos 45 y 90 (tiempo añadido). El **minuto** es una feature contextual útil — estudios muestran que los goles marcados después del minuto 80 tienen un efecto psicológico desproporcionado en el momentum del partido.

---

## 5. Calor Total del Campo
![Heatmap Todos](graficas/05_heatmap_todos.png)

La actividad se concentra en el carril central (y≈50) y zonas de transición. Los laterales (y<20 o y>80) tienen menor densidad, sugiriendo que la Premier League 2024-25 tiene un estilo predominantemente central.

> **💡 Visualización avanzada recomendada:** *Carry Maps* — trazar la progresión de cada conducción de balón con flechas colored por resultado. Es el estándar de StatsBomb para analizar progresión con balón (fuente: [StatsBomb Open Data](https://github.com/statsbomb/open-data)).

---

## 6. Mapa de Tiros y Goles
![Heatmap Tiros](graficas/06_heatmap_tiros.png)

Los tiros se concentran en el área grande (x>83), pero los **goles (cyan)** se agrupan aún más cerca, en la zona de máxima efectividad (x>90, 30<y<70). Este patrón es la intuición detrás del **xG**: la posición geométrica es el predictor más fuerte.

> **💡 Visualización avanzada recomendada:** *xG Shot Map por Partido* (estándar FotMob/StatsBomb) — cada tiro es un círculo escalado por su xG, coloreado por si fue gol. Permite ver en un vistazo si un equipo tuvo suerte o fue merecido ganar.

---

## 7. Efectividad por Zona del Campo
![Efectividad Zona](graficas/07_efectividad_zona.png)

| Zona | % Efectividad |
|---|---|
| Media/Baja (x<33) | <1% |
| Media alta (33-67) | ~2% |
| Zona peligrosa (67-85) | ~8% |
| **Área grande (>85)** | **~18%** |

La efectividad se multiplica **×18** al pasar de media cancha al área. Este gradiente es el eje central del Modelo 1 (xG).

---

## 8. Features Geométricas — Taller2 ML1
![Distancia y Ángulo](graficas/08_distancia_angulo_xg.png)

Siguiendo el Taller2, derivamos `distance_to_goal` y `angle_to_goal`. La distribución confirma que:
- Los goles tienen **menor distancia** (mediana ~10m vs ~18m en no-goles).
- Los goles tienen **mayor ángulo** (ángulo más abierto = más frente al arco).

```python
shots["distance_to_goal"] = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
shots["angle_to_goal"]    = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))
```

> **💡 Visualización avanzada recomendada:** *Beeswarm / Violin Plot de xG por ángulo y distancia* — separa tiros en bins de distancia y muestra la distribución de conversión. Es el gráfico clásico para explicar visualmente el xG a no especialistas.

---

## 9. ¿Por qué estos 10 qualifiers de los ~110 disponibles?

![Efectividad Qualifiers](graficas/09_efectividad_qualifiers.png)

Los 110 tipos únicos de qualifiers en el dataset incluyen mucha metadata técnica de coordenadas (PassEndX, PassEndY, Length, Angle) que describe cada acción con precisión milimétrica. Sin embargo, **para un modelo de xG a nivel de tiro**, la selección óptima sigue un criterio específico:

### Criterio de selección
1. **Alta correlación con probabilidad de gol** — documentada en literatura (StatsBomb, Opta papers).
2. **Features booleanas extraíbles directamente** (str.contains, sin parseo costoso).
3. **Suficiente frecuencia** en el subconjunto de tiros (>0.5% de los tiros).
4. **No redundante** con variables geométricas ya disponibles.

### Los 10 seleccionados y por qué

| Qualifier | Frecuencia en tiros | Por qué es crítico |
|---|---|---|
| `BigChance` | ~15% tiros | xG implícito altísimo (~38%). Mejor feature individual del modelo |
| `Penalty` | ~4% tiros | xG fijo ~76% — requiere categoría propia |
| `Head` | ~22% tiros | Efectividad ~30% menor vs pie — captura dificultad mecánica |
| `RightFoot` | ~55% tiros | Pie dominante vs pie débil: cambia la precisión del golpe |
| `LeftFoot` | ~20% tiros | Contraste con RightFoot |
| `FastBreak` | ~8% tiros | Defensa desorganizada → mayor espacio → más xG |
| `FromCorner` | ~7% tiros | Zona y ángulo controlados — patrón táctico prediseñado |
| `FirstTouch` | ~18% tiros | Menos tiempo de preparación = mayor varianza en resultado |
| `Volley` | ~5% tiros | Alta dificultad técnica — baja conversión (conocido en literatura) |
| `SetPiece` | ~12% tiros | Contexto de balón parado — defensa en bloque estático |

### ¿Qué otros qualifiers podrían ser útiles?

Tras investigar la literatura de Opta/WhoScored y StatsBomb, estos son los descartados inicialmente pero con potencial real:

| Qualifier adicional | Por qué podría ser valioso | Limitación |
|---|---|---|
| `ThroughBall` | Precursor de tiros en espacios — indica ruptura defensiva previa | No es del tiro en sí, sino del pase previo |
| `Blocked` | Indica que hubo un defensor en la trayectoria — reduce el xG real | Requiere razonamiento inverso (el tiro SÍ ocurrió pero fue bloqueado) |
| `Deflected` | Introduce aleatoriedad — el portero no puede anticipar | Raro pero muy informativo sobre el "ruido" del xG |
| `Assisted` | Si el tiro viene de un pase de gol → la ocasión fue más preparada | Requiere join con evento anterior |
| `IntentionalGoalAssist` | Asistencia diseñada = chance de mayor calidad | Similar al anterior, requiere join |
| `SavedOffLine` | Casi goles salvados sobre la línea — xG altísimo | Post-evento, no disponible para predicción del tiro |
| `GoalMouthY` / `GoalMouthZ` | Posición exacta donde impactó el tiro en la portería | Ya tenemos columnas separadas `goal_mouth_y`/`goal_mouth_z` — podemos derivar zonas |
| `Zone14` | Zona 14 del campo (espacio entre los dos bloques defensivos) | Requiere parseo del valor numérico dentro del JSON |

> **Propuesta de Feature Engineering avanzado**: combinar `goal_mouth_y` + `goal_mouth_z` para crear una **"zona de portería"** (high_left, high_right, center_low, etc.) — los porteros tienen efectividades muy diferentes por zona.

---

## 10. Mapa de Tiros por Tipo de Contacto
![Mapa Tiros Tipo](graficas/10_mapa_tiros_tipo.png)

Visualizamos la especialización espacial de cada tipo de contacto: los cabezazos (🔵) se concentran en el segundo palo (centros), mientras remates de pie tienen mayor rango. Los **⭐ dorados** son goles — claramente agrupados en la zona de máxima peligrosidad.

> **💡 Visualización avanzada recomendada:** *Expected Threat (xT) Map* — a diferencia del xG (que solo mide tiros), el xT mide el valor de **cada acción con balón** según cuánto incrementó la probabilidad de gol. Desarrollado por Karun Singh y publicado por StatsBomb. Requiere un modelo de campo completo.

---

## 🌍 Visualizaciones de Estado del Arte — Roadmap

Estas son las visualizaciones más informativas, impactantes e impresionantes en el mundo del football analytics moderno, con referencia técnica:

### Para `events.csv`

| Visualización | Descripción | Dónde añadirla |
|---|---|---|
| **xG Shot Map** | Cada tiro = círculo escalado por xG, coloreado por gol/no-gol | EDA_events / Modelo 1 |
| **Carry Map con flechas** | Conducciones con flechas coloreadas por zona y resultado | EDA_events |
| **Progressive Pass Map** | Pases que avanzan ≥10 yds hacia portería — filtrado por equipo | EDA_events |
| **xT (Expected Threat) Heatmap** | Valor táctico de cada zona del campo según la acción | EDA_events / Modelo 1 |
| **Convex Hull por equipo** | Área táctica que cubre cada equipo durante el juego | EDA_events |
| **Tackle Map** | Localización y resultado de todas las entradas defensivas | EDA_events |

### Para `matches.csv`

| Visualización | Descripción | Dónde añadirla |
|---|---|---|
| **xG Timeline** | Evolución del xG acumulado minuto a minuto (home vs away) | EDA_matches |
| **xG vs Goles reales (scatter)** | Over/underperformance por equipo en la temporada | EDA_matches / Modelo 2 |
| **Betting Calibration Plot** | ¿Cuándo las cuotas B365 predicen bien el resultado? | EDA_matches |
| **Goal Difference Distribution** | Histograma de diferencias de gol — fundamental para Dixon-Coles | EDA_matches |

### Para `players.csv` y `player_history.csv`

| Visualización | Descripción | Dónde añadirla |
|---|---|---|
| **Radar Chart por jugador** (estilo FBRef) | 8-12 métricas normalizadas vs el resto de la liga | EDA_players |
| **xG vs Goals scatter** | Over/underperformers individuales | EDA_players |
| **Shot Placement Heatmap** | Dónde impactan los tiros del jugador en la portería (goal_mouth_y/z) | EDA_players |
| **Form Curve** | Evolución de puntos FPL jornada a jornada con media móvil | EDA_player_history |
| **Price vs Points scatter** | Value-for-money por posición | EDA_players |

---

## Resumen para Modelado

| Métrica | Valor |
|---|---|
| Total Tiros | ~6,400 |
| Total Goles | ~714 |
| Efectividad Global | 11.2% |
| Tipos únicos de qualifiers | **~110** |
| **AUC-ROC Objetivo (xG)** | **> 0.78** |

*Documento consolidado y enriquecido con investigación de literatura de football analytics — 21/03/2026.*
