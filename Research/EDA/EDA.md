# EDA Maestro — Proyecto ML Premier League

**Fecha:** 21/03/2026 | **Autor:** Equipo Análisis Apuestas

---

## 1. Resumen de los 4 Datasets

| Dataset | Registros | Columnas | Calidad |
|---|---|---|---|
| `players.csv` | 822 | 37 | ✅ Excelente — sin nulos en cols críticas |
| `matches.csv` | 291 | 53 | ✅ Excelente — algunas cuotas con nulos |
| `events.csv` | 444,252 | 20 | ✅ Muy buena — nulos esperados en cols opcionales |
| `player_history.csv` | 1,499 | 20 | ✅ Perfecta — 0 nulos (pero dataset reducido por API degradada) |

### Limitación conocida
> ⚠️ `player_history.csv` debería tener ~15,000 registros (1 por jugador × jornada). La API está en estado degradado y solo sirve ~1,500 filas. La arquitectura del proyecto está preparada para reemplazar este archivo cuando la API se recupere ejecutando `scripts/download_bulk_data.py`.

---

## 2. Hallazgos Clave Transversales

### 2.1 La ventaja local es real y medible
- En `matches.csv`: El local gana el **42%** de los partidos vs 32% del visitante.
- En `player_history.csv`: Los jugadores acumulan en promedio **+1.3 puntos FPL más** jugando en casa.
- En `events.csv`: El local genera más tiros, córners y eventos ofensivos.

**→ `was_home` debe ser un feature en ambos modelos.**

### 2.2 Las cuotas del mercado están calibradas
Las probabilidades implícitas de Bet365 predijeron el resultado real con notable precisión. El mercado de apuestas agrega información de miles de analistas y modelos. Usar cuotas como features no es trampa — es información pública disponible antes del partido.

**→ `implied_prob_h`, `implied_prob_d`, `implied_prob_a` son features de alto valor para el Modelo 2.**

### 2.3 La posición del tiro explica la mayoría del xG
El mapa de calor de tiros en `events.csv` muestra que los goles provienen casi exclusivamente del área grande (>85 en coordenada X). La efectividad sube de <1% en zonas medias a **~18% en el área**.

**→ Las coordenadas X,Y del tiro son los features más importantes para el Modelo 1.**

### 2.4 El xG es un buen pero imperfecto predictor
En `players.csv` y `player_history.csv`, xG vs goles reales muestra calibración razonable pero alta dispersión a nivel partido. El xG subestima a los grandes rematadores (Haaland, Isak) y sobreestima a los ineficientes.

**→ Incluir el historial de over/underperformance del jugador como feature de corrección.**

### 2.5 Los datos de tiempo de juego son críticos
El 35% de registros en `player_history.csv` tienen 0 minutos. En `players.csv`, la correlación más alta con `total_points` es `minutes` (r≈0.85). Un jugador que no juega no puede anotar.

**→ Filtrar o segmentar siempre por `minutes > 0` para evitar aprender el caso trivial.**

### 2.6 🔑 El campo `qualifiers` de events.csv es la mina de oro del Modelo 1
El campo `qualifiers` es un array JSON anidado con **110 tipos únicos** de metadata por evento. Extrayendo features booleanas simples via `str.contains()` se obtiene:

| Feature extraída | Qualifier JSON | Impacto en xG |
|---|---|---|
| `is_big_chance` | `BigChance` | xG implícito ~35-40% |
| `is_penalty` | `"Penalty"` | xG fijo ~75-76% |
| `is_header` | `"Head"` | Reduce efectividad ~30% vs pie |
| `is_counter` | `FastBreak` | Defensa desorganizada → mayor xG |
| `from_corner` | `FromCorner` | Ángulo específico y zona conocida |
| `first_touch` | `FirstTouch` | Mayor dificultad técnica |
| `is_volley` | `Volley` | Baja tasa de conversión |

**→ Incorporar estas features en el Modelo 1 puede mejorar el AUC del xG en 5-15 puntos porcentuales respecto a usar solo x,y.**

---

## 3. Propuestas de Modelos

---

### 🥅 Modelo 1 — Expected Goals Predictor
**Objetivo:** Predecir el número de goles que marcará un equipo en un partido dado.

**Enfoque recomendado:** Regresión (goles es una variable de conteo → Poisson Regression o XGBoost Regressor)

**Features candidatas — Modelo 1 (xG a nivel de tiro):**

| Feature | Fuente | Tipo | Importancia | Razón |
|---|---|---|---|---|
| `distance_to_goal` | events (x,y) | float | ⭐⭐⭐⭐⭐ | El predictor más importante del xG según todos los papers |
| `angle_to_goal` | events (x,y) | float | ⭐⭐⭐⭐⭐ | Junto con distancia define la geometría del tiro |
| `dist_angle` | derivada | float | ⭐⭐⭐⭐ | Interacción distancia×ángulo — captura relación no lineal |
| `is_in_area` | derivada (x>83) | bool | ⭐⭐⭐⭐ | Dentro del área → xG se dispara |
| `is_central` | derivada (33<y<67) | bool | ⭐⭐⭐ | Tiros centrales más peligrosos que de banda |
| `is_big_chance` | events (qualifiers) | bool | ⭐⭐⭐⭐⭐ | xG implícito ~38%: la mejor señal individual |
| `is_penalty` | events (qualifiers) | bool | ⭐⭐⭐⭐⭐ | xG fijo ~76% — categoría propia |
| `is_header` | events (qualifiers) | bool | ⭐⭐⭐ | Reduce efectividad ~30% vs pie |
| `is_counter` | events (qualifiers) | bool | ⭐⭐⭐ | Defensa desorganizada → mayor xG |
| `first_touch` | events (qualifiers) | bool | ⭐⭐ | Menos tiempo = más difícil pero más impredecible |
| `is_set_piece` | events (qualifiers) | bool | ⭐⭐⭐ | Situaciones tácticas prediseñadas |
| `from_corner` | events (qualifiers) | bool | ⭐⭐ | Ángulo y zona específicos conocidos |
| `minute` | events | int | ⭐⭐ | Presión temporal — goles en minuto 85+ contexto distinto |
| `game_state` | derivada | cat | ⭐⭐⭐ | ¿Va ganando/perdiendo/empatando? Cambia el contexto del tiro |
| `dist_squared` | derivada | float | ⭐⭐⭐ | Relación no lineal con la probabilidad de gol |

**Feature Engineering sugerido:**
- `distance_to_goal = sqrt((100-x)² + (50-y)²)` — **obligatoria (Taller2)**
- `angle_to_goal = |arctan2(50-y, 100-x)|` — **obligatoria (Taller2)**
- `dist_angle = distance * angle` — interacción clave
- `shot_quality_index = is_big_chance*0.38 + is_in_area*0.18 + (1 - distance_norm)*0.15` — índice compuesto
- Rolling over/underperformance: `xg_overperf = goles_reales / xG_acumulado` por jugador

**Métricas de evaluación:** AUC-ROC (benchmark ≥ 0.78), Brier Score, Log Loss


---

### 🏆 Modelo 2 — Match Predictor (Winner)
**Objetivo:** Predecir el resultado del partido: Victoria Local (H), Empate (D), Victoria Visitante (A).

**Enfoque recomendado:** Clasificación multiclase — XGBoost, Random Forest, o Red Neuronal

**Features candidatas — Modelo 2 (resultado a nivel de partido):**

| Feature | Fuente | Tipo | Importancia | Razón |
|---|---|---|---|---|
| `implied_prob_h` | matches (b365h) | float | ⭐⭐⭐⭐⭐ | Consenso del mercado — mejor predictor único según papers |
| `implied_prob_d` | matches (b365d) | float | ⭐⭐⭐⭐⭐ | Idem empate |
| `implied_prob_a` | matches (b365a) | float | ⭐⭐⭐⭐⭐ | Idem visitante |
| `elo_home` | derivada | float | ⭐⭐⭐⭐⭐ | Elo rating del local — #1 predictor en literatura académica |
| `elo_away` | derivada | float | ⭐⭐⭐⭐⭐ | Elo rating del visitante |
| `elo_diff` | derivada | float | ⭐⭐⭐⭐⭐ | Diferencia de Elo — captura ventaja relativa |
| `home_goals_avg5` | matches (rolling) | float | ⭐⭐⭐⭐ | Forma ofensiva reciente del local |
| `away_goals_avg5` | matches (rolling) | float | ⭐⭐⭐⭐ | Forma ofensiva reciente del visitante |
| `home_goals_conceded_avg5` | matches (rolling) | float | ⭐⭐⭐⭐ | Solidez defensiva local |
| `away_goals_conceded_avg5` | matches (rolling) | float | ⭐⭐⭐⭐ | Solidez defensiva visitante |
| `home_xg_avg5` | events (rolling) | float | ⭐⭐⭐⭐ | Calidad ofensiva vs cantidad de goles |
| `away_xg_avg5` | events (rolling) | float | ⭐⭐⭐⭐ | Idem visitante |
| `was_home` | matches | bool | ⭐⭐⭐⭐ | Ventaja local — confirmada empíricamente |
| `home_win_streak` | derivada | int | ⭐⭐⭐ | Racha victorias en casa — momentum |
| `head_to_head_h_win_rate` | matches (hist.) | float | ⭐⭐⭐ | Historial directo — algunos equipos tienen "maldición" |
| `referee_home_bias` | matches | float | ⭐⭐ | Árbitros con sesgo estadístico hacia local |
| `squad_value_ratio` | players | float | ⭐⭐⭐ | Ratio de calidad de plantilla (precio FPL como proxy) |
| `home_injured_count` | players | int | ⭐⭐ | Titulares lesionados del local |
| `big_chances_avg5` | events (rolling) | float | ⭐⭐⭐ | BigChances generadas últimos 5 partidos |
| `shot_quality_idx_home` | events (derivada) | float | ⭐⭐⭐ | Índice de calidad de tiros ponderado por xG |

**Feature Engineering avanzado:**
- **Elo Rating**: Actualizado K=32 partido a partido, con decay temporal
- **Diferencia de Elo normalizada**: `elo_diff / 400` — escala estándar para probabilidades
- **Forma exponencialmente decaída**: `Σ resultado_i × exp(-λ × (n-i))` — más peso a juegos recientes
- **Shot Quality Index**: `Σ (is_big_chance×0.38 + is_in_area×0.18 + counter×0.12)` por equipo por partido
- **Dixon-Coles Attack/Defense Strength**: parámetros de ataque y defensa independientes por equipo
- **Varianza de Rendimiento**: Std de goles últimos 10 partidos — equipos impredecibles vs estables

**Métricas de evaluación:** Accuracy (benchmark >49.8% Bet365), F1 Macro, Log Loss, ROI simulado

---

## 4. Riesgos y Consideraciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Dataset `player_history` reducido por API degradada | Alto para Modelo 1 | Esperar recuperación API o enriquecer con datos externos |
| Desbalance de clases en `ftr` (42% local gana) | Medio para Modelo 2 | Usar SMOTE, class_weight o calibración de probabilidades |
| Leakage con cuotas post-partido | Alto si no se tiene cuidado | Usar SOLO cuotas de apertura (opening odds), no de cierre |
| Sobreajuste con features de forma reciente | Medio | Cross-validation temporal (TimeSeriesSplit), no fold aleatorio |
| Pocos partidos (291) para DL | Alto para Redes Neuronales | Priorizar modelos como XGBoost o Logistic Regression |

---

---

## 5. Plan de Feature Engineering Recomendado

### Paso 1 — Tabla de eventos enriquecida (Modelo 1)

```python
import pandas as pd
import numpy as np

def prepare_xg_features(df_events):
    """Prepara features para el Modelo 1 (xG). Guiado por Taller2 ML1."""
    shots = df_events[df_events["is_shot"] == True].copy()

    # ── Features Geométricas (Taller2 obligatorias) ──────────────────────────
    shots["distance_to_goal"] = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
    shots["angle_to_goal"]    = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))

    # ── Features de Qualifiers (si disponibles) ──────────────────────────────
    if "qualifiers" in shots.columns:
        q = shots["qualifiers"].astype(str)
        shots["is_header"]     = q.str.contains('"Head"',    na=False).astype(int)
        shots["is_right_foot"] = q.str.contains("RightFoot", na=False).astype(int)
        shots["is_left_foot"]  = q.str.contains("LeftFoot",  na=False).astype(int)
        shots["is_big_chance"] = q.str.contains("BigChance", na=False).astype(int)
        shots["is_penalty"]    = q.str.contains('"Penalty"', na=False).astype(int)
        shots["is_counter"]    = q.str.contains("FastBreak", na=False).astype(int)
        shots["first_touch"]   = q.str.contains("FirstTouch",na=False).astype(int)
        shots["is_volley"]     = q.str.contains("Volley",    na=False).astype(int)
    
    # ── Feature de Zona ──────────────────────────────────────────────────────
    shots["zone"] = pd.cut(shots["x"], bins=[0,67,83,100],
                            labels=["Exterior","Media","Área"], include_lowest=True)
    return shots
```

### Paso 2 — Rolling Stats por equipo (Modelo 2)

```python
def get_rolling_stats(df_matches, team_name, window=5):
    """Calcula estadísticas de forma reciente para un equipo (Taller2 Anexo C)."""
    team_matches = df_matches[
        (df_matches["home_team"] == team_name) | 
        (df_matches["away_team"] == team_name)
    ].copy()
    team_matches["goles"] = team_matches.apply(
        lambda x: x["fthg"] if x["home_team"] == team_name else x["ftag"], axis=1
    )
    team_matches["goles_contra"] = team_matches.apply(
        lambda x: x["ftag"] if x["home_team"] == team_name else x["fthg"], axis=1
    )
    return {
        "form_goles":        team_matches["goles"].rolling(window).mean().iloc[-1],
        "form_goles_contra": team_matches["goles_contra"].rolling(window).mean().iloc[-1],
    }
```

### Paso 3 — Join final para entrenamiento

```
partido → (equipo_local + forma_local_rolling5 + jugadores_local_status)
        + (equipo_visitante + forma_visitante_rolling5 + jugadores_visitante_status)
        + (cuotas implícitas Bet365: implied_prob_h/d/a)
        + (eventos del partido: BigChances, tiros_a_puerta, xG_acumulado)
```

---

## 6. 📐 Feature Engineering & Selection (Guía Taller2 ML1)

> Referencia: *Taller 2 — ¿Puedes Predecir el Fútbol Mejor que las Casas de Apuestas?*
> Machine Learning 1 (ML1-2026) — Universidad Externado de Colombia

### 6.1 Transformaciones Recomendadas

| Transformación | Feature | Razón |
|---|---|---|
| Logarítmica | `distance_to_goal` | Reduce sesgos en distribución de distancias |
| Standard Scaling | Todas las features numéricas | Necesario para Regresión Logística y SVM |
| MinMax Scaling | `distance_to_goal`, `angle_to_goal` | Alternativa si hay outliers severos |
| One-Hot Encoding | `event_type`, `team_name`, `position` | Categoricas con baja cardinalidad |
| Binary (0/1) | `is_header`, `is_big_chance`, etc. | Qualifiers ya en formato binario |

### 6.2 Variables Derivadas Clave

#### Para Modelo 1 (xG):
```python
# Obligatorias (Taller2)
shots["distance"]     = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
shots["angle"]        = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))

# Avanzadas (propias)
shots["dist_squared"] = shots["distance"] ** 2          # Relación no lineal
shots["zone_x"]       = (shots["x"] > 83).astype(int)  # ¿Dentro del área?
shots["central_y"]    = (shots["y"].between(33, 67)).astype(int)  # ¿Zona central?
shots["dist_angle"]   = shots["distance"] * shots["angle"]        # Interacción
```

#### Para Modelo 2 (Match Predictor):
```python
# Rolling averages (últimas 5 jornadas)
matches["home_goals_avg5"]   = ...  # Media goles anotados como local
matches["away_goals_avg5"]   = ...  # Media goles anotados como visitante
matches["home_xg_avg5"]      = ...  # xG generado (de events.csv)
matches["away_xg_avg5"]      = ...  # xG generado visitante

# Probabilidades implícitas normalizadas
total = (1/matches["b365h"]) + (1/matches["b365d"]) + (1/matches["b365a"])
matches["implied_h"] = (1/matches["b365h"]) / total
matches["implied_d"] = (1/matches["b365d"]) / total
matches["implied_a"] = (1/matches["b365a"]) / total
```

### 6.3 Métodos de Feature Selection

| Método | Aplicación | Herramienta |
|---|---|---|
| **Matriz de Correlación** | Detectar multicolinealidad | `df.corr()` + heatmap |
| **RFE** (Recursive Feature Elimination) | Selección wrapper para Logistic Regression | `sklearn.feature_selection.RFE` |
| **L1 Regularization (Lasso)** | Elimina features irrelevantes automáticamente | `LogisticRegression(penalty='l1')` |
| **L2 Regularization (Ridge)** | Reduce pesos sin eliminar features | `LogisticRegression(penalty='l2')` |
| **Feature Importance (Random Forest)** | Ranking no paramétrico de importancia | `RandomForestClassifier.feature_importances_` |
| **Permutation Importance** | Evalúa impacto real en métricas del modelo | `sklearn.inspection.permutation_importance` |

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=500)
rfe = RFE(model, n_features_to_select=8)
rfe.fit(X_train, y_train)
print("Features seleccionadas:", X_train.columns[rfe.support_].tolist())
```

### 6.4 Manejo del Desbalanceo de Clases

**Modelo 1 (xG)**: La tasa de goles es ~11% → **fuertemente desbalanceado**.
```python
# Opción 1: class_weight automático
model = LogisticRegression(class_weight='balanced')

# Opción 2: SMOTE oversampling
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE().fit_resample(X_train, y_train)

# Opción 3: Threshold tuning (ajustar umbral de clasificación)
probs = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # En vez de 0.5 default
preds = (probs >= threshold).astype(int)
```

### 6.5 Benchmarks Objetivo (Taller2)

| Modelo | Métrica | Benchmark mínimo | Objetivo ambicioso |
|---|---|---|---|
| **Modelo 1 (xG)** | AUC-ROC | **0.78** | >0.85 |
| **Modelo 2 (Match Predictor)** | Accuracy | **49.8%** (Bet365) | >55% |
| **Modelo 2** | Log Loss | <1.0 | <0.9 |

> 💡 El benchmark de Match Predictor es el accuracy promedio de Bet365 para esta temporada. Superarlo significa que el modelo tiene valor predictivo real.

---

## 7. Dataset Linking (Relaciones entre tablas)

```
events.csv ──────────────── match_id ──→ matches.csv
player_history.csv ─── player_id ──→ players.csv
player_history.csv ─── gameweek  ──→ (temporada implícita)
events.csv ──────── player_id ──→ players.csv (opcional)
```

---

## 8. 📚 Recursos Técnicos (Taller2 ML1)

> Estos 4 recursos fueron recomendados explícitamente en el Taller2. Los exploramos y extractamos lo más valioso para los modelos.

---

### 📖 1. Soccermatics — David Sumpter

**Libro + Código Python** | Matemáticas del fútbol con implementaciones prácticas.

**Lo más valioso para nuestro proyecto:**
- **xG con Regresión Logística**: Sumpter implementa xG usando distancia y ángulo al gol como features. Exactamente el Taller2 pide. El modelo base es:
  ```python
  # Logistic Regression para xG (Sumpter style)
  from sklearn.linear_model import LogisticRegression
  X = shots[["distance", "angle", "dist_squared"]]
  y = shots["is_goal"]
  model = LogisticRegression().fit(X, y)
  shots["xg_predicted"] = model.predict_proba(X)[:, 1]
  ```
- **Passing Networks**: Visualizar quién pasa a quién con GraƒX de nodos (posición promedio) y aristas (frecuencia de pase). Muy útil para el EDA de events.
- **Simulación Poisson**: Modelo de Dixon-Coles para predecir goles: λ_home = α_home × β_away × γ. Base del Modelo 2.
- **Pitch Control**: Modelo de cuánto campo controla cada equipo instante a instante (requiere datos de tracking, no de eventos).

**📌 Clave:** Sumpter demuestra que las **redes de pase descentralizadas** (donde el balón circula entre más jugadores) correlacionan con mejor rendimiento. España e Italia en Eurocopa 2012 son sus ejemplos estrella.

**Recursos:** [GitHub del libro](https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython)

---

### 🎥 2. Friends of Tracking — Canal de YouTube

**Académicos de fútbol analytics** | David Sumpter (Uppsala), Javier Fernandez (Barcelona), Laurie Shaw (Harvard).

[youtube.com/@friendsoftracking755](https://www.youtube.com/@friendsoftracking755) | [Código GitHub](https://github.com/Friends-of-Tracking-Data-FoTD/)

**Tutoriales más relevantes para nuestro proyecto:**

| Tema | Aplicación directa |
|---|---|
| **Implementación de xG** | Modelo 1 — paso a paso con datos Statsbomb |
| **Expected Threat (xT)** | Feature avanzada: cuánto aumenta la prob. de gol cada acción |
| **Pitch Control** | Visualizar dominio táctico (requiere tracking data) |
| **Passing Networks** | Grafo de conexiones por partido — EDA de events |
| **Poisson Regression** | Predecir goles por partido — base Modelo 2 |
| **Pressing Metrics** | PPDA (Passes Per Defensive Action) — intensidad del pressing |

**💡 Concepto destacado — xT (Expected Threat):**
A diferencia del xG (solo tiros), el xT asigna un **valor a cada acción con balón** dependiendo de cuánto aumenta la probabilidad de gol en los siguientes ~5 acciones. Es la métrica más moderna para evaluar a jugadores que no marcan ni asisten.
```
xT(x, y) = probabilidad de marcar en los próximos 5 eventos si el balón está en (x,y)
```

**💡 Concepto destacado — PPDA:**
```
PPDA = Pases permitidos al rival / Acciones defensivas propias (en campo contrario)
```
Liverpool bajo Klopp tenía PPDA ~7 (pressing intensísimo). Equipos defensivos tienen PPDA >15.

---

### 📊 3. StatsBomb Open Data

**Datos event-level gratuitos** | Formato similar a nuestros datos de WhoScored/Opta.

[github.com/statsbomb/open-data](https://github.com/statsbomb/open-data)

**¿Por qué es valioso?**
- Incluye datos de **La Liga, Champions League, Copa Mundial, NWSL** con +22,000 partidos.
- El formato JSON incluye campos como `freeze_frame` (posiciones de todos los jugadores en el momento del tiro) — esto permite calcular si había portero fuera de posición, número de defensores en la línea, etc.
- **Librería oficial:** `statsbombpy` — permite cargar datos directamente con `from statsbombpy import sb`.

**Campos exclusivos de StatsBomb vs nuestros datos:**

| Campo StatsBomb | Equivalente nuestro | Diferencia |
|---|---|---|
| `shot.freeze_frame` | No disponible | Posiciones de todos los jugadores en el tiro 🔥 |
| `shot.xg` | Lo calculamos | StatsBomb calcula xG propio más preciso |
| `ball_receipt.outcome` | `outcome` | Similar |
| `pass.recipient` | No disponible | Quién recibió el pase |
| `carry` | No disponible | Conducciones de balón |

**💡 Tip:** Podemos usar StatsBomb Open Data para **comparar y validar** nuestro modelo de xG propio. Si nuestro AUC-ROC es similar al de sus datos, sabemos que la metodología es correcta.

```python
# Instalar en el entorno
pip install statsbombpy

# Cargar datos gratuitos de Premier League
from statsbombpy import sb
matches = sb.matches(competition_id=2, season_id=27)  # PL 2015/16
events = sb.events(match_id=3764453)
shots  = events[events["type"] == "Shot"]
```

---

### 📗 4. Expected Goals Philosophy — James Tippett

**Libro accesible** | Tippett trabajó en Smartodds, la consultora de apuestas que popularizó el xG en la industria.

**Insights clave para nuestro proyecto:**

1. **xG para apuestas**: Tippett demuestra que los equipos que consistentemente superan su xG (efficiency) **revertan a la media** en el largo plazo. Un equipo que tiene xG_against < goles_recibidos está siendo "afortunado" defensivamente.

2. **El caso Brentford**: Usaron xG para fichar jugadores infravalorados en ligas menores con alta conversión pero estadísticas brutas mediocres — similar a Moneyball en béisbol.

3. **xG ≠ suerte**: La diferencia entre xG y goles reales a lo largo de una temporada mide la habilidad de finalización (shooting skill) y la suerte del portero. Esto es una **feature poderosa para el Modelo 2**:
   ```python
   # Underperformance / Overperformance de xG
   matches["home_xg_diff"] = matches["home_goals"] - matches["home_xg"]
   matches["away_xg_diff"] = matches["away_goals"] - matches["away_xg"]
   # Si un equipo hace home_xg_diff < 0 consistentemente, tienen un mal portero o mala suerte
   ```

4. **Las cuotas ya incorporan xG**: La industria de apuestas lleva +15 años usando xG en sus modelos. Por eso es tan difícil superarlos — nuestro modelo necesita capturar algo que ellos no tengan (información local, estado del vestuario, rotaciones).

---

## 9. 🗺️ Hoja de Ruta de Visualizaciones (Football Analytics Avanzado)

Visualizaciones que elevarían este proyecto al nivel de dashboards profesionales como FotMob o Sofascore:

| Visualización | Dataset | Librería | Dificultad | Impacto |
|---|---|---|---|---|
| **xG Shot Map** (burbujas escaladas) | events | mplsoccer | ⭐⭐ | 🔥🔥🔥 |
| **Pass Network** (grafo por partido) | events | networkx + mplsoccer | ⭐⭐⭐ | 🔥🔥🔥 |
| **Radar Chart** (8 métricas por jugador) | players | matplotlib | ⭐⭐⭐ | 🔥🔥🔥 |
| **xG Timeline por partido** | events + matches | matplotlib | ⭐⭐ | 🔥🔥🔥 |
| **xG vs Goals scatter** (over/underperformers) | matches | seaborn | ⭐ | 🔥🔥 |
| **Progressive Pass Map** (flechas) | events | mplsoccer | ⭐⭐⭐ | 🔥🔥 |
| **Carry Map** (conducciones) | events | mplsoccer | ⭐⭐⭐⭐ | 🔥🔥 |
| **Shot Placement Heatmap** (portería) | events | seaborn | ⭐⭐ | 🔥🔥 |
| **Betting Calibration Curve** | matches | sklearn | ⭐⭐ | 🔥🔥🔥 |
| **Expected Threat (xT) Heatmap** | events | custom model | ⭐⭐⭐⭐⭐ | 🔥🔥🔥🔥 |

> 💡 **Recomendación**: Para el **dashboard del Taller2**, implementar mínimo los primeros 3 (xG Shot Map + xG Timeline + Radar Chart). Son los más impactantes visualmente y los más directamente relacionados con los modelos.

```python
# Instalar mplsoccer para visualizaciones pro
pip install mplsoccer

# Ejemplo Shot Map básico
from mplsoccer import Pitch
pitch = Pitch(pitch_color='grass', line_color='white')
fig, ax = pitch.draw()
# Escalar burbuja por xG predicho
sc = ax.scatter(shots["x"], shots["y"],
                s=shots["xg_predicted"] * 500,
                c=shots["is_goal"].map({1: "gold", 0: "grey"}),
                alpha=0.6)
```

---

## 10. 🚀 Features Creativas y Originales

> Estas features van **más allá del estándar**. Son el tipo de ideas que la literatura académica sí conoce pero rara vez se implementan en proyectos universitarios. Su originalidad y fundamentación es lo que puede hacer la diferencia en la nota.

---

### 🧪 A — "Defensive Pressure Proxy" *(simulando freeze_frame de StatsBomb)*
**Inspiración:** StatsBomb incluye el `freeze_frame` (posición de todos los jugadores en el momento del tiro). Nosotros no lo tenemos, pero **podemos aproximarlo** contando acciones defensivas rivales en la misma zona y minuto.

**Por qué importa:** Un tiro desde el mismo punto geométrico tiene xG muy diferente si hay 0 vs 3 defensores cerca. Esta variable capture ese contexto que los modelos básicos ignoran.

```python
def defensive_pressure(row, all_events, radius=10):
    nearby_rivals = all_events[
        (all_events["match_id"] == row["match_id"]) &
        (all_events["minute"] == row["minute"]) &
        (all_events["team_name"] != row["team_name"]) &
        (abs(all_events["x"] - row["x"]) < radius) &
        (abs(all_events["y"] - row["y"]) < radius)
    ]
    return len(nearby_rivals)
shots["defensive_pressure"] = shots.apply(lambda r: defensive_pressure(r, events), axis=1)
```

---

### 🧪 B — "Buildup Quality Index" *(el contexto antes del tiro)*
**Inspiración:** Soccermatics muestra que cómo se crea la ocasión importa tanto como dónde ocurre. Un tiro tras 8 pases exitosos tiene diferente xG que uno de primer contacto tras un despeje.

```python
# ¿Cuántos pases exitosos tuvo el equipo en el minuto previo al tiro?
shots["buildup_passes"] = shots.apply(lambda row:
    len(events[(events["match_id"] == row["match_id"]) &
               (events["minute"].between(row["minute"]-1, row["minute"])) &
               (events["team_name"] == row["team_name"]) &
               (events["event_type"] == "Pass") &
               (events["outcome"] == "Successful")]), axis=1)
```

---

### 🧪 C — "Portería Zone" *(dónde exactamente fue el tiro)*
**Inspiración:** Usamos `goal_mouth_y` y `goal_mouth_z` ya disponibles. Dividimos la portería en 9 zonas (3×3) y las codificamos. Los porteros no cubren igual las 9 zonas — los corners altos son los más difíciles.

```python
def porteria_zone(row):
    if pd.isna(row["goal_mouth_y"]) or pd.isna(row["goal_mouth_z"]):
        return "unknown"
    y_zone = "left" if row["goal_mouth_y"] < 33 else ("right" if row["goal_mouth_y"] > 67 else "center")
    z_zone = "high" if row["goal_mouth_z"] > 0.6 else "low"
    return f"{y_zone}_{z_zone}"
shots["porteria_zone"] = shots.apply(porteria_zone, axis=1)
```

---

### 🧪 D — "xG Debt" *(ineficiencia del mercado — Tippett)*
**Inspiración:** James Tippett demostró que los equipos que generan más xG del que convierten están "debiendo goles". Esta regresión a la media **no siempre la precia bien el mercado de apuestas**.

```python
# Feature para Modelo 2: ¿Cuánto le debe el azar a este equipo?
matches["home_xg_debt_5"] = (
    matches.groupby("home_team")["home_xg"].transform(lambda x: x.rolling(5).mean())
  - matches.groupby("home_team")["fthg"].transform(lambda x: x.rolling(5).mean())
)
# > 0 → el equipo genera más xG de lo que marca → regreserá a la media
```

---

### 🧪 E — "PPDA Proxy" *(pressing intensity desde events)*
**Inspiración:** Friends of Tracking — el PPDA (Passes Per Defensive Action) es la métrica de pressing más usada en el análisis profesional. **Podemos calcularlo solo con nuestros eventos.** Liverpool de Klopp: PPDA ~7. Equipos defensivos: PPDA >15.

```python
# PPDA = pases rivales en zona ofensiva / acciones defensivas nuestras allí
def ppda_by_match(match_events, team):
    opp_passes = len(match_events[
        (match_events["team_name"] != team) & (match_events["event_type"] == "Pass") &
        (match_events["outcome"] == "Successful") & (match_events["x"] > 40)])
    own_def    = len(match_events[
        (match_events["team_name"] == team) &
        (match_events["event_type"].isin(["Tackle","Interception","BlockedPass"])) &
        (match_events["x"] > 40)])
    return opp_passes / max(own_def, 1)
```

---

### 🧪 F — "Passing Decentralization Index" *(Sumpter's key finding)*
**Inspiración:** David Sumpter demostró que España/Italia ganaron Euro 2012 gracias a redes de pase más descentralizadas — más jugadores distintos tocando el balón = más resistentes tácticamente. **Podemos medir esto en cada partido.**

```python
decentralization = events[
    (events["event_type"] == "Pass") & (events["outcome"] == "Successful")
].groupby(["match_id", "team_name"])["player_id"].nunique().reset_index()
decentralization.columns = ["match_id", "team_name", "pass_decentralization"]
```

---

### 🧪 G — "Momentum Oscillator" *(MACD del fútbol — finanzas aplicadas al deporte)*
**Inspiración 100% original:** El MACD en trading financiero detecta cuándo un activo está acelerando. Aplicado al fútbol: cuando la forma-3 es mayor que la forma-10, el equipo está en punto de inflexión ascendente — correlaciona con victorias inesperadas.

```python
matches["home_form_3"]  = matches.groupby("home_team")["fthg"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
matches["home_form_10"] = matches.groupby("home_team")["fthg"].transform(
    lambda x: x.shift(1).rolling(10, min_periods=3).mean())
matches["home_momentum"] = matches["home_form_3"] - matches["home_form_10"]
# > 0 → equipo ascendente (hot) | < 0 → equipo en declive (cold)
```

---

### 🧪 H — "Referee Bias Index" *(análisis forense de árbitros)*
**Por qué es valioso:** Algunos árbitros muestran sesgo estadístico hacia el equipo local (más penaltis, menos tarjetas al local). Esto es **medible** y **el mercado de apuestas no siempre lo captura completamente**.

```python
ref_stats = matches.groupby("referee").agg(
    home_win_rate=("ftr", lambda x: (x == "H").mean()),
    avg_yellow_home=("hy", "mean"),
    avg_yellow_away=("ay", "mean")
)
ref_stats["referee_home_bias"] = ref_stats["avg_yellow_away"] - ref_stats["avg_yellow_home"]
matches = matches.merge(ref_stats[["referee_home_bias"]], left_on="referee", right_index=True)
```

---

### 🧪 I — "Altitude of Play" *(dónde vive el balón del equipo)*
**Idea original:** El promedio de coordenada X de todos los eventos de un equipo en un partido indica si juegan alto (pressing) o bajo (repliegue). Es el "centro de gravedad táctico" del equipo.

```python
altitude = events.groupby(["match_id","team_name"])["x"].mean().reset_index()
altitude.columns = ["match_id","team_name","altitude_of_play"]
# > 60 = juego en campo rival (ofensivo) | < 45 = repliegue defensivo
```

---

### 🧪 J — "Clutch Factor" *(rendimiento bajo presión real)*
**Idea:** ¿Cuántos goles marca un equipo en los últimos 15 minutos vs los primeros 75? Teams with `clutch_ratio > 1` son "killers" — marcan cuando más importa. Esto **no está en ninguna cuota de Bet365**.

```python
late_goals  = events[(events["is_goal"]==True) & (events["minute"]>=75)
    ].groupby(["match_id","team_name"]).size().rename("late_goals")
early_goals = events[(events["is_goal"]==True) & (events["minute"]<75)
    ].groupby(["match_id","team_name"]).size().rename("early_goals")
clutch = pd.concat([late_goals, early_goals], axis=1).fillna(0)
clutch["clutch_ratio"] = clutch["late_goals"] / (clutch["early_goals"] + 1)
```

---

> 📖 **Para implementación completa:** Ver [`hoja_de_ruta_modelamiento.md`](hoja_de_ruta_modelamiento.md) con arquitecturas de modelos, pipelines de entrenamiento, validación temporal y checklist del Taller2.

---

*Documento actualizado con investigación de fuentes Taller2 ML1-2026 (Soccermatics, Friends of Tracking, StatsBomb, xG Philosophy).*

