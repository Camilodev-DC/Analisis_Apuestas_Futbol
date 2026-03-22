# 🗺️ Hoja de Ruta — Modelamiento ML Premier League

> **Proyecto:** Taller 2 — ¿Puedes Predecir el Fútbol Mejor que las Casas de Apuestas?
> **Curso:** Machine Learning 1 (ML1-2026) — Universidad Externado de Colombia
> **Última actualización:** 21/03/2026

---

## 🧭 Visión General del Pipeline

```
data/raw/ → Feature Engineering → Modelo 1 (xG) → Modelo 2 (Match Predictor) → Dashboard Web
```

Los dos modelos son complementarios: **el output del Modelo 1 alimenta al Modelo 2**.

```
events.csv ──[xG por tiro]──→ xg_per_match (home/away) ──→ matches_enriched.csv ──→ Modelo 2
```

---

## 📅 Fases del Proyecto

| Fase | Nombre | Estado | Entregable |
|---|---|---|---|
| ✅ 0 | Datos + EDA | Completo | `events.csv` con qualifiers, reportes EDA |
| 🔄 1 | Feature Engineering | En curso | `data/processed/features_xg.csv`, `features_matches.csv` |
| ⬜ 2 | Modelo 1 — xG | Pendiente | `src/models/xg_model.py`, AUC-ROC > 0.78 |
| ⬜ 3 | Modelo 2 — Match Predictor | Pendiente | `src/models/match_predictor.py`, Acc > 49.8% |
| ⬜ 4 | Evaluación y Calibración | Pendiente | Comparación vs Bet365 |
| ⬜ 5 | Dashboard Web | Pendiente | Deploy en Vercel/Netlify con URL pública |

---

## ⚙️ Fase 1 — Feature Engineering Creativo

### 1A. Features Estándar (Taller2 — obligatorias)
```python
# Script: scripts/feature_engineering.py

import pandas as pd
import numpy as np

shots = events[events["is_shot"] == True].copy()

# ── Geométricas (Taller2 obligatorias) ─────────────────────────────────
shots["distance_to_goal"] = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
shots["angle_to_goal"]    = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))
shots["dist_squared"]     = shots["distance_to_goal"] ** 2
shots["dist_angle"]       = shots["distance_to_goal"] * shots["angle_to_goal"]
shots["is_in_area"]       = (shots["x"] > 83).astype(int)
shots["is_central"]       = shots["y"].between(33, 67).astype(int)

# ── Qualifiers booleanos ────────────────────────────────────────────────
q = shots["qualifiers"].astype(str)
shots["is_big_chance"]  = q.str.contains("BigChance",  na=False).astype(int)
shots["is_header"]      = q.str.contains('"Head"',     na=False).astype(int)
shots["is_right_foot"]  = q.str.contains("RightFoot",  na=False).astype(int)
shots["is_left_foot"]   = q.str.contains("LeftFoot",   na=False).astype(int)
shots["is_counter"]     = q.str.contains("FastBreak",  na=False).astype(int)
shots["from_corner"]    = q.str.contains("FromCorner", na=False).astype(int)
shots["is_penalty"]     = q.str.contains('"Penalty"',  na=False).astype(int)
shots["is_volley"]      = q.str.contains("Volley",     na=False).astype(int)
shots["first_touch"]    = q.str.contains("FirstTouch", na=False).astype(int)
shots["is_set_piece"]   = q.str.contains("SetPiece",   na=False).astype(int)
```

---

### 1B. Features Avanzadas 🔬 (Originales / No estándar)

#### 🧪 Feature A — "Defensive Pressure Proxy" (simulando freeze_frame)
> **Idea:** Approximamos cuántos defensores había cerca del tirador mirando acciones del equipo contrario en el mismo minuto y zona.

```python
def defensive_pressure(row, all_events, radius=10):
    same_match = all_events[all_events["match_id"] == row["match_id"]]
    same_moment = same_match[same_match["minute"] == row["minute"]]
    rival_actions = same_moment[same_moment["team_name"] != row["team_name"]]
    nearby = rival_actions[
        (abs(rival_actions["x"] - row["x"]) < radius) &
        (abs(rival_actions["y"] - row["y"]) < radius)
    ]
    return len(nearby)

shots["defensive_pressure"] = shots.apply(
    lambda row: defensive_pressure(row, events), axis=1)
```

> **Por qué importa:** StatsBomb incluye freeze_frame exactamente por esto. Nosotros lo aproximamos. Un tiro con pressure=0 tiene xG muy diferente que uno con pressure=3.

---

#### 🧪 Feature B — "Buildup Quality Index" (inspirada en Soccermatics)
> **Idea:** Los últimos 3 eventos antes del tiro dicen mucho sobre la calidad de la ocasión. ¿Fue un contraataque de 2 pases? ¿O una jugada elaborada de 8 toques?

```python
def buildup_quality(row, all_events, window_secs=60):
    same_match = all_events[all_events["match_id"] == row["match_id"]]
    same_team  = same_match[same_match["team_name"] == row["team_name"]]
    buildup    = same_team[
        (same_team["minute"] == row["minute"]) |
        ((same_team["minute"] == row["minute"] - 1) & (same_team["second"] > 0))
    ]
    passes_ok = len(buildup[(buildup["event_type"] == "Pass") &
                             (buildup["outcome"] == "Successful")])
    unique_players = buildup["player_id"].nunique()
    return pd.Series({
        "buildup_passes": passes_ok,
        "buildup_unique_players": unique_players,
        "buildup_decentralized": int(unique_players > 3)
    })
```

---

#### 🧪 Feature C — "Portería Zone" desde goal_mouth_y / goal_mouth_z
> **Idea:** Usar las coordenadas donde el tiro impactó en portería para crear zonas. Los porteros no cubren igual las 9 zonas.

```python
# Dividir portería en 3×3 = 9 zonas
def porteria_zone(row):
    if pd.isna(row["goal_mouth_y"]) or pd.isna(row["goal_mouth_z"]):
        return "unknown"
    y_zone = "left" if row["goal_mouth_y"] < 33 else ("right" if row["goal_mouth_y"] > 67 else "center")
    z_zone = "high" if row["goal_mouth_z"] > 0.6 else "low"
    return f"{y_zone}_{z_zone}"

shots["porteria_zone"] = shots.apply(porteria_zone, axis=1)
# Convertir a dummies: porteria_zone_left_low, porteria_zone_center_high, etc.
shots = pd.get_dummies(shots, columns=["porteria_zone"], prefix="pz")
```

> **Por qué es mind-blowing:** Podemos cruzar la zona de portería con la predicción del xG y crear un "shot placement score" que mide cuánto el jugador "elige" zonas difíciles para el portero.

---

#### 🧪 Feature D — "xG Debt" (Tippett-inspired — market inefficiency)
> **Idea:** Si un equipo genera más xG del que convierte en los últimos 5 partidos, está "debiendo goles" y es más probable que los marque pronto.

```python
# A nivel de partido (para el Match Predictor)
matches = matches.sort_values(["home_team", "date"])

matches["home_xg_debt_5"] = (
    matches.groupby("home_team")["home_xg"].transform(
        lambda x: x.rolling(5, min_periods=2).mean())
    - matches.groupby("home_team")["fthg"].transform(
        lambda x: x.rolling(5, min_periods=2).mean())
)
# Positivo → el equipo genera más xG de lo que marca → pronto "pagará la deuda"
# Negativo → el equipo sobreanota vs su xG → "prestamos tomados" que se devolverán
```

---

#### 🧪 Feature E — "PPDA Proxy" (Pressing Intensity desde events)
> Ideada por Friends of Tracking. Nadie en un proyecto universitario la calcula.

```python
def ppda_by_match(match_events, team):
    """
    PPDA = Pases rivales permitidos en campo ofensivo / Acciones defensivas propias allí
    Mayor PPDA = Pressing más bajo / Defensivo
    Menor PPDA = Pressing intensísimo (Liverpool ~7, Guardiola City ~8)
    """
    opp_passes_in_attack = match_events[
        (match_events["team_name"] != team) &
        (match_events["event_type"] == "Pass") &
        (match_events["outcome"] == "Successful") &
        (match_events["x"] > 40)  # Campo propio del equipo que presiona
    ]
    own_def_actions = match_events[
        (match_events["team_name"] == team) &
        (match_events["event_type"].isin(["Tackle", "Interception", "BlockedPass", "Clearance"])) &
        (match_events["x"] > 40)
    ]
    return len(opp_passes_in_attack) / max(len(own_def_actions), 1)

# Aplicar por partido
ppda_results = events.groupby(["match_id", "team_name"]).apply(
    lambda g: ppda_by_match(events[events["match_id"] == g["match_id"].iloc[0]], g["team_name"].iloc[0])
).reset_index()
ppda_results.columns = ["match_id", "team_name", "ppda"]
```

---

#### 🧪 Feature F — "Passing Decentralization Index" (Sumpter's key finding)
> Teams that circulate the ball through MORE different players win more consistently.

```python
# ¿Cuántos jugadores únicos tocaron el balón en pases exitosos por equipo por partido?
decentralization = events[
    (events["event_type"] == "Pass") &
    (events["outcome"] == "Successful")
].groupby(["match_id", "team_name"])["player_id"].nunique().reset_index()
decentralization.columns = ["match_id", "team_name", "pass_decentralization"]
# Mayor valor → más jugadores distintos participan → estilo colectivo (Sumpter-validated)
```

---

#### 🧪 Feature G — "Momentum Oscillator" (MACD del fútbol — 100% original)
> Tomado directamente del trading financiero: cuando la media corta cruza la media larga hacia arriba → señal alcista.

```python
# MACD Football: forma_3 - forma_10 = "¿está acelerando o desacelerando?"
for team_col in ["home_team", "away_team"]:
    prefix = "home" if "home" in team_col else "away"
    goal_col = "fthg" if "home" in team_col else "ftag"
    
    matches[f"{prefix}_form_3"] = matches.groupby(team_col)[goal_col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    matches[f"{prefix}_form_10"] = matches.groupby(team_col)[goal_col].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    matches[f"{prefix}_momentum"] = (
        matches[f"{prefix}_form_3"] - matches[f"{prefix}_form_10"])
    # > 0 → en racha ascendente (hot team)
    # < 0 → en declive (cold team)
```

---

#### 🧪 Feature H — "Referee Bias Index" (análisis forense de árbitros)
> Algunos árbitros favorecen estadísticamente al equipo local. Esto es medible.

```python
# ¿Cuál es la tasa de victoria local cuando arbitra X?
ref_stats = matches.groupby("referee").agg(
    total_matches=("id", "count"),
    home_wins=("ftr", lambda x: (x == "H").sum()),
    home_win_rate=("ftr", lambda x: (x == "H").mean()),
    avg_yellow_home=("hy", "mean"),
    avg_yellow_away=("ay", "mean"),
    home_bias=("hy", lambda x: matches.loc[x.index, "ay"].mean() - x.mean())
    # Positivo → árbitro da más amarillas a visitante (favorable al local)
)
matches = matches.merge(ref_stats[["home_win_rate", "home_bias"]], 
                         left_on="referee", right_index=True)
```

---

#### 🧪 Feature I — "Altitude of Play" (dónde vive el balón)
> No es sobre cuántos pases, sino **en qué zona del campo** vive el balón típicamente. Una métrica del estilo de juego real vs nominal.

```python
# Posición promedio de los eventos del equipo en cada partido
altitude = events[events["event_type"].isin(["Pass", "BallRecovery", "Tackle"])].groupby(
    ["match_id", "team_name"])["x"].mean().reset_index()
altitude.columns = ["match_id", "team_name", "altitude_of_play"]
# > 60 = equipo que juega alto (presión alta, estilo ofensivo)
# < 45 = equipo que se repliega (espera en su campo)
```

---

#### 🧪 Feature J — "Clutch Factor" (rendimiento bajo presión)
> ¿Cómo rinde cada equipo en los minutos 75-90 comparado con 0-75? Los equipos con más goles en el tramo final son más "clutch".

```python
late_goals = events[
    (events["is_goal"] == True) &
    (events["minute"] >= 75)
].groupby(["match_id", "team_name"]).size().reset_index(name="late_goals")

early_goals = events[
    (events["is_goal"] == True) &
    (events["minute"] < 75)
].groupby(["match_id", "team_name"]).size().reset_index(name="early_goals")

# Equipos con clutch_ratio > 1 → tienden a mejorar al final del partido
```

---

## 🤖 Fase 2 — Modelo 1: Expected Goals (xG)

### Objetivo
Predecir la probabilidad de gol para cada tiro (`is_goal ~ 1`).

### Features de entrada
```python
FEATURES_XG = [
    # Obligatorias Taller2
    "distance_to_goal", "angle_to_goal", "dist_squared", "dist_angle",
    "is_in_area", "is_central",
    # Qualifiers
    "is_big_chance", "is_header", "is_right_foot", "is_left_foot",
    "is_counter", "from_corner", "is_penalty", "is_volley",
    "first_touch", "is_set_piece",
    # Avanzadas (originales)
    "defensive_pressure", "buildup_passes", "buildup_decentralized",
    "minute",
]
```

### Arquitectura del Modelo
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Baseline (Taller2 mínimo): Logistic Regression
pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=500, C=0.1))
])

# Avanzado: Gradient Boosting (XGBoost style)
pipeline_gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10
)
```

### Estrategia de Evaluación
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

# NO cross-validation aleatoria — usar splits temporales o por partido
# para evitar data leakage (un partido no debería entrenarse y testarse)
cv = StratifiedKFold(n_splits=5, shuffle=False)  # temporal order preserved
```

### 🎯 Benchmarks
| Métrica | Mínimo Taller2 | Objetivo ambicioso |
|---|---|---|
| AUC-ROC | **0.78** | >0.85 |
| Brier Score | <0.10 | <0.08 |
| Log Loss | <0.35 | <0.28 |

---

## 🏆 Fase 3 — Modelo 2: Match Predictor

### Objetivo
Predecir: **H** (victoria local), **D** (empate), **A** (victoria visitante).

### Features de entrada
```python
FEATURES_MATCH = [
    # Cuotas (mejor predictor único — Tippett)
    "implied_prob_h", "implied_prob_d", "implied_prob_a",
    # xG generado por el Modelo 1 (rolling últimas 5 jornadas)
    "home_xg_avg5", "away_xg_avg5",
    "home_goals_avg5", "away_goals_avg5",
    "home_goals_conceded_avg5", "away_goals_conceded_avg5",
    # Features originales
    "home_xg_debt_5",        # 🧪 xG Debt (Tippett)
    "home_ppda",             # 🧪 Pressing intensity
    "home_momentum",         # 🧪 MACD momentum
    "home_decentralization", # 🧪 Sumpter passing 
    "referee_home_bias",     # 🧪 Árbitro
    "home_altitude",         # 🧪 Dónde vive el balón
    "home_clutch_factor",    # 🧪 Rendimiento bajo presión
    "was_home",              # Ventaja local (+0.26 goles en PL)
]
```

### Arquitectura
```python
from sklearn.ensemble import RandomForestClassifier
import xgboost as XGBClassifier

# Opción 1: Random Forest (interpretable, robusto)
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, 
    class_weight="balanced", min_samples_leaf=5,
    random_state=42
)

# Opción 2: XGBoost (más potente, requiere tuning)
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss"
)

# Opción 3 (avanzada): Ensemble con votación ponderada
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[("lr", pipeline_lr), ("rf", rf), ("xgb", xgb)],
    voting="soft"  # usa probabilidades, no clases
)
```

### 🎯 Benchmarks vs Bet365
```python
# Bet365 accuracy en esta temporada
BET365_BENCHMARK = 0.498  # 49.8%

# Calcular ROI simulado
def simulate_roi(y_true, y_pred_proba, odds_df, stake=1.0):
    """Simula apostar 1 unidad cuando el modelo da prob > implied_prob"""
    profit = []
    for i, (true, pred) in enumerate(zip(y_true, y_pred_proba)):
        predicted_class = np.argmax(pred)  # H, D, A
        implied = [odds_df.iloc[i]["implied_h"],
                   odds_df.iloc[i]["implied_d"],
                   odds_df.iloc[i]["implied_a"]][predicted_class]
        odd = [odds_df.iloc[i]["b365h"],
               odds_df.iloc[i]["b365d"],
               odds_df.iloc[i]["b365a"]][predicted_class]
        
        if pred[predicted_class] > implied + 0.05:  # Edge mínimo de 5%
            profit.append(odd * stake - stake if true == predicted_class else -stake)
    return sum(profit) / len(profit)  # ROI promedio por apuesta
```

---

## 📊 Fase 4 — Validación y Calibración

### Plots obligatorios
```python
# 1. Calibration Curve — ¿El modelo sobreestima o subestima xG?
from sklearn.calibration import CalibrationDisplay
CalibrationDisplay.from_predictions(y_test, xg_proba, n_bins=10)

# 2. Confusion Matrix para Match Predictor
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["H","D","A"])

# 3. Feature Importance
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # 🔥 SHAP — mejor que feature_importance
```

---

## 🌐 Fase 5 — Dashboard Web (Requisito Taller2)

### Stack recomendado
```
Frontend: Streamlit (Python puro, deploy en Streamlit Cloud gratis) o
          Next.js + Recharts (más visual, deploy en Vercel)
Backend:  Modelos guardados con joblib/pickle, cargados en memoria
Deploy:   Streamlit Cloud / Vercel / Netlify — URL pública obligatoria para nota
```

### Secciones mínimas (Taller2)
1. **🎯 Shot Map** — Campo interactivo con tiros escalados por xG predicho
2. **⚔️ Match Predictor** — Select 2 equipos → probabilidades H/D/A + goles esperados
3. **📊 Performance** — Accuracy, AUC-ROC, confusion matrix vs Bet365
4. **🔍 EDA** — 3 visualizaciones interactivas con nuestros datos

### Ideas de visualización impactante para el dash
```python
# En Streamlit:
import streamlit as st
from mplsoccer import Pitch

# Shot map interactivo por equipo
equipo = st.selectbox("Seleccionar equipo", teams)
partido = st.selectbox("Seleccionar partido", matches)
# → Dibuja el campo con tiros filtrados, escalados por xG del modelo
```

---

## 🛠️ Scripts por desarrollar

| Script | Propósito | Inputs | Outputs |
|---|---|---|---|
| `scripts/feature_engineering.py` | Generar todas las features | `events.csv`, `matches.csv` | `features_xg.csv`, `features_matches.csv` |
| `src/models/xg_model.py` | Entrenar Modelo 1 | `features_xg.csv` | `models/xg_model.pkl` |
| `src/models/match_predictor.py` | Entrenar Modelo 2 | `features_matches.csv` | `models/match_model.pkl` |
| `src/models/evaluate.py` | Métricas + benchmark | Modelos + test sets | `reports/metrics.json` |
| `dashboard/app.py` | Dashboard Streamlit | Modelos pkl | URL pública |

---

## 🎓 Checklist de Evaluación Taller2

- [ ] Modelo 1 (xG) con AUC-ROC > 0.78
- [ ] Modelo 2 (Match Predictor) con Accuracy > 49.8%
- [ ] Dashboard deployed con URL pública funcionando
- [ ] Shot Map visual con xG predicho (no hardcoded)
- [ ] Interfaz de predicción de partido (H/D/A + goles)
- [ ] Metrics display (accuracy, confusion matrix, vs Bet365)
- [ ] Mínimo 3 visualizaciones exploratorias de datos
- [ ] Presentación del proceso de feature engineering

---

*Documento creado el 21/03/2026 | Basado en Soccermatics, Friends of Tracking, StatsBomb Open Data y Expected Goals Philosophy.*
