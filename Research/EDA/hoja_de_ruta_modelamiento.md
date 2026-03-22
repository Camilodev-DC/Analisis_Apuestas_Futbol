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

## ⚙️ Fase 1C — Feature Engineering Creativo para Modelo 2 (Match Predictor) 🏆

> Estas features se construyen a nivel de **partido** cruzando `matches.csv`, `events.csv`, `players.csv` y `player_history.csv`. El objetivo: capturar lo que las cuotas de Bet365 **no ven** o **capturan tarde**.

---

### 🏦 M1 — "xG Breakdown por Tipo de Jugada"
> **Idea:** Las cuotas saben cuántos goles marca un equipo, pero ¿saben si vienen de jugada corrida, set-pieces o contras? Equipos con xG alto desde corners son vulnerables cuando el rival es alto. Es un **matchup táctico** que el mercado no siempre prica.

```python
# Agregar xG por tipo de situación desde events.csv (usando qualifiers)
q = events["qualifiers"].astype(str)
events["from_set_piece"] = q.str.contains("SetPiece|FromCorner|Penalty", na=False).astype(int)
events["from_counter"]   = q.str.contains("FastBreak", na=False).astype(int)
events["from_open_play"] = (~events["from_set_piece"].astype(bool)).astype(int)

xg_breakdown = events[events["is_shot"]==True].groupby(["match_id","team_name"]).agg(
    xg_total=("xg_predicted", "sum"),          # Del Modelo 1
    xg_open_play=("xg_open_play_flag", "sum"),  # xG de jugada abierta
    xg_set_piece=("xg_set_piece_flag", "sum"),  # xG de balón parado
    xg_counter=("xg_counter_flag", "sum"),      # xG de contra
    big_chances=("is_big_chance","sum")
).reset_index()
```

---

### 📊 M2 — "Shot Quality Premium" *(el verdadero estilo de juego)*
> **Idea:** Dos equipos pueden marcar 2 goles por partido, pero uno lo hace con 12 tiros de alta calidad y el otro con 20 tiros mediocres. El `mean_xg_per_shot` mide eficiencia táctica — y predice sostenibilidad.

```python
# xG medio por tiro (calidad del tiro, no cantidad)
shot_quality = events[events["is_shot"]==True].groupby(["match_id","team_name"]).agg(
    shots=("is_shot", "count"),
    xg_total=("xg_predicted", "sum"),
    mean_xg_per_shot=("xg_predicted", "mean"),   # ← THE FEATURE
    shots_on_target=("outcome", lambda x: (x=="SavedShot").sum())
).reset_index()
shot_quality["shot_on_target_rate"] = shot_quality["shots_on_target"] / shot_quality["shots"]

# Rolling: media de las últimas 5 jornadas
shot_quality_roll = shot_quality.groupby("team_name")["mean_xg_per_shot"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=2).mean())
```

---

### ⏳ M3 — "Fatigue & Rest Days Proxy" *(fisiología del deporte)*
> **Inspiración:** Estudios de medicina deportiva muestran caída del rendimiento con <4 días de recuperación entre partidos. En el calendario de PL, esto ocurre regularmente en diciembre y enero. **Bet365 sí lo modela parcialmente, pero mal a nivel granular.**

```python
# Días desde el último partido (proxy de fatiga)
matches["date"] = pd.to_datetime(matches["date"])
matches = matches.sort_values("date")

def days_since_last_match(team, current_date, all_matches):
    prev = all_matches[
        ((all_matches["home_team"] == team) | (all_matches["away_team"] == team)) &
        (all_matches["date"] < current_date)
    ]["date"].max()
    return (current_date - prev).days if pd.notna(prev) else 7

matches["home_rest_days"] = matches.apply(
    lambda r: days_since_last_match(r["home_team"], r["date"], matches), axis=1)
matches["away_rest_days"] = matches.apply(
    lambda r: days_since_last_match(r["away_team"], r["date"], matches), axis=1)
matches["rest_advantage"] = matches["home_rest_days"] - matches["away_rest_days"]
# > 0 → local más descansado; < 0 → visitante más descansado
```

---

### 📉 M4 — "Form Volatility" *(el equipo impredecible)*
> **Idea radical:** No es solo la forma en sí, sino la **varianza** de esa forma. Un equipo con media 1.5 goles últimos 5 partidos pero std=1.8 es MUCHO más impredecible que uno con media 1.5 y std=0.3. La volatilidad es información pura que el mercado subestima.

```python
for team_col, goal_col, prefix in [("home_team","fthg","home"), ("away_team","ftag","away")]:
    matches[f"{prefix}_form_mean5"] = matches.groupby(team_col)[goal_col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    matches[f"{prefix}_form_std5"]  = matches.groupby(team_col)[goal_col].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std())
    matches[f"{prefix}_form_cv5"]   = (
        matches[f"{prefix}_form_std5"] / (matches[f"{prefix}_form_mean5"] + 0.01))
    # CV alto → muy volátil (difícil de predecir) | CV bajo → consistente
```

---

### 🏡 M5 — "Personalised Home Advantage" *(no todos los estadios son iguales)*
> **Crushed fact:** La ventaja local promedio en la Premier League es +0.26 goles. Pero **Liverpool en Anfield** tiene +0.65 goles de ventaja local, mientras equipos como el Brighton tienen +0.10. Modelar esto por equipo es más preciso que usar un valor global.

```python
# Ventaja local histórica por equipo (diferencia de goles home vs away)
home_adv = matches.groupby("home_team").agg(
    home_goals_avg=("fthg", "mean"),
    home_conceded_avg=("ftag", "mean")
).reset_index()

away_adv = matches.groupby("away_team").agg(
    away_goals_avg=("ftag", "mean"),
    away_conceded_avg=("fthg", "mean")
).reset_index().rename(columns={"away_team": "home_team"})

home_adv = home_adv.merge(away_adv, on="home_team")
home_adv["personalized_home_advantage"] = (
    home_adv["home_goals_avg"] - home_adv["away_goals_avg"])  # ¿Cuánto más marca en casa?

matches = matches.merge(home_adv[["home_team","personalized_home_advantage"]], on="home_team")
```

---

### 🎰 M6 — "Bookmaker Disagreement Index" *(dónde el mercado duda)*
> **Idea de hedge funds aplicada al fútbol:** En los mercados financieros, el spread bid-ask mide cuánto desacuerdo hay. En apuestas, si B365h=2.5 y BWhh=2.9, el mercado está dividido — alta incertidumbre. La **divergencia entre bookmakers es una feature de información**. Cuando hay mucho acuerdo, el modelo debe confiar más en las cuotas. Cuando hay poco acuerdo, hay oportunidad.

```python
# Divergencia entre bookmakers para el resultado local
# Tenemos B365, BW, VContes, PSW en matches.csv
matches["bookmaker_spread_home"] = matches[["b365h", "bwh", "vch"]].std(axis=1)
matches["bookmaker_spread_draw"] = matches[["b365d", "bwd", "vcd"]].std(axis=1)
matches["bookmaker_spread_away"] = matches[["b365a", "bwa", "vca"]].std(axis=1)
# ALTA std → mercado incierto → el modelo tiene más margen para divergir
# BAJA std → mercado convencido → cuotas implícitas son muy fiables
```

---

### 🧬 M7 — "Tactical Matchup Matrix" *(quién le gana a quién tácticamente)*
> **Idea:** Cruza el estilo de juego de los dos equipos. Un equipo con PPDA bajo (pressing intenso) vs uno con **alta descentralización** de pases crea un matchup específico. ¿El pressing rompe las redes descentralizadas, o estas fluyen alrededor? Los datos lo dirán.

```python
# Interacción entre PPDA del local y descentralización del visitante
matches["tactical_clash"] = (
    matches["home_ppda_roll5"] * matches["away_decentralization_roll5"])
# Si home_ppda es BAJO (pressing intenso) y away_decentralization es Alto → "táctico" 
# Si ambos son medios → partido abierto
# Crear bins para categorizarlo
matches["matchup_type"] = pd.cut(
    matches["tactical_clash"],
    bins=3, labels=["defensive_duel", "open_game", "possession_battle"])
```

---

### 💰 M8 — "FPL Attacking Threat Score" *(el estado del ataque con datos de jugadores)*
> **Idea disruptiva:** Usa `players.csv` + `player_history.csv` para calcular el **potencial ofensivo disponible** del equipo en ese partido. Un equipo sin sus 3 mejores atacantes (injured/status≠Available) tiene menos xG esperado — y este dato **es público pero Bet365 tarda en procesarlo comprando lineups.**

```python
# Calcular "threat score" del once disponible por jornada
import json

# Filtrar jugadores disponibles en esa jornada (status = Available)
available = players[players["status"] == "a"]  # 'a' = available en FPL

# Sumar el threat (ataque) de los top-5 jugadores disponibles por equipo
team_threat = available.groupby("team").agg(
    attacking_threat=("threat", lambda x: x.nlargest(5).sum()),
    top_form=("form", lambda x: x.nlargest(5).mean()),
    avg_xg_player=("expected_goals", lambda x: x.nlargest(5).mean())
).reset_index()

matches = matches.merge(team_threat.rename(columns={"team":"home_team", 
                                                      "attacking_threat":"home_attacking_threat"}),
                        on="home_team", how="left")
```

---

### 📈 M9 — "Poisson-Expected Points" *(convertir xG en distribución de resultado)*
> **La feature más sofisticada y más usada en la industria.** En vez de usar xG directamente, simulamos 10,000 partidos con distribución Poisson y calculamos la probabilidad real de H/D/A. Esto es la base del modelo Dixon-Coles.

```python
from scipy.stats import poisson

def poisson_match_probs(xg_home, xg_away, max_goals=8):
    """Calcula P(H), P(D), P(A) dado xG de cada equipo usando Poisson"""
    prob_home, prob_draw, prob_away = 0, 0, 0
    for gh in range(max_goals):
        for ga in range(max_goals):
            p = poisson.pmf(gh, xg_home) * poisson.pmf(ga, xg_away)
            if gh > ga:   prob_home += p
            elif gh == ga: prob_draw += p
            else:          prob_away += p
    return prob_home, prob_draw, prob_away

# Aplicar con el xG rolling de últimas 5 jornadas
matches[["poisson_prob_h","poisson_prob_d","poisson_prob_a"]] = matches.apply(
    lambda r: pd.Series(poisson_match_probs(r["home_xg_avg5"], r["away_xg_avg5"])), axis=1)

# Diferencia entre nuestra Poisson-prob y la implied de Bet365 = "EDGE"
matches["edge_home"] = matches["poisson_prob_h"] - matches["implied_prob_h"]
# Edge positivo: nuestro modelo cree que el local tiene MÁS probabilidad de la que B365 paga
```

---

### 🎭 M10 — "Psychological Momentum" *(goles en injury time y su efecto)*
> **Idea behavioral economics:** Un gol en el minuto 90+3 que empata un partido tiene un efecto psicológico desproporcionado en el siguiente partido de AMBOS equipos. El equipo que recibe ese gol llega "roto". El que lo marca llega "invicto". Esto **no está en ninguna estadística de cuotas** porque requiere leer los datos evento por evento.

```python
# ¿El último gol del partido fue en injury time?
last_goals = events[events["is_goal"]==True].groupby("match_id").apply(
    lambda g: g.loc[g["minute"].idxmax()])

late_drama = last_goals[last_goals["minute"] >= 88].copy()
late_drama["was_equalizer"]  = late_drama.apply(lambda r:
    abs(r["home_score"] - r["away_score"]) <= 1, axis=1)
late_drama["psychological_shock"] = (
    late_drama["minute"] >= 90) & late_drama["was_equalizer"]

# Unir al siguiente partido de cada equipo
# Si fue un gol de impacto psicológico → efecto en el rendimiento siguiente partido
```

---

### 🔮 M11 — "Dixon-Coles Strength Parameters" *(el modelo profesional)*
> **Inspiración:** El modelo Dixon-Coles (1997) estima parámetros de ataque `α` y defensa `β` para cada equipo. Un equipo con α_home=1.5 y defendiendo contra un equipo con β_away=0.7 da `λ_goals = 1.5 × 0.7 × γ_home`. Es el estándar de la industria de apuestas. Podemos estimarlo con MLE (Maximum Likelihood Estimation).

```python
from scipy.optimize import minimize

def dixon_coles_log_likelihood(params, home_teams, away_teams, home_goals, away_goals, teams):
    n = len(teams)
    attack  = dict(zip(teams, params[:n]))
    defense = dict(zip(teams, params[n:2*n]))
    gamma   = params[2*n]  # home advantage
    
    ll = 0
    for h, a, gh, ga in zip(home_teams, away_teams, home_goals, away_goals):
        lambda_h = np.exp(attack[h] + defense[a] + gamma)
        lambda_a = np.exp(attack[a] + defense[h])
        ll += poisson.logpmf(gh, lambda_h) + poisson.logpmf(ga, lambda_a)
    return -ll  # minimizar negativo = maximizar

# Los parámetros estimados son features directas para el Modelo 2
matches["home_attack_strength"]  = matches["home_team"].map(attack_params)
matches["away_defense_weakness"] = matches["away_team"].map(defense_params)
matches["expected_goals_poisson_home"] = np.exp(
    matches["home_attack_strength"] + matches["away_defense_weakness"] + gamma)
```

---

### 🌡️ M12 — "Temperature of the Season" *(cuándo más importa un partido)*
> **Insight:** Los partidos en las últimas 10 jornadas de la temporada tienen dinámicas completamente diferentes: equipos en relegación pelean más, equipos en Champions League rotation differently. El **matchday relativo a la temporada** es un feature de contexto poderoso.

```python
# Semana de la temporada (1 = agosto, 38 = mayo)
matches["matchday"] = matches.groupby("season").cumcount() + 1

# ¿Partido en la fase crítica de la temporada?
matches["is_crunch_time"]     = (matches["matchday"] >= 30).astype(int)  # Top 6 y relegación
matches["is_early_season"]    = (matches["matchday"] <= 8).astype(int)   # Volatilidad alta
matches["season_temperature"] = matches["matchday"] / 38  # 0→1 a lo largo de la temporada

# Estadios de la temporada donde los equipos se comportan diferente
# → feature continua que el modelo puede aprender no linealmente
```

---

### 🎯 Feature Set Completo Modelo 2

```python
FEATURES_MATCH_V2 = [
    # ── Probabilidades implícitas (baseline fuerte) ──
    "implied_prob_h", "implied_prob_d", "implied_prob_a",
    "bookmaker_spread_home",           # M6: incertidumbre del mercado
    
    # ── xG y calidad de ataque (Modelo 1 → Modelo 2) ──
    "home_xg_avg5", "away_xg_avg5",
    "home_mean_xg_per_shot_roll5",     # M2: calidad del tiro
    "home_xg_set_piece_roll5",         # M1: dependencia de set pieces
    "home_xg_counter_roll5",           # M1: xG de contras
    "poisson_prob_h", "poisson_prob_d",# M9: distribución Poisson propia
    "edge_home",                       # M9: ventaja vs Bet365
    
    # ── Forma y momentum ──
    "home_momentum", "away_momentum",  # G: MACD del fútbol
    "home_form_cv5",                   # M4: volatilidad de forma
    "home_xg_debt_5",                  # D: deuda de xG (Tippett)
    
    # ── Táctica y pressing ──
    "home_ppda_roll5", "away_ppda_roll5",           # E: pressing
    "home_decentralization_roll5",                  # F: Sumpter
    "home_altitude_roll5", "away_altitude_roll5",   # I: centro de gravedad
    "tactical_clash",                               # M7: matchup táctico
    
    # ── Contexto partido ──
    "home_rest_days", "away_rest_days",             # M3: fatiga
    "rest_advantage",                               # M3: diferencial descanso
    "personalized_home_advantage",                  # M5: ventaja local ajustada
    "referee_home_bias",                            # H: sesgo árbitro
    
    # ── Calidad de plantilla (FPL) ──
    "home_attacking_threat", "away_attacking_threat",# M8: FPL threat
    
    # ── Parámetros Dixon-Coles ──
    "home_attack_strength", "away_defense_weakness", # M11: modelo profesional
    "expected_goals_poisson_home",
    
    # ── Contexto de temporada ──
    "season_temperature",                            # M12: fase del año
    "is_crunch_time",
    
    # ── Clutch y psicología ──
    "home_clutch_ratio_roll5",                       # J: rendimiento final
    "home_psychological_shock",                      # M10: gol injury time anterior
]
```

---

## 🔗 Fase 1D — Features Faltantes del EDA Maestro (incorporación)

> Estas features aparecen en `EDA.md` con ⭐⭐⭐⭐⭐ de importancia y **no estaban en la hoja de ruta**. Se incorporan aquí para que no se pierdan.

---

### 🏅 M13 — "Elo Rating" *(el predictor #1 en la literatura académica)*
> **EDA.md §3**: `elo_diff` tiene **⭐⭐⭐⭐⭐** — el predictor más fuerte no derivado de cuotas. Refleja la fortaleza histórica relativa de los equipos, actualizado partido a partido.

```python
def update_elo(home_elo, away_elo, result, k=32):
    """
    result: 1 = victoria local, 0.5 = empate, 0 = victoria visitante
    k = 32 (FIFA standard); usar k=20 para Premier League (menor varianza)
    """
    expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    expected_away = 1 - expected_home
    new_home = home_elo + k * (result - expected_home)
    new_away = away_elo + k * ((1 - result) - expected_away)
    return new_home, new_away

# Inicializar todos los equipos en 1500, actualizar jornada a jornada
elo_ratings = {team: 1500 for team in matches["home_team"].unique()}

for _, row in matches.sort_values("date").iterrows():
    h, a = row["home_team"], row["away_team"]
    result = 1 if row["ftr"] == "H" else (0.5 if row["ftr"] == "D" else 0)
    elo_ratings[h], elo_ratings[a] = update_elo(elo_ratings[h], elo_ratings[a], result)
    matches.loc[_, "elo_home"] = elo_ratings[h]  # guardar ANTES de actualizar
    matches.loc[_, "elo_away"] = elo_ratings[a]

matches["elo_diff"] = matches["elo_home"] - matches["elo_away"]
# elo_diff / 400 normalizado es directamente convertible a probabilidad de victoria
matches["elo_win_prob"] = 1 / (1 + 10 ** (-matches["elo_diff"] / 400))
```

---

### 🎮 M14 — "Game State" *(el contexto táctico más importante a nivel de tiro)*
> **EDA.md §3**: `game_state` tiene **⭐⭐⭐** — cambia radicalmente el contexto del tiro. Un equipo perdiendo 0-2 al minuto 80 abre espacios; un equipo ganando 1-0 se cierra. El xG del mismo tiro geométrico es diferente en cada game state.

```python
# Calcular el marcador EN EL MOMENTO de cada tiro (pre-tiro)
events_sorted = events.sort_values(["match_id", "minute", "second"])
events_sorted["cumulative_home_goals"] = events_sorted.groupby("match_id")["is_goal"].transform(
    lambda x: x.cumsum().shift(1, fill_value=0))

shots["game_state"] = shots.apply(lambda r: (
    "winning"  if (r["team_is_home"] and r["cumulative_home_goals"] > r["cumulative_away_goals"]) or
                  (not r["team_is_home"] and r["cumulative_away_goals"] > r["cumulative_home_goals"])
    else "drawing" if r["cumulative_home_goals"] == r["cumulative_away_goals"]
    else "losing"
), axis=1)
# One-hot: game_state_winning, game_state_drawing, game_state_losing
shots = pd.get_dummies(shots, columns=["game_state"], prefix="gs")
```

---

### 💎 M15 — "Shot Quality Index" *(feature compuesta ponderada)*
> **EDA.md §3**: `shot_quality_idx` con **⭐⭐⭐** — combina múltiples señales en un único índice. Más interpretable que variables separadas y reduce dimensionalidad.

```python
# Shot Quality Index = ponderación de los mejores predictores de xG
# Pesos basados en su contribución media al xG real
shots["shot_quality_index"] = (
    shots["is_big_chance"]  * 0.38 +   # xG implícito BigChance
    shots["is_in_area"]     * 0.18 +   # Dentro del área
    shots["is_counter"]     * 0.12 +   # Contra (defensa desorganizada)
    shots["is_central"]     * 0.10 +   # Carril central
    (1 - shots["distance_to_goal"] / shots["distance_to_goal"].max()) * 0.22
)
# Útil como feature única Y como variable de resumen para el Modelo 2
```

---

### 📋 M16 — "Head-to-Head Win Rate" *(la maldición de ciertos duelos)*
> **EDA.md §3**: `head_to_head_h_win_rate` con **⭐⭐⭐** — algunos equipos históricamente superan a rivales específicos independientemente de la forma reciente. Es información que el mercado no siempre pondera correctamente.

```python
# Historial directo entre equipos (últimas N temporadas)
h2h = matches.groupby(["home_team", "away_team"]).agg(
    h2h_total=("ftr", "count"),
    h2h_home_wins=("ftr", lambda x: (x == "H").sum()),
    h2h_draws=("ftr", lambda x: (x == "D").sum()),
    h2h_away_wins=("ftr", lambda x: (x == "A").sum()),
    h2h_home_goals_avg=("fthg", "mean"),
    h2h_away_goals_avg=("ftag", "mean")
).reset_index()

h2h["h2h_home_win_rate"]  = h2h["h2h_home_wins"] / h2h["h2h_total"]
h2h["h2h_total_goals_avg"]= h2h["h2h_home_goals_avg"] + h2h["h2h_away_goals_avg"]
# Total de goles históricos H2H predice Over/Under del partido
```

---

### 📈 M17 — "xG Overperformance per Player" *(corrección de finishing skill)*
> **EDA.md §2.4**: El xG subestima a los grandes rematadores (Haaland, Isak). Incluir la diferencia histórica `goles_reales - xG` por jugador corrige este sesgo sistemático.

```python
# Por jugador: ¿cuánto sobre/subanota vs su xG acumulado?
player_xg_stats = player_history.groupby("player_id").agg(
    total_goals=("goals_scored", "sum"),
    total_xg=("expected_goals", "sum"),
    total_assists=("assists", "sum"),
    total_xa=("expected_assists", "sum")
).reset_index()

player_xg_stats["finishing_skill"] = (
    player_xg_stats["total_goals"] - player_xg_stats["total_xg"])
player_xg_stats["creation_skill"]  = (
    player_xg_stats["total_assists"] - player_xg_stats["total_xa"])
# Haaland: finishing_skill ≈ +8 (convierte mucho más de lo esperado)
# Jugador ineficiente: finishing_skill ≈ -3
```

---

### 🏃 M18 — "Exponential Decay Form" *(más peso a los juegos recientes)*
> **EDA.md §3 Feature Engineering Avanzado**: La forma simple (rolling mean) da igual peso a todos los partidos. El decay exponencial pondera más los resultados recientes — más realista.

```python
import numpy as np

def exp_decay_form(results, lambda_decay=0.3):
    """
    Promedio ponderado de resultados con decay exponencial.
    Resultado: 1=victoria, 0.5=empate, 0=derrota
    lambda_decay: mayor λ = más peso a partidos recientes
    """
    n = len(results)
    weights = np.exp(-lambda_decay * np.arange(n - 1, -1, -1))
    weights /= weights.sum()
    return np.dot(weights, results)

# Aplicar por equipo: últimos 10 partidos con decay
matches["home_exp_form"] = matches.groupby("home_team").apply(
    lambda g: g["fthg"].rolling(10).apply(
        lambda x: exp_decay_form([1 if v > 0 else 0.5 if v == 0 else 0 for v in x])
    )
).reset_index(level=0, drop=True)
```

---

### 💰 M19 — "Squad Value Ratio" *(proxy de calidad de plantilla)*
> **EDA.md §3**: `squad_value_ratio` con **⭐⭐⭐** — el precio medio en FPL es proxy del valor de mercado. El cociente entre los precios medios de los onces titulares captura la diferencia de calidad de plantillas.

```python
# Precio FPL como proxy de valor de mercado
team_values = players.groupby("team").agg(
    squad_value_mean=("now_cost", "mean"),
    top11_value=("now_cost", lambda x: x.nlargest(11).mean()),
    squad_depth_value=("now_cost", lambda x: x.nlargest(15).mean())
).reset_index()

matches = matches.merge(
    team_values.rename(columns={"team": "home_team", "top11_value": "home_squad_value"}),
    on="home_team", how="left"
)
matches = matches.merge(
    team_values.rename(columns={"team": "away_team", "top11_value": "away_squad_value"}),
    on="away_team", how="left"
)
matches["squad_value_ratio"] = matches["home_squad_value"] / matches["away_squad_value"]
# > 1 → local tiene plantilla más valiosa en papel | < 1 → visitante más valioso
```

---

## ⚠️ Fase 1E — Riesgos y Mitigaciones (EDA.md §4)

> **Crítico:** Estos riesgos pueden invalidar los modelos si no se atienden desde el inicio.

| Riesgo | Impacto | Señal de alerta | Mitigación |
|---|---|---|---|
| **Data Leakage con cuotas post-partido** | 🔴 MUY ALTO | AUC irrealmente alto en train (>0.99) | Usar **SOLO cuotas de apertura**, NUNCA de cierre |
| **API degradada: player_history reducido** | 🟠 ALTO para M2 | Solo ~1,500 filas vs ~15,000 esperadas | Usar features de players.csv como proxy; re-descargar cuando API se recupere |
| **Desbalance de clases en `ftr`** | 🟡 MEDIO | Modelo siempre predice "H" | Usar `class_weight='balanced'`, SMOTE, o calibración |
| **Sobreajuste en rolling features** | 🟡 MEDIO | Gap grande entre CV y test | **TimeSeriesSplit**, NUNCA KFold aleatorio |
| **Pocos partidos (291) para DL** | 🔴 ALTO si usas redes neuronales | Overfitting en validación | Priorizar XGBoost/RF; DL solo con regularización fuerte |
| **Multicolinealidad entre cuotas** | 🟡 MEDIO | VIF alto entre implied_h/d/a | Solo usar 2 de las 3 (la tercera es implícita: suma=1) |
| **xG de partidos futuros como feature** | 🔴 MUY ALTO | Usar xG del partido a predecir | Solo usar xG **rolling histórico** (shift(1) antes de calcular medias) |

```python
# ✅ CORRECTO — xG histórico con shift para evitar leakage
matches["home_xg_avg5"] = matches.groupby("home_team")["home_xg"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=2).mean()
)

# ❌ INCORRECTO — incluye el partido actual
matches["home_xg_avg5"] = matches.groupby("home_team")["home_xg"].transform(
    lambda x: x.rolling(5, min_periods=2).mean()  # ← LEAKAGE
)
```

---

## 🔀 Fase 1F — Transformaciones y Feature Selection

### Tabla de Transformaciones (EDA.md §6.1)

| Transformación | Feature | Razón | Código |
|---|---|---|---|
| **Log** | `distance_to_goal` | Distribución sesgada a la derecha | `np.log1p(distance)` |
| **StandardScaler** | Todas las features numéricas | Regresión Logística es sensible a escala | `StandardScaler().fit_transform(X)` |
| **MinMax** | `distance`, `angle` | Si hay outliers severos | `MinMaxScaler()` |
| **One-Hot Encoding** | `event_type`, `position`, `zone` | Categóricas de baja cardinalidad | `pd.get_dummies()` |
| **Binary** | Todos los qualifiers | Ya binarios por diseño | Sin transformación |
| **Binning** | `minute` | Efectos no lineales del tiempo | `pd.cut(minute, bins=[0,15,30,45,60,75,90])` |

### Feature Selection — Métodos (EDA.md §6.3)

```python
# 1. Eliminar multicolinealidad primero
correlation_matrix = X_train.corr()
high_corr = (correlation_matrix.abs() > 0.90).any()  # eliminar una de cada par

# 2. RFE (Recursive Feature Elimination)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=500)
rfe = RFE(model, n_features_to_select=10)
rfe.fit(X_train, y_train)
selected = X_train.columns[rfe.support_].tolist()

# 3. L1 Regularization (Lasso elimina features irrelevantes automáticamente)
model_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
model_lasso.fit(X_train, y_train)
active_features = X_train.columns[model_lasso.coef_[0] != 0].tolist()

# 4. SHAP para interpretabilidad post-hoc (mejor que feature_importance)
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # 🔥 el gráfico más impresionante del proyecto
```

### Manejo del Desbalanceo de Clases (EDA.md §6.4)

**Modelo 1 (xG):** tasa de goles ~11% → fuertemente desbalanceado.

```python
# Opción A: class_weight automático (recomendada como baseline)
model = LogisticRegression(class_weight='balanced')

# Opción B: SMOTE oversampling (solo sobre training set, NUNCA sobre test)
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Opción C: Threshold tuning (sin alterar el dataset)
probs = model.predict_proba(X_test)[:, 1]
threshold = 0.3   # Bajamos de 0.5 a 0.3 para detectar más goles reales
preds = (probs >= threshold).astype(int)
```

**Modelo 2 (Match Predictor):** H=42%, D=26%, A=32% — desbalanceo leve.

```python
# Para multiclase: solo class_weight en XGBoost/RF, no SMOTE
rf = RandomForestClassifier(class_weight="balanced_subsample")
# Evaluar F1 Macro (no accuracy) para detectar sesgo hacia la clase mayoritaria
from sklearn.metrics import f1_score
f1_macro = f1_score(y_test, y_pred, average="macro")
```

---

## 🔗 Dataset Linking — Cómo unir las 4 fuentes (EDA.md §7)

```
events.csv ─────────── match_id ──────→ matches.csv (partidos)
                                              ↑
player_history.csv ─── player_id ──→ players.csv
player_history.csv ─── gameweek  ──→ (fecha implícita de jornada FPL)
events.csv ──────────── player_id ──→ players.csv  (opcional — enriquece tiros con stats del jugador)
```

**Join principal para el Feature Engineering:**
```python
# 1. Agregar xG por partido desde events
xg_per_match = events[events["is_shot"]==True].groupby(["match_id","team_name"]).agg(
    xg_total=("xg_predicted", "sum"),
    big_chances=("is_big_chance", "sum"),
    shots=("is_shot", "count")
).reset_index()

# 2. Unir a matches
matches_enriched = matches.merge(
    xg_per_match.rename(columns={"team_name":"home_team","xg_total":"home_xg"}),
    on=["match_id","home_team"], how="left"
)

# 3. Unir estado de plantilla desde players
squad_status = players.groupby("team").agg(
    injured_count=("status", lambda x: (x != "a").sum()),
    attacking_threat=("threat", lambda x: x.nlargest(5).sum())
).reset_index()

matches_enriched = matches_enriched.merge(
    squad_status.rename(columns={"team":"home_team",
                                  "injured_count":"home_injured",
                                  "attacking_threat":"home_threat"}),
    on="home_team", how="left"
)
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

### 📋 Tabla Completa de Variables — Modelo 1 (xG)

| Variable | Fuente | Tipo | Descripción | Impacto Esperado |
|---|---|---|---|---|
| `distance_to_goal` | events | Estándar ✅ | Distancia euclidiana al centro de portería | 🔥🔥🔥 Más alta correlación con gol |
| `angle_to_goal` | events | Estándar ✅ | Ángulo de apertura hacia portería | 🔥🔥🔥 Feature Taller2 obligatoria |
| `dist_squared` | events | Estándar ✅ | Cuadrado de la distancia — relación no lineal | 🔥🔥 Captura asimetría |
| `dist_angle` | events | Estándar ✅ | Interacción distancia × ángulo | 🔥🔥 Interacción geométrica |
| `is_in_area` | events | Estándar ✅ | ¿Está dentro del área grande? (x>83) | 🔥🔥🔥 Zona de máximo peligro |
| `is_central` | events | Estándar ✅ | ¿Está en carril central? (33<y<67) | 🔥🔥 Frente al arco |
| `is_big_chance` | qualifiers | Estándar ✅ | Oportunidad clara de gol (BigChance) | 🔥🔥🔥 xG implícito ~38% |
| `is_penalty` | qualifiers | Estándar ✅ | Penalti — categoría especial | 🔥🔥🔥 xG fijo ~76% |
| `is_header` | qualifiers | Estándar ✅ | Remate de cabeza | 🔥🔥 -30% efectividad vs pie |
| `is_right_foot` | qualifiers | Estándar ✅ | Pie dominante derecho | 🔥 Referencia base |
| `is_left_foot` | qualifiers | Estándar ✅ | Pie no dominante izquierdo | 🔥 Menor precisión media |
| `is_counter` | qualifiers | Estándar ✅ | Contraataque (FastBreak) | 🔥🔥 Defensa desorganizada |
| `from_corner` | qualifiers | Estándar ✅ | Balón viene de saque de esquina | 🔥 Situación de balón parado |
| `is_volley` | qualifiers | Estándar ✅ | Volea — alta dificultad técnica | 🔥 Menor conversión |
| `first_touch` | qualifiers | Estándar ✅ | Disparo a primer toque | 🔥🔥 Mayor varianza |
| `is_set_piece` | qualifiers | Estándar ✅ | Balón parado genérico | 🔥 Defensa en bloque |
| `minute` | events | Estándar ✅ | Minuto del partido | 🔥 Efecto tiempo/presión |
| `defensive_pressure` 🧪 | events | **Original** | Nº acciones rivales en radio del tirador | 🔥🔥🔥 Simula freeze_frame StatsBomb |
| `buildup_passes` 🧪 | events | **Original** | Pases exitosos del equipo en el min previo | 🔥🔥 Calidad de la jugada previa |
| `buildup_decentralized` 🧪 | events | **Original** | ¿Participaron >3 jugadores distintos? | 🔥🔥 Sumpter-validated |
| `porteria_zone` 🧪 | events | **Original** | Zona de portería donde fue el tiro (9 zonas) | 🔥🔥 Cobertura del portero |

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

### 📋 Tabla Completa de Variables — Modelo 2 (Match Predictor)

| Variable | Fuente | Tipo | Descripción | Impacto Esperado |
|---|---|---|---|---|
| **── CUOTAS (Baseline oro) ──** | | | | |
| `implied_prob_h/d/a` | matches | Estándar ✅ | Probabilidad implícita de B365 normalizada | 🔥🔥🔥 Mejor predictor individual |
| `bookmaker_spread_home` 🧪 | matches | **Original** | Std entre B365, BW, VC para H — desacuerdo del mercado | 🔥🔥🔥 Señal de incertidumbre |
| **── xG (Modelo 1 → 2) ──** | | | | |
| `home/away_xg_avg5` | events→M1 | Estándar ✅ | xG rolling de las últimas 5 jornadas | 🔥🔥🔥 Base de predicción de goles |
| `home_mean_xg_per_shot_roll5` 🧪 | events→M1 | **Original** | Calidad del tiro, no cantidad — estilo de juego real | 🔥🔥🔥 Shot Quality Premium |
| `home_xg_set_piece_roll5` 🧪 | events→M1 | **Original** | xG proveniente de balón parado | 🔥🔥 Matchup táctico |
| `home_xg_counter_roll5` 🧪 | events→M1 | **Original** | xG proveniente de contragolpe | 🔥🔥 Estilo de ataque |
| `poisson_prob_h/d/a` 🧪 | xG→Scipy | **Original** | P(H/D/A) calculada con distribución Poisson propia | 🔥🔥🔥 Modelo profesional interno |
| `edge_home` 🧪 | xG vs B365 | **Original** | Diferencia entre nuestra P(H) y la implied de B365 | 🔥🔥🔥 Edge real sobre la casa |
| `home_xg_debt_5` 🧪 | events→M1 | **Original** | Goles debidos: xG acumulado − goles reales últimas 5 jornadas | 🔥🔥 Ineficiencia de mercado Tippett |
| **── FORMA Y MOMENTUM ──** | | | | |
| `home/away_goals_avg5` | matches | Estándar ✅ | Media de goles anotados últimas 5 jornadas | 🔥🔥 Forma ofensiva reciente |
| `home/away_goals_conceded_avg5` | matches | Estándar ✅ | Media de goles recibidos últimas 5 jornadas | 🔥🔥 Forma defensiva reciente |
| `home/away_momentum` 🧪 | matches | **Original** | MACD del fútbol: form_3 − form_10 | 🔥🔥 Equipo acelerando vs frenando |
| `home_form_cv5` 🧪 | matches | **Original** | Coeficiente de variación de goles últimas 5 jornadas | 🔥🔥 Equipo impredecible vs consistente |
| **── TÁCTICA Y PRESSING ──** | | | | |
| `home/away_ppda_roll5` 🧪 | events | **Original** | PPDA proxy: pressing intensity desde events.csv | 🔥🔥🔥 Liverpool <7, defensivos >15 |
| `home_decentralization_roll5` 🧪 | events | **Original** | Jugadores únicos en pases exitosos — Sumpter | 🔥🔥 Red de pase descentralizada |
| `home/away_altitude_roll5` 🧪 | events | **Original** | Posición X media del equipo — "dónde vive el balón" | 🔥🔥 Estilo defensivo vs ofensivo |
| `tactical_clash` 🧪 | events | **Original** | PPDA_local × Descentralización_visitante | 🔥🔥 Matchup táctico directo |
| **── CONTEXTO DEL PARTIDO ──** | | | | |
| `home/away_rest_days` 🧪 | matches | **Original** | Días desde el último partido de cada equipo | 🔥🔥 Fatiga física documentada |
| `rest_advantage` 🧪 | matches | **Original** | Diferencia de días de descanso (home − away) | 🔥🔥 Ventaja fisiológica |
| `personalized_home_advantage` 🧪 | matches | **Original** | Ventaja local histórica específica por equipo | 🔥🔥🔥 Anfield ≠ cualquier estadio |
| `referee_home_bias` 🧪 | matches | **Original** | Sesgo histórico del árbitro hacia el local | 🔥 Corrección estadística del árbitro |
| `season_temperature` 🧪 | matches | **Original** | Jornada/38 — fase de la temporada como feature continua | 🔥🔥 Dinámica de temporada |
| `is_crunch_time` 🧪 | matches | **Original** | ¿Es jornada 30+? Zona de relegación y Champions | 🔥🔥 Equipos juegan diferente |
| **── PLANTILLA (FPL) ──** | | | | |
| `home/away_attacking_threat` 🧪 | players | **Original** | Suma `threat` FPL de los top-5 disponibles | 🔥🔥🔥 Estado real del ataque con lesiones |
| **── MODELO PROFESIONAL ──** | | | | |
| `home_attack_strength` 🧪 | matches→MLE | **Original** | Parámetro de ataque Dixon-Coles estimado con MLE | 🔥🔥🔥 Estándar de la industria |
| `away_defense_weakness` 🧪 | matches→MLE | **Original** | Parámetro de defensa Dixon-Coles | 🔥🔥🔥 Fuerza defensiva estimada |
| **── PSICOLOGÍA ──** | | | | |
| `home_clutch_ratio_roll5` 🧪 | events | **Original** | Goles minuto 75+ / Goles minuto <75 | 🔥🔥 Rendimiento bajo presión |
| `home_psychological_shock` 🧪 | events | **Original** | Gol dramático en injury time en el partido anterior | 🔥 Efecto behavioral economics |

> 🧪 = Feature original no estándar en proyectos universitarios

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
