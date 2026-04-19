"""
eda_events.py — EDA Completo de events.csv (con soporte para qualifiers)
Detecta automáticamente si el CSV tiene la columna 'qualifiers' y extrae features adicionales.
Genera gráficas en: Research/EDA/events/graficas/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[3]
DATA_FILE  = ROOT / "data" / "raw" / "events.csv"
OUT_PLOTS  = ROOT / "Research" / "EDA" / "events" / "graficas"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi": 150, "figure.max_open_warning": 0})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

def draw_pitch(ax, color="#1b4332", line_color="white"):
    ax.set_facecolor(color); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    kw = dict(fill=False, edgecolor=line_color, linewidth=1.5)
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, **kw))
    ax.plot([50, 50], [0, 100], color=line_color, linewidth=1.5)
    ax.add_patch(patches.Circle((50, 50), 9.15, **kw))
    ax.add_patch(patches.Rectangle((83, 21.1), 17, 57.8, **kw))
    ax.add_patch(patches.Rectangle((94.5, 36.8), 5.5, 26.4, **kw))
    ax.add_patch(patches.Rectangle((0, 21.1), 17, 57.8, **kw))
    ax.plot(88.5, 50, 'o', color=line_color, markersize=3)
    ax.axis("off")

# ─────────────────────────────────────────────────────────────────────────────
print("Cargando events.csv...")
df = pd.read_csv(DATA_FILE)
HAS_QUALIFIERS = "qualifiers" in df.columns
print(f"Shape: {df.shape} | qualifiers disponibles: {HAS_QUALIFIERS}")

# Extraer features de qualifiers si están disponibles
if HAS_QUALIFIERS:
    print("Parseando qualifiers...")
    q = df["qualifiers"].fillna("[]").astype(str)
    df["is_big_chance"]  = q.str.contains("BigChance",  na=False).astype(int)
    df["is_header"]      = q.str.contains('"Head"',     na=False).astype(int)
    df["is_right_foot"]  = q.str.contains("RightFoot",  na=False).astype(int)
    df["is_left_foot"]   = q.str.contains("LeftFoot",   na=False).astype(int)
    df["is_counter"]     = q.str.contains("FastBreak",  na=False).astype(int)
    df["from_corner"]    = q.str.contains("FromCorner", na=False).astype(int)
    df["is_penalty"]     = q.str.contains('"Penalty"',  na=False).astype(int)
    df["is_volley"]      = q.str.contains("Volley",     na=False).astype(int)
    df["first_touch"]    = q.str.contains("FirstTouch", na=False).astype(int)
    df["is_set_piece"]   = q.str.contains("SetPiece",   na=False).astype(int)

# ─── Features geométricas (Taller2 ML1) ─────────────────────────────────────
if "x" in df.columns and "y" in df.columns:
    df["distance_to_goal"] = np.sqrt((100 - df["x"])**2 + (50 - df["y"])**2)
    df["angle_to_goal"]    = np.abs(np.arctan2(50 - df["y"], 100 - df["x"]))

shots = df[df["is_shot"] == True].copy() if "is_shot" in df.columns else pd.DataFrame()

# ─── 1. NULOS ────────────────────────────────────────────────────────────────
print("[1] Nulos...")
null_pct  = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_data = null_pct[null_pct > 0]
if len(null_data) > 0:
    fig, ax = plt.subplots(figsize=(10, max(3, len(null_data)*0.45)))
    null_data.plot(kind="barh", ax=ax, color="tomato", edgecolor="black")
    ax.set_xlabel("% Nulos"); ax.set_title("Porcentaje de Nulos — events.csv")
    save("01_nulos")

# ─── 2. TIPOS DE EVENTOS ─────────────────────────────────────────────────────
print("[2] Tipos de eventos...")
event_counts = df["event_type"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=event_counts.values, y=event_counts.index, ax=ax, palette="viridis", edgecolor="black")
ax.set_title("Top 20 Tipos de Evento — events.csv")
save("02_tipos_evento")

# ─── 3. OUTCOME ──────────────────────────────────────────────────────────────
print("[3] Outcome...")
out_counts = df["outcome"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
out_counts.plot(kind="bar", ax=axes[0], edgecolor="black", color=["#4CAF50","#F44336"])
axes[0].set_title("Eventos por Outcome"); axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20)
axes[1].pie(out_counts, labels=out_counts.index, autopct="%1.1f%%", startangle=90, colors=["#4CAF50","#F44336"])
fig.suptitle("Desbalance Outcome — events.csv", fontweight="bold")
save("03_outcome")

# ─── 4. DISTRIBUCIÓN TEMPORAL ────────────────────────────────────────────────
print("[4] Minuto...")
game_ev = df[df["minute"].between(0, 95)]
fig, ax = plt.subplots(figsize=(14, 5))
ax.hist(game_ev["minute"], bins=95, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(45, color="red", linestyle="--", linewidth=1.5, label="HT (45')")
ax.axvline(90, color="orange", linestyle="--", linewidth=1.5, label="FT (90')")
ax.set_xlabel("Minuto"); ax.legend()
ax.set_title("Distribución de Eventos por Minuto")
save("04_eventos_minuto")

# ─── 5. HEATMAP TODOS ────────────────────────────────────────────────────────
print("[5] Heatmap todos...")
valid = df[df["x"].between(0,100) & df["y"].between(0,100)].sample(min(80_000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(12, 8))
draw_pitch(ax)
h2d = ax.hist2d(valid["x"], valid["y"], bins=50, cmap="hot", alpha=0.75)
plt.colorbar(h2d[3], ax=ax, label="Densidad")
ax.set_title("Mapa de Calor — Todos los Eventos", fontweight="bold")
save("05_heatmap_todos")

# ─── 6. HEATMAP TIROS EN CAMPO ───────────────────────────────────────────────
print("[6] Heatmap tiros...")
if len(shots) > 0:
    s_valid = shots[shots["x"].between(0,100) & shots["y"].between(0,100)]
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)
    h2d = ax.hist2d(s_valid["x"], s_valid["y"], bins=40, cmap="YlOrRd", alpha=0.8)
    plt.colorbar(h2d[3], ax=ax, label="Densidad Tiros")
    if "is_goal" in s_valid.columns:
        goals = s_valid[s_valid["is_goal"]==True]
        ax.scatter(goals["x"], goals["y"], c="cyan", s=30, zorder=5, label="Gol",
                   edgecolors="black", linewidths=0.4)
        ax.legend()
    ax.set_title("Mapa de Calor — Tiros (Cyan=Goles)", fontweight="bold")
    save("06_heatmap_tiros")

# ─── 7. EFECTIVIDAD POR ZONA ─────────────────────────────────────────────────
print("[7] Efectividad zona...")
if len(shots) > 0 and "is_goal" in shots.columns:
    shots2 = shots[shots["x"].between(0,100)].copy()
    shots2["zona"] = pd.cut(shots2["x"], bins=[0,33,67,85,100],
                            labels=["Media/Baja","Media","Alta(>67)","Área(>85)"], include_lowest=True)
    eff = shots2.groupby("zona")["is_goal"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=eff.index.astype(str), y=eff.values, ax=ax, palette="RdYlGn", edgecolor="black")
    ax.set_title("% Efectividad de Tiro por Zona"); ax.set_ylabel("% Goles/Tiros")
    for bar, val in zip(ax.patches, eff.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f"{val:.1f}%", ha="center")
    save("07_efectividad_zona")

# ─── 8. FEATURES GEOMÉTRICAS: DISTANCIA Y ÁNGULO ─────────────────────────────
print("[8] Distancia y ángulo (Taller2 features)...")
if len(shots) > 0 and "distance_to_goal" in shots.columns and "is_goal" in shots.columns:
    shots["distance_to_goal"] = np.sqrt((100 - shots["x"])**2 + (50 - shots["y"])**2)
    shots["angle_to_goal"]    = np.abs(np.arctan2(50 - shots["y"], 100 - shots["x"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col, label in zip(axes, ["distance_to_goal","angle_to_goal"],["Distancia al Arco","Ángulo al Arco (rad)"]):
        goles = shots[shots["is_goal"]==True][col].dropna()
        no_goles = shots[shots["is_goal"]==False][col].dropna()
        ax.hist(no_goles, bins=30, color="steelblue", alpha=0.6, density=True, label="No Gol", edgecolor="white")
        ax.hist(goles, bins=30, color="gold", alpha=0.8, density=True, label="⭐ Gol", edgecolor="black")
        ax.set_title(f"{label} — Goles vs No Goles"); ax.legend()
    fig.suptitle("Features Geométricas Clave para xG (Taller2)", fontweight="bold")
    save("08_distancia_angulo_xg")

# ─── 9. QUALIFIERS: EFECTIVIDAD POR TIPO ─────────────────────────────────────
if HAS_QUALIFIERS and len(shots) > 0 and "is_goal" in shots.columns:
    print("[9] Efectividad por qualifier...")
    # Re-extraer en el sub-dataframe de shots
    q_shots = df["qualifiers"].fillna("[]").astype(str)[shots.index] if "qualifiers" in df.columns else None
    if q_shots is not None:
        shots = shots.copy()
        shots["is_big_chance"] = q_shots.str.contains("BigChance",  na=False).astype(int)
        shots["is_header"]     = q_shots.str.contains('"Head"',     na=False).astype(int)
        shots["is_right_foot"] = q_shots.str.contains("RightFoot",  na=False).astype(int)
        shots["is_left_foot"]  = q_shots.str.contains("LeftFoot",   na=False).astype(int)
        shots["is_penalty"]    = q_shots.str.contains('"Penalty"',  na=False).astype(int)
        shots["is_counter"]    = q_shots.str.contains("FastBreak",  na=False).astype(int)
        shots["is_volley"]     = q_shots.str.contains("Volley",     na=False).astype(int)
        shots["first_touch"]   = q_shots.str.contains("FirstTouch", na=False).astype(int)

    efectividad = {
        "Penalti":        shots[shots["is_penalty"]==1]["is_goal"].mean()*100,
        "Big Chance":     shots[shots["is_big_chance"]==1]["is_goal"].mean()*100,
        "Volea":          shots[shots["is_volley"]==1]["is_goal"].mean()*100,
        "Primer Toque":   shots[shots["first_touch"]==1]["is_goal"].mean()*100,
        "Pie Derecho":    shots[shots["is_right_foot"]==1]["is_goal"].mean()*100,
        "Contraataque":   shots[shots["is_counter"]==1]["is_goal"].mean()*100,
        "Pie Izquierdo":  shots[shots["is_left_foot"]==1]["is_goal"].mean()*100,
        "Cabeza":         shots[shots["is_header"]==1]["is_goal"].mean()*100,
    }
    eff_s = pd.Series(efectividad).dropna().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors_eff = ["#e63946" if v < 10 else "#f4a261" if v < 20 else "#2a9d8f" for v in eff_s.values]
    bars = ax.barh(eff_s.index, eff_s.values, color=colors_eff, edgecolor="black")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
    ax.axvline(shots["is_goal"].mean()*100, color="yellow", linestyle="--", linewidth=2,
               label=f"Global: {shots['is_goal'].mean()*100:.1f}%")
    ax.set_xlabel("% Efectividad"); ax.legend()
    ax.set_title("Efectividad por Tipo de Acción — Qualifiers JSON", fontweight="bold")
    save("09_efectividad_qualifiers")

    # ─── 10. MAPA TIROS POR TIPO EN CAMPO ────────────────────────────────────
    print("[10] Mapa tipo de contacto...")
    shots_valid = shots[shots["x"].between(50,100) & shots["y"].between(0,100)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    tipos = [
        (shots_valid[shots_valid["is_right_foot"]==1], "#e63946", "Pie Derecho"),
        (shots_valid[shots_valid["is_left_foot"]==1],  "#457b9d", "Pie Izquierdo"),
        (shots_valid[shots_valid["is_header"]==1],     "#f4a261", "Cabeza"),
    ]
    for ax, (data, color, label) in zip(axes, tipos):
        draw_pitch(ax)
        if len(data) > 0:
            goals = data[data["is_goal"]==True]; no_g = data[data["is_goal"]==False]
            ax.scatter(no_g["x"], no_g["y"], c=color, s=15, alpha=0.35, zorder=3, edgecolors="none")
            ax.scatter(goals["x"], goals["y"], c="gold", s=60, alpha=0.9, zorder=5,
                       edgecolors="white", linewidths=0.8, marker="*")
            ax.set_title(f"{label}\n{len(data):,} tiros · {len(goals):,} goles ⭐",
                         color="white", fontsize=11, fontweight="bold", pad=10)
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("Mapa de Tiros por Tipo de Contacto — ⭐ = Gol", color="white", fontsize=13, fontweight="bold")
    save("10_mapa_tiros_tipo")
else:
    # Sin qualifiers: gráfica de coordenadas y correlación flags
    print("[9] Coordenadas...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df[df["x"].between(0,100)]["x"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribución X"); axes[0].set_xlabel("X (0=Defensa→100=Ataque)")
    axes[1].hist(df[df["y"].between(0,100)]["y"], bins=50, color="tomato", edgecolor="white")
    axes[1].set_title("Distribución Y"); axes[1].set_xlabel("Y (0=Izq→100=Dcha)")
    fig.suptitle("Distribución Espacial de Eventos", fontweight="bold")
    save("09_coordenadas")

    print("[10] Correlación flags...")
    bool_cols = [c for c in ["is_touch","is_shot","is_goal"] if c in df.columns]
    if len(bool_cols) >= 2:
        corr = df[bool_cols].astype(float).corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", ax=ax, linewidths=0.5)
        ax.set_title("Correlación Flags de Evento")
        save("10_correlacion_flags")

print(f"\n✅ EDA events.csv completado. qualifiers={'Sí' if HAS_QUALIFIERS else 'No'}")
print(f"   Gráficas en: {OUT_PLOTS}")
