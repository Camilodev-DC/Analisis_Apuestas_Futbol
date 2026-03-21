"""
eda_events_qualifiers.py — Análisis profundo del campo 'qualifiers' en events.csv
Genera gráficas en: Research/EDA/events/graficas/qualifiers/
Ejecutar con venv activo: python3 Research/EDA/scripts_EDA/eda_events_qualifiers.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

ROOT      = Path(__file__).resolve().parents[3]
DATA_FILE = ROOT / "data" / "raw" / "events.csv"
OUT_PLOTS = ROOT / "Research" / "EDA" / "events" / "graficas" / "qualifiers"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "figure.max_open_warning": 0})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

# ─── Función para dibujar campo de fútbol ────────────────────────────────────
def draw_pitch(ax, color="#2d6a4f", line_color="white", alpha=1.0):
    ax.set_facecolor(color)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    lw, kw = 1.5, dict(fill=False, edgecolor=line_color, linewidth=lw)
    # Campo completo
    ax.add_patch(patches.Rectangle((0, 0), 100, 100, **kw))
    # Línea central
    ax.plot([50, 50], [0, 100], color=line_color, linewidth=lw)
    # Círculo central
    ax.add_patch(patches.Circle((50, 50), 9.15, **kw))
    ax.plot(50, 50, 'o', color=line_color, markersize=3)
    # Área grande local (x=0)
    ax.add_patch(patches.Rectangle((0, 21.1), 17, 57.8, **kw))
    # Área pequeña local
    ax.add_patch(patches.Rectangle((0, 36.8), 5.5, 26.4, **kw))
    # Área grande rival (x=83)
    ax.add_patch(patches.Rectangle((83, 21.1), 17, 57.8, **kw))
    # Área pequeña rival
    ax.add_patch(patches.Rectangle((94.5, 36.8), 5.5, 26.4, **kw))
    # Portería rival
    ax.add_patch(patches.Rectangle((100, 44.5), 2, 11, fill=True,
                                    facecolor="white", edgecolor=line_color, linewidth=lw))
    # Puntos de penalti
    ax.plot(11.5, 50, 'o', color=line_color, markersize=3)
    ax.plot(88.5, 50, 'o', color=line_color, markersize=3)
    # Arcos de área
    ax.add_patch(patches.Arc((11.5, 50), 18.3, 18.3, angle=0, theta1=310, theta2=50,
                               color=line_color, linewidth=lw))
    ax.add_patch(patches.Arc((88.5, 50), 18.3, 18.3, angle=0, theta1=130, theta2=230,
                               color=line_color, linewidth=lw))
    ax.axis("off")

# ─── CARGA Y PARSEO ──────────────────────────────────────────────────────────
print("📦 Cargando events.csv...")
df = pd.read_csv(DATA_FILE)
print(f"   Shape: {df.shape}")

print("🔍 Parseando qualifiers JSON...")
q = df["qualifiers"].fillna("[]")

# Extraer features booleanas desde el JSON string (método rápido via str.contains)
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
df["is_through_ball"]= q.str.contains("ThroughBall",na=False).astype(int)
df["is_cross"]       = q.str.contains("Cross",      na=False).astype(int)

# Extraer tipos de qualifiers únicos
def extract_types(s):
    try:
        items = json.loads(s) if isinstance(s, str) else []
        return [item.get("type", {}).get("displayName","") for item in items if isinstance(item, dict)]
    except:
        return []

print("   Extrayendo tipos únicos de qualifiers (puede tardar ~30s)...")
all_types = df["qualifiers"].dropna().apply(extract_types)
flat = [t for sublist in all_types for t in sublist if t]
from collections import Counter
type_counts = Counter(flat)
print(f"   Total de tipos únicos de qualifiers: {len(type_counts)}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 1 — Top 30 Qualifiers más frecuentes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[1] Top 30 Qualifiers...")
top30 = pd.Series(type_counts).nlargest(30)
fig, ax = plt.subplots(figsize=(13, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top30)))
bars = ax.barh(top30.index[::-1], top30.values[::-1], color=colors[::-1], edgecolor="black", linewidth=0.4)
ax.set_xlabel("Frecuencia", fontsize=12)
ax.set_title(f"Top 30 Qualifiers — Frecuencia en events.csv\n({len(type_counts)} tipos únicos encontrados)",
             fontsize=13, fontweight="bold")
ax.bar_label(bars, padding=3, fontsize=7)
ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
ax.title.set_color("white"); ax.spines['bottom'].set_color('#555')
save("Q01_top30_qualifiers")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 2 — Distribución de features booleanas extraídas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[2] Features booleanas extraídas...")
bool_feats = {
    "Big Chance": df["is_big_chance"].sum(),
    "Header": df["is_header"].sum(),
    "Pie Derecho": df["is_right_foot"].sum(),
    "Pie Izquierdo": df["is_left_foot"].sum(),
    "Contraataque": df["is_counter"].sum(),
    "Desde Córner": df["from_corner"].sum(),
    "Penalti": df["is_penalty"].sum(),
    "Volea": df["is_volley"].sum(),
    "Primer Toque": df["first_touch"].sum(),
    "Jugada a Balón Parado": df["is_set_piece"].sum(),
}
s = pd.Series(bool_feats).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(11, 7))
palette = ["#e63946" if v < 5000 else "#457b9d" if v < 20000 else "#2a9d8f" for v in s.values]
bars = ax.barh(s.index, s.values, color=palette, edgecolor="black", linewidth=0.4)
ax.bar_label(bars, fmt="%,.0f", padding=4, fontsize=9)
ax.set_title("Frequencia de Features Extraídas de Qualifiers\n(Todos los eventos)", fontsize=13, fontweight="bold")
ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.title.set_color("white")
save("Q02_features_booleanas")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 3 — Mapa de tiros: Pie Derecho vs Izquierdo vs Cabeza
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[3] Mapa campo: tipo de contacto en tiros...")
shots = df[df["is_shot"] == True].copy() if "is_shot" in df.columns else df[df["event_type"]=="ShotOnTarget"].copy()
shots = shots[shots["x"].between(50, 100) & shots["y"].between(0, 100)]

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
tipos = [
    (shots[shots["is_right_foot"]==1],  "#e63946", "Pie Derecho"),
    (shots[shots["is_left_foot"]==1],   "#457b9d", "Pie Izquierdo"),
    (shots[shots["is_header"]==1],      "#f4a261", "Cabeza"),
]
for ax, (data, color, label) in zip(axes, tipos):
    draw_pitch(ax, color="#1b4332")
    if len(data) > 0:
        goals = data[data["is_goal"]==True] if "is_goal" in data.columns else data.iloc[0:0]
        no_goals = data[data["is_goal"]==False] if "is_goal" in data.columns else data
        ax.scatter(no_goals["x"], no_goals["y"], c=color, s=15, alpha=0.35, zorder=3, edgecolors="none")
        ax.scatter(goals["x"], goals["y"], c="gold", s=60, alpha=0.9, zorder=5,
                   edgecolors="white", linewidths=0.8, marker="*")
        ax.set_title(f"{label}\n{len(data):,} tiros · {len(goals):,} goles ⭐",
                     color="white", fontsize=11, fontweight="bold", pad=10)
fig.suptitle("Mapa de Tiros por Tipo de Contacto\n⭐ = Gol · Fondo = Terreno de juego",
             color="white", fontsize=14, fontweight="bold")
fig.patch.set_facecolor("#0d1117")
save("Q03_tiros_tipo_contacto_campo")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 4 — Efectividad de gol por tipo de contacto
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[4] Efectividad por tipo de contacto...")
if "is_shot" in df.columns and "is_goal" in df.columns:
    s_df = df[df["is_shot"]==True].copy()
    efectividad = {
        "Pie Derecho": s_df[s_df["is_right_foot"]==1]["is_goal"].mean()*100,
        "Pie Izquierdo": s_df[s_df["is_left_foot"]==1]["is_goal"].mean()*100,
        "Cabeza": s_df[s_df["is_header"]==1]["is_goal"].mean()*100,
        "Penalti": s_df[s_df["is_penalty"]==1]["is_goal"].mean()*100,
        "Volea": s_df[s_df["is_volley"]==1]["is_goal"].mean()*100,
        "Gran Oportunidad": s_df[s_df["is_big_chance"]==1]["is_goal"].mean()*100,
        "Primer Toque": s_df[s_df["first_touch"]==1]["is_goal"].mean()*100,
        "Contraataque": s_df[s_df["is_counter"]==1]["is_goal"].mean()*100,
    }
    eff_s = pd.Series(efectividad).dropna().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    colors_eff = ["#e63946" if v < 10 else "#f4a261" if v < 20 else "#2a9d8f" for v in eff_s.values]
    bars = ax.barh(eff_s.index, eff_s.values, color=colors_eff, edgecolor="black", linewidth=0.4)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold")
    ax.axvline(s_df["is_goal"].mean()*100, color="yellow", linestyle="--", linewidth=2,
               label=f"Global: {s_df['is_goal'].mean()*100:.1f}%")
    ax.set_xlabel("% Efectividad (Goles / Tiros)", fontsize=12)
    ax.set_title("Efectividad de Conversión por Tipo de Acción\n(Extraído de qualifiers JSON)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_facecolor("#1a1a2e"); fig.patch.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.title.set_color("white")
    save("Q04_efectividad_por_tipo")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 5 — Big Chances: Mapa de las "grandes oportunidades" perdidas
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[5] Big Chances en el campo...")
bc = df[df["is_big_chance"]==1][["x","y","is_goal"]].dropna() if "is_goal" in df.columns else df[df["is_big_chance"]==1][["x","y"]]
bc = bc[bc["x"].between(0,100) & bc["y"].between(0,100)]
fig, ax = plt.subplots(figsize=(14, 9))
draw_pitch(ax, color="#052e16", line_color="#a7f3d0")
if "is_goal" in bc.columns:
    missed = bc[bc["is_goal"]==False]
    scored = bc[bc["is_goal"]==True]
    ax.scatter(missed["x"], missed["y"], c="#ef4444", s=80, alpha=0.6, zorder=3,
               edgecolors="white", linewidths=0.5, marker="X", label=f"Fallada ({len(missed)})")
    ax.scatter(scored["x"], scored["y"], c="#fbbf24", s=120, alpha=0.95, zorder=5,
               edgecolors="white", linewidths=1, marker="*", label=f"Gol ⭐ ({len(scored)})")
    ax.legend(loc="lower left", fontsize=11, facecolor="#1a1a2e", labelcolor="white",
              edgecolor="white", framealpha=0.7)
ax.set_title(f"Big Chances — Las Oportunidades Más Claras del Torneo\n"
             f"Total: {len(bc):,} | % Convertidas: {bc['is_goal'].mean()*100:.1f}%",
             color="white", fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#0d1117")
save("Q05_big_chances_campo")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 6 — Contraataques: velocidad y zona de inicio
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[6] Contraataques en el campo...")
counters = df[(df["is_counter"]==1) & df["x"].between(0,100) & df["y"].between(0,100)]
fig, ax = plt.subplots(figsize=(14, 9))
draw_pitch(ax, color="#1e1b4b", line_color="#c7d2fe")
if len(counters) > 0:
    scatter = ax.scatter(counters["x"], counters["y"],
                          c=counters["x"], cmap="plasma", s=20, alpha=0.5, zorder=3, edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="Posición X (zona del campo)")
    if "is_shot" in counters.columns:
        shots_c = counters[counters["is_shot"]==True]
        ax.scatter(shots_c["x"], shots_c["y"], c="white", s=60, zorder=5,
                   edgecolors="#f0abfc", linewidths=1.5, marker="^", label=f"Tiro en contra ({len(shots_c)})")
        ax.legend(loc="lower left", fontsize=10, facecolor="#1e1b4b", labelcolor="white")
ax.set_title(f"Mapa de Eventos en Contraataque (FastBreak)\nTotal: {len(counters):,} acciones",
             color="white", fontsize=13, fontweight="bold")
fig.patch.set_facecolor("#0d1117")
save("Q06_contraataques_campo")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 7 — Goles de córner vs jugada abierta (origen de los goles)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[7] Origen de los goles...")
if "is_goal" in df.columns:
    goals_df = df[df["is_goal"]==True].copy()
    origen = {
        "Jugada Abierta": (goals_df["is_set_piece"] == 0).sum(),
        "Desde Córner": goals_df["from_corner"].sum(),
        "Penalti": goals_df["is_penalty"].sum(),
        "Balón Parado (otro)": (goals_df["is_set_piece"] == 1).sum() - goals_df["from_corner"].sum() - goals_df["is_penalty"].sum(),
    }
    origen = {k: max(0, v) for k, v in origen.items()}
    colors_o = ["#2a9d8f", "#e9c46a", "#e63946", "#457b9d"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    wedges, texts, autotexts = axes[0].pie(
        origen.values(), labels=origen.keys(), colors=colors_o,
        autopct="%1.1f%%", startangle=140, pctdistance=0.7,
        wedgeprops=dict(edgecolor="white", linewidth=1.5))
    for t in autotexts: t.set_fontweight("bold"); t.set_fontsize(11)
    axes[0].set_title("Distribución de Goles por Origen", fontweight="bold")
    pd.Series(origen).sort_values().plot(kind="barh", ax=axes[1], color=colors_o, edgecolor="black")
    axes[1].set_title("Conteo de Goles por Origen"); axes[1].set_xlabel("Goles")
    axes[1].bar_label(axes[1].containers[0], padding=3)
    fig.suptitle("¿De Dónde Vienen los Goles en la Premier League?\n(Extraído de qualifiers JSON)",
                 fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes: ax.set_facecolor("#1a1a2e"); ax.tick_params(colors="white")
    axes[1].xaxis.label.set_color("white"); axes[0].title.set_color("white"); axes[1].title.set_color("white")
    fig.suptitle("¿De Dónde Vienen los Goles en la Premier League?", color="white", fontsize=13, fontweight="bold")
    save("Q07_origen_goles")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRÁFICA 8 — xG implícito por zona + tipo (heatmap conceptual)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("[8] Heatmap xG implícito por zona + tipo de tiro...")
if "is_goal" in df.columns and "is_shot" in df.columns:
    shots_all = df[df["is_shot"]==True][["x","y","is_goal","is_header","is_big_chance","is_penalty"]].dropna()
    shots_all = shots_all[shots_all["x"].between(50,100) & shots_all["y"].between(0,100)]
    shots_all["zona_x"] = pd.cut(shots_all["x"], bins=[50,67,80,90,100], labels=["Lejano","Medio","Cerca","Área"])
    shots_all["zona_y"] = pd.cut(shots_all["y"], bins=[0,30,50,70,100], labels=["Banda Izq","Centro Izq","Centro Dcha","Banda Dcha"])
    pivot = shots_all.groupby(["zona_x","zona_y"])["is_goal"].mean() * 100
    pivot = pivot.unstack()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
                linewidths=0.5, cbar_kws={"label":"% Efectividad"}, annot_kws={"fontsize":11,"fontweight":"bold"})
    ax.set_title("Mapa de Efectividad xG Implícito\nZona X (profundidad) × Zona Y (anchura)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Zona Lateral"); ax.set_ylabel("Zona de Profundidad")
    save("Q08_xg_zona_heatmap")

print(f"\n✅ Análisis de qualifiers completado. {len(list(OUT_PLOTS.iterdir()))} gráficas en: {OUT_PLOTS}")
print("\n📋 Resumen de features extraídas:")
for feat, count in sorted(bool_feats.items(), key=lambda x: -x[1]):
    print(f"   {feat:<25}: {count:>8,}")
print(f"\n   Tipos únicos de qualifiers: {len(type_counts)}")
print(f"   Top 5: {', '.join([k for k,v in type_counts.most_common(5)])}")
