"""
EDA Completo — players.csv
Genera gráficas en: Research/EDA/players/graficas/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[3]
DATA_FILE  = ROOT / "data" / "raw" / "players.csv"
OUT_PLOTS  = ROOT / "Research" / "EDA" / "players" / "graficas"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "figure.max_open_warning": 0})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

print("Cargando players.csv...")
df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")

# 1. NULOS
print("[1] Nulos...")
null_pct  = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_data = null_pct[null_pct > 0]
if len(null_data) > 0:
    fig, ax = plt.subplots(figsize=(10, max(3, len(null_data)*0.4)))
    null_data.plot(kind="barh", ax=ax, color="tomato", edgecolor="black")
    ax.set_xlabel("% Nulos"); ax.set_title("Porcentaje de Nulos — players.csv")
    save("01_nulos")

# 2. DISTRIBUCIÓN DE POSICIONES
print("[2] Posiciones...")
pos_counts = df["position"].value_counts()
colors_pos = sns.color_palette("Set2", len(pos_counts))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
pos_counts.plot(kind="bar", ax=axes[0], color=colors_pos, edgecolor="black")
axes[0].set_title("Jugadores por posición"); axes[0].set_xlabel("Posición"); axes[0].set_ylabel("Cantidad")
axes[1].pie(pos_counts, labels=pos_counts.index, colors=colors_pos, autopct="%1.1f%%", startangle=90)
axes[1].set_title("Proporción por posición")
fig.suptitle("Distribución por Posición — players.csv", fontsize=13, fontweight="bold")
save("02_posiciones")

# 3. STATUS
print("[3] Status...")
map_status = {"a": "Disponible", "d": "Dudoso", "i": "Lesionado", "s": "Suspendido", "u": "No disponible"}
df["status_label"] = df["status"].map(map_status).fillna(df["status"])
st_counts = df["status_label"].value_counts()
fig, ax = plt.subplots(figsize=(8, 5))
st_counts.plot(kind="bar", ax=ax, color=sns.color_palette("RdYlGn_r", len(st_counts)), edgecolor="black")
ax.set_title("Estado de Disponibilidad de Jugadores"); ax.set_ylabel("Cantidad")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
save("03_status")

# 4. DISTRIBUCIONES NUMÉRICAS
print("[4] Distribuciones...")
key_cols = [c for c in ["total_points","minutes","goals_scored","assists",
                         "expected_goals","expected_assists","price","ict_index"] if c in df.columns]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(key_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[i].axvline(data.mean(),   color="red",   linestyle="--", linewidth=1.5, label=f"Media: {data.mean():.2f}")
    axes[i].axvline(data.median(), color="green", linestyle="-",  linewidth=1.5, label=f"Mediana: {data.median():.2f}")
    axes[i].set_title(col, fontsize=10, fontweight="bold"); axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Distribuciones Numéricas — players.csv", fontsize=13, fontweight="bold")
save("04_distribuciones")

# 5. BOXPLOTS POR POSICIÓN
print("[5] Boxplots por posición...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
order = [o for o in ["GKP","DEF","MID","FWD"] if o in df["position"].unique()]
for ax, col in zip(axes, ["total_points","expected_goals","ict_index"]):
    if col in df.columns:
        sns.boxplot(data=df, x="position", y=col, ax=ax, palette="Set2", order=order)
        ax.set_title(f"{col} por posición")
fig.suptitle("Outliers por Posición — players.csv", fontsize=13, fontweight="bold")
save("05_boxplot_posicion")

# 6. OUTLIERS Z-SCORE
print("[6] Z-Score...")
outlier_summary = {}
for col in [c for c in ["total_points","goals_scored","assists","expected_goals","minutes"] if c in df.columns]:
    z = np.abs(stats.zscore(df[col].fillna(0)))
    outlier_summary[col] = (z > 3).sum()
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(outlier_summary.keys(), outlier_summary.values(), color="darkorange", edgecolor="black")
ax.bar_label(bars); ax.set_title("Outliers |Z|>3 por Variable"); ax.set_ylabel("Cantidad")
save("06_outliers_zscore")

# 7. CORRELACIÓN
print("[7] Correlación...")
corr_cols = [c for c in ["total_points","minutes","goals_scored","assists","expected_goals",
                          "expected_assists","ict_index","price","selected_by_percent"] if c in df.columns]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, linewidths=0.5, ax=ax)
ax.set_title("Mapa de Correlación — players.csv", fontsize=13, fontweight="bold")
save("07_correlacion")

# 8. xG vs GOLES REALES
print("[8] xG vs goles...")
if "expected_goals" in df.columns and "goals_scored" in df.columns:
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(df["expected_goals"], df["goals_scored"],
                         c=df["total_points"], cmap="viridis", alpha=0.6,
                         edgecolors="white", linewidths=0.4, s=50)
    lim = max(df["expected_goals"].max(), df["goals_scored"].max()) + 1
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="xG = Goles")
    plt.colorbar(scatter, ax=ax, label="Total Points FPL")
    ax.set_xlabel("xG"); ax.set_ylabel("Goles Reales")
    ax.set_title("Calibración xG vs Goles Reales — players.csv"); ax.legend()
    save("08_xg_vs_goles")

# 9. TOP 20 JUGADORES
print("[9] Top 20...")
from matplotlib.patches import Patch
top20 = df.nlargest(20, "total_points")[["web_name","total_points","position"]]
pal = {"GKP":"#4CAF50","DEF":"#2196F3","MID":"#FF9800","FWD":"#F44336"}
colors = [pal.get(p, "gray") for p in top20["position"]]
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(top20["web_name"], top20["total_points"], color=colors, edgecolor="black")
ax.bar_label(bars, padding=3)
ax.legend(handles=[Patch(color=c, label=p) for p, c in pal.items()], title="Posición")
ax.set_title("Top 20 Jugadores por Puntos FPL"); ax.invert_yaxis()
save("09_top20")

# 10. PRECIO VS RENDIMIENTO
print("[10] Precio vs rendimiento...")
if "price" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    for pos, grp in df.groupby("position"):
        ax.scatter(grp["price"], grp["total_points"], label=pos, alpha=0.6, s=40)
    ax.set_xlabel("Precio (M£)"); ax.set_ylabel("Total Points FPL")
    ax.set_title("Precio vs Rendimiento — players.csv"); ax.legend(title="Posición")
    save("10_precio_vs_puntos")

# 11. PAIR PLOT
print("[11] Pair plot...")
pair_cols = [c for c in ["expected_goals","expected_assists","ict_index","total_points"] if c in df.columns]
if len(pair_cols) >= 3:
    g = sns.pairplot(df[pair_cols + ["position"]].dropna(), hue="position",
                     plot_kws={"alpha": 0.4, "s": 15}, palette="Set2")
    g.fig.suptitle("Pair Plot — Métricas Avanzadas", y=1.02, fontsize=12)
    plt.savefig(OUT_PLOTS / "11_pairplot.png", bbox_inches="tight"); plt.close()
    print("  ✓ 11_pairplot.png")

print(f"\n✅ EDA players.csv completado. {len(list(OUT_PLOTS.iterdir()))} gráficas en: {OUT_PLOTS}")
