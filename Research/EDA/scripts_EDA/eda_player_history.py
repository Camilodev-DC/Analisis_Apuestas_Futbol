"""
EDA Completo — player_history.csv
Genera gráficas en: Research/EDA/player_history/graficas/
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
DATA_FILE  = ROOT / "data" / "raw" / "player_history.csv"
OUT_PLOTS  = ROOT / "Research" / "EDA" / "player_history" / "graficas"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "figure.max_open_warning": 0})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

print("Cargando player_history.csv...")
df = pd.read_csv(DATA_FILE)
print(f"Shape: {df.shape}")

# 1. NULOS
print("[1] Nulos...")
null_pct  = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_data = null_pct[null_pct > 0]
if len(null_data) > 0:
    fig, ax = plt.subplots(figsize=(10, max(3, len(null_data)*0.45)))
    null_data.plot(kind="barh", ax=ax, color="tomato", edgecolor="black")
    ax.set_xlabel("% Nulos"); ax.set_title("Nulos — player_history.csv")
    save("01_nulos")
else:
    print("  Sin nulos detectados.")

# 2. DISTRIBUCIONES NUMÉRICAS
print("[2] Distribuciones...")
num_cols = [c for c in ["total_points","minutes","goals_scored","assists",
                         "expected_goals","expected_assists","influence","creativity","threat"] if c in df.columns]
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    data = df[col].dropna()
    axes[i].hist(data, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    axes[i].axvline(data.mean(),   color="red",   linestyle="--", linewidth=1.5, label=f"Media:{data.mean():.2f}")
    axes[i].axvline(data.median(), color="green", linestyle="-",  linewidth=1.5, label=f"Mediana:{data.median():.2f}")
    axes[i].set_title(col, fontsize=9, fontweight="bold"); axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)): axes[j].set_visible(False)
fig.suptitle("Distribuciones Numéricas — player_history.csv", fontweight="bold")
save("02_distribuciones")

# 3. PUNTOS POR JORNADA
print("[3] Puntos por jornada...")
if "gameweek" in df.columns:
    gw_avg = df.groupby("gameweek")["total_points"].agg(["mean","std"]).reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(gw_avg["gameweek"], gw_avg["mean"], "o-", color="steelblue", linewidth=2, label="Promedio")
    ax.fill_between(gw_avg["gameweek"], gw_avg["mean"]-gw_avg["std"],
                    gw_avg["mean"]+gw_avg["std"], alpha=0.2, color="steelblue", label="±1 Std")
    ax.set_xlabel("Jornada"); ax.set_ylabel("Puntos FPL Promedio")
    ax.set_title("Evolución de Puntos por Jornada"); ax.legend(); ax.grid(True, alpha=0.5)
    save("03_puntos_jornada")

# 4. LOCAL VS VISITANTE
print("[4] Local vs Visitante...")
if "was_home" in df.columns:
    df["condicion"] = df["was_home"].map({1: "Local", 0: "Visitante"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["total_points","goals_scored","assists"]):
        if col in df.columns:
            sns.boxplot(data=df, x="condicion", y=col, ax=ax,
                        palette={"Local":"steelblue","Visitante":"tomato"})
            ax.set_title(f"{col}")
    fig.suptitle("Rendimiento Local vs Visitante — player_history.csv", fontweight="bold")
    save("04_local_vs_visitante")

# 5. MINUTOS JUGADOS
print("[5] Minutos...")
if "minutes" in df.columns:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(df["minutes"], bins=range(0, 95, 5), color="mediumpurple", edgecolor="white", alpha=0.85)
    ax.axvline(45, color="orange", linestyle="--", linewidth=2, label="45'")
    ax.axvline(90, color="red",    linestyle="--", linewidth=2, label="90' (Completo)")
    ax.set_xlabel("Minutos"); ax.set_title("Distribución Minutos Jugados por Jornada"); ax.legend()
    save("05_minutos")

# 6. TOP JUGADORES HISTÓRICO
print("[6] Top jugadores...")
if "web_name" in df.columns:
    top = df.groupby("web_name")["total_points"].sum().nlargest(20).reset_index()
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=top, x="total_points", y="web_name", ax=ax, palette="viridis", edgecolor="black")
    ax.set_title("Top 20 Jugadores — Puntos FPL Acumulados (Historial)")
    save("06_top20_historico")

# 7. CORRELACIÓN
print("[7] Correlación...")
corr_cols = [c for c in ["total_points","minutes","goals_scored","assists",
                          "expected_goals","expected_assists","influence","creativity","threat"] if c in df.columns]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, linewidths=0.5, ax=ax)
ax.set_title("Mapa de Correlación — player_history.csv", fontweight="bold")
save("07_correlacion")

# 8. xG vs GOLES POR JORNADA
print("[8] xG vs goles...")
if "expected_goals" in df.columns and "goals_scored" in df.columns:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["expected_goals"], df["goals_scored"], alpha=0.3, s=15, color="steelblue", edgecolors="white", linewidths=0.2)
    lim = max(df["expected_goals"].max(), df["goals_scored"].max()) + 0.5
    ax.plot([0,lim],[0,lim],"r--",linewidth=1.5,label="xG=Goles")
    ax.set_xlabel("xG"); ax.set_ylabel("Goles Reales")
    ax.set_title("Calibración xG vs Goles por Jornada"); ax.legend()
    save("08_xg_calibracion")

# 9. BOXPLOT POR JORNADA
print("[9] Boxplot jornadas...")
if "gameweek" in df.columns:
    fig, ax = plt.subplots(figsize=(16, 6))
    df.boxplot(column="total_points", by="gameweek", ax=ax,
               patch_artist=True, boxprops=dict(facecolor="lightblue"),
               medianprops=dict(color="red", linewidth=2))
    plt.suptitle("")
    ax.set_title("Distribución Puntos por Jornada (Outliers)")
    ax.set_xlabel("Jornada"); ax.set_ylabel("Puntos FPL")
    save("09_boxplot_jornada")

# 10. DESBALANCE NO JUGÓ
print("[10] Desbalance no jugó...")
if "minutes" in df.columns:
    df["jugó"] = df["minutes"].apply(lambda x: "Jugó" if x > 0 else "No jugó")
    j_counts = df["jugó"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    j_counts.plot(kind="bar", ax=axes[0], color=["#4CAF50","#F44336"], edgecolor="black")
    axes[0].set_title("Registros con/sin minutos")
    axes[1].pie(j_counts, labels=j_counts.index, autopct="%1.1f%%", colors=["#4CAF50","#F44336"], startangle=90)
    axes[1].set_title("Proporción Jugó / No jugó")
    fig.suptitle("Desbalance — Jornadas No Jugadas", fontweight="bold")
    save("10_desbalance_minutos")

print(f"\n✅ EDA player_history.csv completado. {len(list(OUT_PLOTS.iterdir()))} gráficas en: {OUT_PLOTS}")
