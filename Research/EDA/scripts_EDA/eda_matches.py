"""
EDA Completo — matches.csv
Genera gráficas en: Research/EDA/matches/graficas/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[3]
DATA_FILE  = ROOT / "data" / "raw" / "matches.csv"
OUT_PLOTS  = ROOT / "Research" / "EDA" / "matches" / "graficas"
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({"figure.dpi": 150, "figure.max_open_warning": 0})

def save(name):
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / f"{name}.png", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {name}.png")

print("Cargando matches.csv...")
df = pd.read_csv(DATA_FILE)
if "fthg" in df.columns and "ftag" in df.columns:
    df["total_goals"] = df["fthg"] + df["ftag"]
    df["goal_diff"]   = df["fthg"] - df["ftag"]
print(f"Shape: {df.shape}")

# 1. NULOS
print("[1] Nulos...")
null_pct  = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_data = null_pct[null_pct > 0]
if len(null_data) > 0:
    fig, ax = plt.subplots(figsize=(10, max(3, len(null_data)*0.45)))
    null_data.plot(kind="barh", ax=ax, color="tomato", edgecolor="black")
    ax.set_xlabel("% Nulos"); ax.set_title("Porcentaje de Nulos — matches.csv")
    save("01_nulos")

# 2. DISTRIBUCIÓN DE RESULTADOS
print("[2] Resultados...")
if "ftr" in df.columns:
    map_res = {"H": "Local Gana", "D": "Empate", "A": "Visitante Gana"}
    df["ftr_label"] = df["ftr"].map(map_res)
    res = df["ftr_label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cols_res = ["#4CAF50","#FF9800","#F44336"]
    res.plot(kind="bar", ax=axes[0], color=cols_res, edgecolor="black")
    axes[0].set_title("Distribución de Resultados FT"); axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=20)
    axes[1].pie(res, labels=res.index, colors=cols_res, autopct="%1.1f%%", startangle=90)
    axes[1].set_title("Proporción de Resultados")
    fig.suptitle("Desbalance Variable Objetivo — matches.csv", fontweight="bold")
    save("02_resultados")

# 3. DISTRIBUCIÓN DE GOLES
print("[3] Goles...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, title, color in zip(axes, ["fthg","ftag","total_goals"],
    ["Goles Local","Goles Visitante","Total Goles"], ["steelblue","tomato","mediumpurple"]):
    if col in df.columns:
        data = df[col].dropna()
        ax.hist(data, bins=range(0, int(data.max())+2), color=color, edgecolor="white", alpha=0.85)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Media:{data.mean():.2f}")
        ax.set_title(title); ax.legend()
fig.suptitle("Distribución de Goles — matches.csv", fontweight="bold")
save("03_goles")

# 4. GOLES POR EQUIPO LOCAL
print("[4] Goles por equipo...")
if "home_team" in df.columns and "fthg" in df.columns:
    tg = df.groupby("home_team")["fthg"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=tg.index, y=tg.values, ax=ax, palette="coolwarm_r", edgecolor="black")
    ax.axhline(tg.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Promedio: {tg.mean():.2f}")
    ax.set_title("Promedio Goles Local por Equipo"); ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    save("04_goles_equipo_local")

# 5. VENTAJA LOCAL
print("[5] Ventaja local...")
if "goal_diff" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["goal_diff"].dropna(), bins=range(-6, 8), color="teal", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Empate")
    axes[0].set_title("Diferencia de Goles (Local-Visitante)"); axes[0].legend()
    hw = (df["ftr"]=="H").mean()*100; dw = (df["ftr"]=="D").mean()*100; aw = (df["ftr"]=="A").mean()*100
    axes[1].bar(["Local Gana","Empate","Visitante Gana"], [hw,dw,aw],
                color=["#4CAF50","#FF9800","#F44336"], edgecolor="black")
    axes[1].set_ylabel("% Partidos"); axes[1].set_title("Ventaja de Local")
    for bar, val in zip(axes[1].patches, [hw,dw,aw]):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%", ha="center")
    save("05_ventaja_local")

# 6. LOCAL VS VISITANTE ESTADÍSTICAS
print("[6] Local vs Visitante...")
stat_pairs = [("hs","as"),("hst","ast"),("hc","ac")]
labels = ["Tiros","Tiros a puerta","Córners"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (h,a), lbl in zip(axes, stat_pairs, labels):
    if h in df.columns and a in df.columns:
        dm = pd.DataFrame({"Local":df[h],"Visitante":df[a]}).melt(var_name="Equipo",value_name="Valor")
        sns.boxplot(data=dm, x="Equipo", y="Valor", ax=ax, palette={"Local":"steelblue","Visitante":"tomato"})
        ax.set_title(lbl)
fig.suptitle("Local vs Visitante — Estadísticas", fontweight="bold")
save("06_local_vs_visitante")

# 7. CUOTAS
print("[7] Cuotas...")
odds_cols = [c for c in ["b365h","b365d","b365a"] if c in df.columns]
if odds_cols:
    fig, axes = plt.subplots(1, len(odds_cols), figsize=(15, 5))
    if len(odds_cols) == 1: axes = [axes]
    for ax, col in zip(axes, odds_cols):
        data = df[col].dropna()
        ax.hist(data, bins=25, edgecolor="white", alpha=0.85, color="steelblue")
        ax.axvline(data.mean(), color="red", linestyle="--", label=f"Media:{data.mean():.2f}")
        ax.set_title(f"Cuota {col.upper()}"); ax.legend()
    fig.suptitle("Distribución de Cuotas Bet365", fontweight="bold")
    save("07_cuotas")

# 8. CALIBRACIÓN DE CUOTAS
print("[8] Calibración cuotas...")
if "b365h" in df.columns and "ftr" in df.columns:
    df["prob_h"] = 1 / df["b365h"]; df["actual_h"] = (df["ftr"]=="H").astype(int)
    df["bucket"] = pd.cut(df["prob_h"], bins=10)
    cal = df.groupby("bucket")["actual_h"].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([0,1],[0,1],"r--",label="Calibración perfecta")
    mids = [i.mid for i in cal.index]
    ax.plot(mids, cal.values, "o-", color="steelblue", label="Real")
    ax.set_xlabel("Prob. Implícita (Bet365 Local)"); ax.set_ylabel("Tasa Real de Victoria")
    ax.set_title("Curva de Calibración de Cuotas"); ax.legend()
    save("08_calibracion_cuotas")

# 9. CORRELACIÓN
print("[9] Correlación...")
corr_cols = [c for c in ["fthg","ftag","total_goals","hs","hst","as","ast","b365h","b365d","b365a"] if c in df.columns]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", mask=mask, linewidths=0.5, ax=ax)
ax.set_title("Mapa de Correlación — matches.csv", fontweight="bold")
save("09_correlacion")

# 10. GOLES POR MITAD
print("[10] Goles por mitad...")
if all(c in df.columns for c in ["hthg","htag","fthg","ftag"]):
    df["ht_goals"] = df["hthg"] + df["htag"]
    df["ft_goals"] = df["fthg"] + df["ftag"]
    df["2h_goals"] = df["ft_goals"] - df["ht_goals"]
    fig, ax = plt.subplots(figsize=(9, 5))
    pd.DataFrame({"1ª Mitad": df["ht_goals"], "2ª Mitad": df["2h_goals"]}).plot(
        kind="hist", bins=range(0,11), ax=ax, alpha=0.7, edgecolor="white", color=["#2196F3","#FF9800"])
    ax.set_xlabel("Goles"); ax.set_title("Goles por Mitad — matches.csv"); ax.legend()
    save("10_goles_por_mitad")

print(f"\n✅ EDA matches.csv completado. {len(list(OUT_PLOTS.iterdir()))} gráficas en: {OUT_PLOTS}")
