from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "features_modelo1_a_j.csv"
ARTIFACTS_DIR = ROOT / "modelo 1" / "artifacts"
PLOTS_DIR = ROOT / "modelo 1" / "graficas"
MODEL_FILE = ARTIFACTS_DIR / "modelo_1_xg_logistic.joblib"

FINAL_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "is_big_chance",
    "defensive_pressure",
    "buildup_passes",
    "buildup_unique_players",
    "buildup_decentralized",
    "first_touch",
]
TARGET = "is_goal"


def load_split():
    df = pd.read_csv(DATA_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True, errors="coerce")
    selected = ["match_id", "match_date", TARGET] + FINAL_FEATURES
    work = (
        df[selected]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(["match_date", "match_id"])
        .reset_index(drop=True)
    )
    split_idx = int(len(work) * 0.8)
    train_df = work.iloc[:split_idx].copy()
    test_df = work.iloc[split_idx:].copy()
    return train_df, test_df


def save_current(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, bbox_inches="tight", dpi=180)
    plt.close()


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#1d3557", linewidth=2.5, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Azar")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Modelo 1 - Curva ROC")
    plt.legend()
    save_current("01_curva_roc.png")


def plot_calibration(y_true, y_prob):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, color="#e76f51", label="Modelo")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Calibracion perfecta")
    plt.xlabel("Probabilidad media predicha")
    plt.ylabel("Frecuencia real de gol")
    plt.title("Modelo 1 - Curva de Calibracion")
    plt.legend()
    save_current("02_calibracion.png")


def plot_distance_effect(test_df, y_prob):
    work = test_df.copy()
    work["xg_pred"] = y_prob
    work["distance_bin"] = pd.qcut(work["distance_to_goal"], q=10, duplicates="drop")
    grouped = work.groupby("distance_bin", observed=False).agg(
        goal_rate=(TARGET, "mean"),
        xg_pred=("xg_pred", "mean"),
        distance_mid=("distance_to_goal", "mean"),
    )
    plt.figure(figsize=(8, 5))
    plt.plot(grouped["distance_mid"], grouped["goal_rate"], marker="o", linewidth=2, label="Gol real")
    plt.plot(grouped["distance_mid"], grouped["xg_pred"], marker="s", linewidth=2, label="xG medio predicho")
    plt.xlabel("Distancia media al arco")
    plt.ylabel("Probabilidad")
    plt.title("Distancia al arco vs probabilidad de gol")
    plt.legend()
    save_current("03_distancia_vs_gol.png")


def plot_context_bars(test_df):
    summary = pd.DataFrame(
        {
            "feature": ["is_big_chance", "first_touch", "buildup_decentralized"],
            "goal_rate_yes": [
                test_df.loc[test_df["is_big_chance"] == 1, TARGET].mean(),
                test_df.loc[test_df["first_touch"] == 1, TARGET].mean(),
                test_df.loc[test_df["buildup_decentralized"] == 1, TARGET].mean(),
            ],
            "goal_rate_no": [
                test_df.loc[test_df["is_big_chance"] == 0, TARGET].mean(),
                test_df.loc[test_df["first_touch"] == 0, TARGET].mean(),
                test_df.loc[test_df["buildup_decentralized"] == 0, TARGET].mean(),
            ],
        }
    )
    melted = summary.melt(id_vars="feature", var_name="grupo", value_name="goal_rate")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="feature", y="goal_rate", hue="grupo", palette=["#457b9d", "#f4a261"])
    plt.xlabel("")
    plt.ylabel("Frecuencia real de gol")
    plt.title("Impacto de variables contextuales binarias")
    plt.xticks(rotation=12)
    save_current("04_contexto_binario.png")


def plot_pressure_and_buildup(test_df):
    work = test_df.copy()
    work["pressure_bin"] = work["defensive_pressure"].clip(upper=5)
    pressure = work.groupby("pressure_bin", observed=False)[TARGET].mean().reset_index()
    buildup = work.groupby("buildup_unique_players", observed=False)[TARGET].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.barplot(data=pressure, x="pressure_bin", y=TARGET, color="#2a9d8f", ax=axes[0])
    axes[0].set_title("Presion defensiva vs frecuencia de gol")
    axes[0].set_xlabel("Defensive pressure (5 = 5 o mas)")
    axes[0].set_ylabel("Frecuencia de gol")

    sns.lineplot(data=buildup, x="buildup_unique_players", y=TARGET, marker="o", color="#264653", ax=axes[1])
    axes[1].set_title("Jugadores en el buildup vs frecuencia de gol")
    axes[1].set_xlabel("Buildup unique players")
    axes[1].set_ylabel("Frecuencia de gol")
    save_current("05_presion_y_buildup.png")


def plot_model_diagnostics():
    coef_df = pd.read_csv(ARTIFACTS_DIR / "coefficients.csv").sort_values("coefficient")
    vif_df = pd.read_csv(ARTIFACTS_DIR / "vif.csv").sort_values("VIF")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=coef_df, y="feature", x="coefficient", color="#4c78a8", ax=axes[0])
    axes[0].set_title("Coeficientes del modelo logit")
    axes[0].set_xlabel("Coeficiente")
    axes[0].set_ylabel("")

    sns.barplot(data=vif_df, y="feature", x="VIF", color="#6d597a", ax=axes[1])
    axes[1].axvline(5, linestyle="--", color="#e63946", linewidth=1.5, label="VIF=5")
    axes[1].set_title("Diagnostico de multicolinealidad")
    axes[1].set_xlabel("VIF")
    axes[1].set_ylabel("")
    axes[1].legend()
    save_current("06_coeficientes_y_vif.png")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 140})

    _, test_df = load_split()
    model = joblib.load(MODEL_FILE)
    y_prob = model.predict_proba(test_df[FINAL_FEATURES])[:, 1]
    y_true = test_df[TARGET].astype(int)

    plot_roc(y_true, y_prob)
    plot_calibration(y_true, y_prob)
    plot_distance_effect(test_df, y_prob)
    plot_context_bars(test_df)
    plot_pressure_and_buildup(test_df)
    plot_model_diagnostics()

    print(f"Graficas guardadas en: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
