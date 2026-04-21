import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "features_modelo1_a_j.csv"
OUT_DIR = ROOT / "modelo 1" / "comparacion_variantes"
TARGET = "is_goal"
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


def load_dataset():
    df = pd.read_csv(DATA_FILE)
    df["match_date"] = pd.to_datetime(df["match_date"], dayfirst=True, errors="coerce")
    cols = ["match_id", "match_date", TARGET] + FINAL_FEATURES
    work = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values(["match_date", "match_id"])
        .reset_index(drop=True)
    )
    return work


def time_split(df, test_size=0.2):
    split = int(len(df) * (1 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def build_base_pipeline(class_weight=None):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "variant": name,
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "accuracy_05": float(accuracy_score(y_test, pred)),
        "mean_pred_xg": float(proba.mean()),
        "actual_goal_rate": float(y_test.mean()),
        "predicted_goals_sum": float(proba.sum()),
        "actual_goals_sum": float(y_test.sum()),
        "overestimate_ratio": float(proba.sum() / max(float(y_test.sum()), 1.0)),
    }, proba, model


def plot_calibration_compare(y_test, variant_probs):
    plt.figure(figsize=(7.5, 5.5))
    for name, proba, color in variant_probs:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=name, color=color)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#999999", label="Perfecta")
    plt.xlabel("Probabilidad media predicha")
    plt.ylabel("Frecuencia real de gol")
    plt.title("Comparacion de calibracion - Modelo 1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "calibracion_variantes.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_mean_probability(results_df):
    melted = results_df[
        ["variant", "mean_pred_xg", "actual_goal_rate"]
    ].melt(id_vars="variant", var_name="metric", value_name="value")
    plt.figure(figsize=(8, 5))
    colors = ["#457b9d", "#f4a261"]
    for i, metric in enumerate(["mean_pred_xg", "actual_goal_rate"]):
        subset = melted[melted["metric"] == metric]
        plt.bar(
            np.arange(len(subset)) + (i - 0.5) * 0.35,
            subset["value"],
            width=0.35,
            label=metric,
            color=colors[i],
        )
    plt.xticks(np.arange(len(results_df)), results_df["variant"], rotation=10)
    plt.ylabel("Promedio")
    plt.title("xG medio predicho vs tasa real de gol")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "promedio_probabilidades.png", dpi=180, bbox_inches="tight")
    plt.close()


def build_report(results_df):
    best_brier = results_df.sort_values("brier_score").iloc[0]["variant"]
    best_logloss = results_df.sort_values("log_loss").iloc[0]["variant"]
    best_auc = results_df.sort_values("auc_roc", ascending=False).iloc[0]["variant"]
    closest_rate = results_df.iloc[
        (results_df["mean_pred_xg"] - results_df["actual_goal_rate"]).abs().argmin()
    ]["variant"]

    table_lines = "\n".join(
        f"| {row.variant} | {row.auc_roc:.4f} | {row.log_loss:.4f} | {row.brier_score:.4f} | {row.mean_pred_xg:.4f} | {row.actual_goal_rate:.4f} | {row.overestimate_ratio:.2f} |"
        for row in results_df.itertuples()
    )

    report = f"""# Comparacion de variantes logisticas - Modelo 1

## Objetivo

Comparar tres versiones del Modelo 1 para decidir cual sirve mejor como `xG` final:

- Logistica con `class_weight="balanced"`
- Logistica sin `class_weight`
- Logistica calibrada sobre la version sin pesos

## Resultados

| Variante | AUC | Log Loss | Brier | xG medio predicho | Tasa real de gol | Ratio sobreestimacion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table_lines}

## Lectura

- Mejor AUC: `{best_auc}`
- Mejor Log Loss: `{best_logloss}`
- Mejor Brier Score: `{best_brier}`
- Variante mas cercana a la tasa real de gol: `{closest_rate}`

## Conclusion recomendada

Si el objetivo es `xG` como probabilidad creible, la mejor candidata debe priorizar calibracion (`Brier`, `Log Loss` y cercania a la tasa real) por encima de un pequeño beneficio en discriminacion.

En este contexto, la variante recomendada es la que mejor balancee:

1. Probabilidades realistas
2. Buena separacion entre tiros peligrosos y no peligrosos
3. Menor sobreestimacion agregada de goles

## Graficas

- `calibracion_variantes.png`
- `promedio_probabilidades.png`
"""
    return report


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    train_df, test_df = time_split(df)

    X_train = train_df[FINAL_FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[FINAL_FEATURES]
    y_test = test_df[TARGET].astype(int)

    balanced_model = build_base_pipeline(class_weight="balanced")
    unweighted_model = build_base_pipeline(class_weight=None)

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_proc = scaler.fit_transform(imp.fit_transform(X_train))
    X_test_proc = scaler.transform(imp.transform(X_test))
    calibrated_base = LogisticRegression(max_iter=2000, class_weight=None)
    calibrated_model = CalibratedClassifierCV(calibrated_base, method="sigmoid", cv=5)

    results = []

    res_bal, proba_bal, model_bal = evaluate(
        "logit_balanced", balanced_model, X_train, y_train, X_test, y_test
    )
    results.append(res_bal)

    res_unw, proba_unw, model_unw = evaluate(
        "logit_unweighted", unweighted_model, X_train, y_train, X_test, y_test
    )
    results.append(res_unw)

    calibrated_model.fit(X_train_proc, y_train)
    proba_cal = calibrated_model.predict_proba(X_test_proc)[:, 1]
    pred_cal = (proba_cal >= 0.5).astype(int)
    res_cal = {
        "variant": "logit_calibrated",
        "auc_roc": float(roc_auc_score(y_test, proba_cal)),
        "log_loss": float(log_loss(y_test, proba_cal, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, proba_cal)),
        "accuracy_05": float(accuracy_score(y_test, pred_cal)),
        "mean_pred_xg": float(proba_cal.mean()),
        "actual_goal_rate": float(y_test.mean()),
        "predicted_goals_sum": float(proba_cal.sum()),
        "actual_goals_sum": float(y_test.sum()),
        "overestimate_ratio": float(proba_cal.sum() / max(float(y_test.sum()), 1.0)),
    }
    results.append(res_cal)

    results_df = pd.DataFrame(results).sort_values("brier_score").reset_index(drop=True)
    results_df.to_csv(OUT_DIR / "comparacion_metricas.csv", index=False)
    (OUT_DIR / "comparacion_metricas.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    plot_calibration_compare(
        y_test,
        [
            ("Balanced", proba_bal, "#e63946"),
            ("Unweighted", proba_unw, "#457b9d"),
            ("Calibrated", proba_cal, "#2a9d8f"),
        ],
    )
    plot_mean_probability(results_df)

    report = build_report(results_df)
    (OUT_DIR / "comparacion_variantes.md").write_text(report, encoding="utf-8")

    joblib.dump(model_bal, OUT_DIR / "logit_balanced.joblib")
    joblib.dump(model_unw, OUT_DIR / "logit_unweighted.joblib")
    joblib.dump(
        {
            "imputer": imp,
            "scaler": scaler,
            "calibrated_model": calibrated_model,
            "features": FINAL_FEATURES,
        },
        OUT_DIR / "logit_calibrated.joblib",
    )

    print("Comparacion completada")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
