import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "features_modelo1_a_j.csv"
ARTIFACTS_DIR = ROOT / "modelo 1" / "artifacts"

MANDATORY_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
]

ADVANCED_FEATURES = [
    "is_big_chance",
    "defensive_pressure",
    "buildup_passes",
    "buildup_unique_players",
    "buildup_decentralized",
    "first_touch",
]

FINAL_FEATURES = MANDATORY_FEATURES + ADVANCED_FEATURES
TARGET = "is_goal"


def load_dataset():
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
    return work


def time_split(df, test_size=0.2):
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def compute_vif(df, features):
    X = add_constant(df[features], has_constant="add")
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        rows.append(
            {
                "feature": col,
                "VIF": variance_inflation_factor(X.values, i),
            }
        )
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def train_model(train_df, test_df):
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    X_train = train_df[FINAL_FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[FINAL_FEATURES]
    y_test = test_df[TARGET].astype(int)

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    labels = (probabilities >= 0.5).astype(int)

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "goal_rate_test": float(y_test.mean()),
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "accuracy_05": float(accuracy_score(y_test, labels)),
    }

    coefficients = pd.DataFrame(
        {
            "feature": FINAL_FEATURES,
            "coefficient": pipeline.named_steps["model"].coef_[0],
            "odds_ratio": np.exp(pipeline.named_steps["model"].coef_[0]),
        }
    ).sort_values("coefficient", ascending=False)

    predictions = test_df[["match_id", "match_date", TARGET]].copy()
    predictions["xg_logistic"] = probabilities
    predictions["pred_label_05"] = labels

    return pipeline, metrics, coefficients, predictions


def fit_logit(train_df):
    X = sm.add_constant(train_df[FINAL_FEATURES])
    y = train_df[TARGET].astype(int)
    return sm.Logit(y, X).fit(disp=False)


def build_report(metrics, vif_df, coefficients):
    top_positive = coefficients.sort_values("coefficient", ascending=False).head(4)
    top_negative = coefficients.sort_values("coefficient", ascending=True).head(4)

    vif_lines = "\n".join(
        f"| `{row.feature}` | {row.VIF:.2f} |" for row in vif_df.itertuples()
    )
    pos_lines = "\n".join(
        f"| `{row.feature}` | {row.coefficient:.4f} | {row.odds_ratio:.3f} |"
        for row in top_positive.itertuples()
    )
    neg_lines = "\n".join(
        f"| `{row.feature}` | {row.coefficient:.4f} | {row.odds_ratio:.3f} |"
        for row in top_negative.itertuples()
    )

    report = f"""# Reporte Modelo 1

## Enfoque

Modelo de Regresion Logistica para estimar `xG` a nivel de tiro. El target es binario (`is_goal`) y la salida del modelo es una probabilidad interpretable en `[0,1]`.

## Features finales

- Obligatorias: `distance_to_goal`, `angle_to_goal`
- Avanzadas: `is_big_chance`, `defensive_pressure`, `buildup_passes`, `buildup_unique_players`, `buildup_decentralized`, `first_touch`

## Metricas

- Train rows: {metrics["train_rows"]}
- Test rows: {metrics["test_rows"]}
- Goal rate test: {metrics["goal_rate_test"]:.4f}
- AUC-ROC: {metrics["auc_roc"]:.4f}
- Log Loss: {metrics["log_loss"]:.4f}
- Brier Score: {metrics["brier_score"]:.4f}
- Accuracy @ 0.5: {metrics["accuracy_05"]:.4f}

## VIF

| Feature | VIF |
| --- | ---: |
{vif_lines}

## Coeficientes positivos mas fuertes

| Feature | Coeficiente | Odds Ratio |
| --- | ---: | ---: |
{pos_lines}

## Coeficientes negativos mas fuertes

| Feature | Coeficiente | Odds Ratio |
| --- | ---: | ---: |
{neg_lines}

## Lectura futbolistica

- `is_big_chance` aumenta con fuerza la probabilidad de gol, como era esperable para una ocasion clara.
- `defensive_pressure` reduce la probabilidad al capturar incomodidad de ejecucion.
- Las features de `buildup` ayudan a separar tiros creados en secuencias mas limpias frente a posesiones donde la defensa ya alcanzo a cerrar lineas.
- `first_touch` agrega una capa biomecanica y temporal de ejecucion del remate.

## Nota metodologica

- Se uso split temporal por `match_date`.
- Se excluyeron features con riesgo de leakage post-shot como `porteria_zone_*`.
- Se excluyeron derivadas geometricas como `dist_squared` y `dist_angle` para no inflar multicolinealidad.
- Se usa `class_weight="balanced"` por el desbalance natural entre goles y no goles.

## Logit

El resumen completo esta disponible en `logit_summary.txt`.
"""
    return report


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df, test_df = time_split(df)

    vif_df = compute_vif(train_df, FINAL_FEATURES)
    model, metrics, coefficients, predictions = train_model(train_df, test_df)
    logit_model = fit_logit(train_df)
    report = build_report(metrics, vif_df, coefficients)

    train_df.to_csv(ARTIFACTS_DIR / "train_dataset_modelo_1.csv", index=False)
    predictions.to_csv(ARTIFACTS_DIR / "test_predictions_modelo_1.csv", index=False)
    coefficients.to_csv(ARTIFACTS_DIR / "coefficients.csv", index=False)
    vif_df.to_csv(ARTIFACTS_DIR / "vif.csv", index=False)
    pd.DataFrame({"feature": FINAL_FEATURES}).to_csv(
        ARTIFACTS_DIR / "selected_features.csv", index=False
    )
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "logit_summary.txt").write_text(
        logit_model.summary().as_text(), encoding="utf-8"
    )
    (ARTIFACTS_DIR / "reporte_modelo_1.md").write_text(report, encoding="utf-8")
    joblib.dump(model, ARTIFACTS_DIR / "modelo_1_xg_logistic.joblib")

    print("Modelo 1 logistico entrenado")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")


if __name__ == "__main__":
    main()
