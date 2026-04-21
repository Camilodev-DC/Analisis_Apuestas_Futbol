import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
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


def save_confusion_matrix(metrics):
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1], labels=["No gol", "Gol"])
    ax.set_yticks([0, 1], labels=["No gol", "Gol"])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion - Modelo 1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close()


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
            ("model", LogisticRegression(max_iter=2000, class_weight=None)),
        ]
    )

    X_train = train_df[FINAL_FEATURES]
    y_train = train_df[TARGET].astype(int)
    X_test = test_df[FINAL_FEATURES]
    y_test = test_df[TARGET].astype(int)

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    labels = (probabilities >= 0.5).astype(int)
    naive_labels = np.zeros_like(y_test)
    tn, fp, fn, tp = confusion_matrix(y_test, labels).ravel()

    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "goal_rate_test": float(y_test.mean()),
        "auc_roc": float(roc_auc_score(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "accuracy_05": float(accuracy_score(y_test, labels)),
        "precision_05": float(precision_score(y_test, labels, zero_division=0)),
        "recall_05": float(recall_score(y_test, labels, zero_division=0)),
        "f1_05": float(f1_score(y_test, labels, zero_division=0)),
        "baseline_accuracy_naive_no_goal": float(accuracy_score(y_test, naive_labels)),
        "mean_pred_xg": float(probabilities.mean()),
        "predicted_goals_sum": float(probabilities.sum()),
        "actual_goals_sum": float(y_test.sum()),
        "overestimate_ratio": float(probabilities.sum() / max(float(y_test.sum()), 1.0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
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
- Precision @ 0.5: {metrics["precision_05"]:.4f}
- Recall @ 0.5: {metrics["recall_05"]:.4f}
- F1 @ 0.5: {metrics["f1_05"]:.4f}
- Baseline naive accuracy: {metrics["baseline_accuracy_naive_no_goal"]:.4f}
- xG medio predicho: {metrics["mean_pred_xg"]:.4f}
- Tasa real de gol: {metrics["goal_rate_test"]:.4f}

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

## Matriz de confusion @ 0.5

| Real \\ Pred | No gol | Gol |
| --- | ---: | ---: |
| No gol | {metrics["tn"]} | {metrics["fp"]} |
| Gol | {metrics["fn"]} | {metrics["tp"]} |

Interpretacion:

- el baseline naive gana mucha `accuracy` porque casi todos los tiros son `no gol`
- aun asi, el modelo es mucho mas util porque entrega probabilidades y separa mejor los tiros de alta calidad

## Por que no dejamos `RightFoot`, `LeftFoot`, `Head` y zonas de disparo como features finales independientes

- `RightFoot`, `LeftFoot` y `Head` si se construyeron en el feature engineering y existen en la tabla procesada.
- No quedaron en el set final porque entre si forman un bloque muy redundante: casi todos los tiros pertenecen a una de esas categorias y eso introduce colinealidad estructural.
- En pruebas de VIF previas, estas variables elevaban inestabilidad e inflaban la interpretacion lineal del modelo.
- Su informacion no se perdio del todo: parte del contexto del remate queda absorbido por `is_big_chance`, `first_touch`, `defensive_pressure` y la propia geometria del tiro.

- Las zonas de disparo tipo `BoxCentre`, `OutOfBoxCentre` o `SmallBoxCentre` tambien aparecen dentro de los `qualifiers` y conceptualmente ya estaban representadas por las variables geometricas.
- En el feature engineering, esa idea de zona se resume principalmente en `distance_to_goal`, `angle_to_goal` y, en exploracion interna, en variables como `is_in_area` e `is_central`.
- No se dejaron como bloque final separado porque duplicaban la informacion espacial ya contenida en la geometria y empeoraban la parsimonia del modelo.

## Nota metodologica

- Se uso split temporal por `match_date`.
- Se excluyeron features con riesgo de leakage post-shot como `porteria_zone_*`.
- Se excluyeron derivadas geometricas como `dist_squared` y `dist_angle` para no inflar multicolinealidad.
- Se dejo la variante `unweighted` como modelo oficial porque la version `balanced` sobreestimaba fuertemente la probabilidad de gol.

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
    save_confusion_matrix(metrics)
    joblib.dump(model, ARTIFACTS_DIR / "modelo_1_xg_logistic.joblib")

    print("Modelo 1 logistico entrenado")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")


if __name__ == "__main__":
    main()
