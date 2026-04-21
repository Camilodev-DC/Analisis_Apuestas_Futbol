import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_features_modelo_2 import FEATURES_FILE, build_features_dataset


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "modelo_2" / "artifacts"
N_SPLITS = 5

REGRESSION_FEATURES = [
    "ref_home_win_rate_pre",
    "ref_avg_total_goals_pre",
    "ref_card_bias_pre",
]

CLASSIFICATION_FEATURES = [
    "implied_prob_h",
    "implied_prob_d",
    "implied_prob_a",
    "bookmaker_spread_home",
    "bookmaker_spread_draw",
    "bookmaker_spread_away",
    "market_entropy",
]


def load_features():
    if not FEATURES_FILE.exists():
        return build_features_dataset()
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = [
        c for c in df.columns
        if c not in {"match_id", "match_date", "home_team", "away_team", "referee", "ftr", "total_goals"}
    ]
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.sort_values(["match_date", "match_id"]).reset_index(drop=True)
    return df, feature_cols


def build_regression_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def build_classification_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                ),
            ),
        ]
    )


def bet365_prediction(df):
    mapping = {
        "implied_prob_h": "H",
        "implied_prob_d": "D",
        "implied_prob_a": "A",
    }
    return df[["implied_prob_h", "implied_prob_d", "implied_prob_a"]].idxmax(axis=1).map(mapping)


def run_cross_validation(df):
    X_reg = df[REGRESSION_FEATURES]
    y_reg = df["total_goals"].astype(float)
    X_clf = df[CLASSIFICATION_FEATURES]
    y_clf = df["ftr"].astype(str)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    reg_rows = []
    clf_rows = []

    clf_oof_parts = []
    reg_oof_parts = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        X_reg_train, X_reg_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
        y_reg_train, y_reg_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        X_clf_train, X_clf_test = X_clf.iloc[train_idx], X_clf.iloc[test_idx]
        y_clf_train, y_clf_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]

        reg_model = build_regression_pipeline()
        reg_model.fit(X_reg_train, y_reg_train)
        reg_pred = reg_model.predict(X_reg_test)
        reg_rows.append(
            {
                "fold": fold,
                "rmse": float(np.sqrt(mean_squared_error(y_reg_test, reg_pred))),
                "mae": float(mean_absolute_error(y_reg_test, reg_pred)),
                "r2": float(r2_score(y_reg_test, reg_pred)),
            }
        )
        reg_fold_df = df.iloc[test_idx][["match_id", "match_date", "total_goals"]].copy()
        reg_fold_df["pred_total_goals"] = reg_pred
        reg_fold_df["fold"] = fold
        reg_oof_parts.append(reg_fold_df)

        clf_model = build_classification_pipeline()
        clf_model.fit(X_clf_train, y_clf_train)
        clf_pred = clf_model.predict(X_clf_test)
        clf_proba = clf_model.predict_proba(X_clf_test)
        bet365_pred = bet365_prediction(df.iloc[test_idx])
        clf_rows.append(
            {
                "fold": fold,
                "accuracy": float(accuracy_score(y_clf_test, clf_pred)),
                "f1_macro": float(f1_score(y_clf_test, clf_pred, average="macro")),
                "log_loss": float(log_loss(y_clf_test, clf_proba, labels=clf_model.named_steps["model"].classes_)),
                "bet365_accuracy": float(accuracy_score(y_clf_test, bet365_pred)),
            }
        )

        clf_fold_df = df.iloc[test_idx][["match_id", "match_date", "ftr"]].copy()
        clf_fold_df["pred_ftr"] = clf_pred
        clf_fold_df["bet365_pred"] = bet365_pred.values
        clf_fold_df["fold"] = fold
        for i, cls in enumerate(clf_model.named_steps["model"].classes_):
            clf_fold_df[f"proba_{cls}"] = clf_proba[:, i]
        clf_oof_parts.append(clf_fold_df)

    reg_cv = pd.DataFrame(reg_rows)
    clf_cv = pd.DataFrame(clf_rows)
    reg_oof = pd.concat(reg_oof_parts, ignore_index=True)
    clf_oof = pd.concat(clf_oof_parts, ignore_index=True)
    return reg_cv, clf_cv, reg_oof, clf_oof


def fit_final_models(df):
    X_reg = df[REGRESSION_FEATURES]
    y_reg = df["total_goals"].astype(float)
    X_clf = df[CLASSIFICATION_FEATURES]
    y_clf = df["ftr"].astype(str)

    reg_model = build_regression_pipeline()
    reg_model.fit(X_reg, y_reg)

    clf_model = build_classification_pipeline()
    clf_model.fit(X_clf, y_clf)

    reg_coef = pd.DataFrame(
        {
            "feature": REGRESSION_FEATURES,
            "coefficient": reg_model.named_steps["model"].coef_,
        }
    ).sort_values("coefficient", ascending=False)

    logistic_rows = []
    classes = clf_model.named_steps["model"].classes_
    for cls, coef_row in zip(classes, clf_model.named_steps["model"].coef_):
        for feature, coef in zip(CLASSIFICATION_FEATURES, coef_row):
            logistic_rows.append({"class": cls, "feature": feature, "coefficient": coef})
    logistic_coef = pd.DataFrame(logistic_rows).sort_values(["class", "coefficient"], ascending=[True, False])
    return reg_model, clf_model, reg_coef, logistic_coef


def plot_confusion(clf_oof):
    order = ["H", "D", "A"]
    cm = confusion_matrix(clf_oof["ftr"], clf_oof["pred_ftr"], labels=order)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=order, yticklabels=order, ax=ax)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Modelo 2 - Matriz de confusion multiclase")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix_multiclass.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_regression_residuals(reg_oof):
    reg_oof = reg_oof.copy()
    reg_oof["residual"] = reg_oof["total_goals"] - reg_oof["pred_total_goals"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.scatterplot(data=reg_oof, x="pred_total_goals", y="residual", color="#1d3557", ax=ax)
    ax.axhline(0, linestyle="--", color="#999999")
    ax.set_title("Modelo 2 - Residuales regresion total_goals")
    ax.set_xlabel("Goles predichos")
    ax.set_ylabel("Residual")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "regression_residuals.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_accuracy_benchmark(clf_cv):
    summary = pd.DataFrame(
        {
            "metric": ["Modelo 2", "Bet365"],
            "accuracy": [clf_cv["accuracy"].mean(), clf_cv["bet365_accuracy"].mean()],
        }
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.barplot(data=summary, x="metric", y="accuracy", color="#2a9d8f", ax=ax)
    ax.patches[1].set_color("#e76f51")
    ax.axhline(0.498, linestyle="--", color="#555555", linewidth=1.5, label="Benchmark 49.8%")
    ax.set_ylim(0, max(0.65, summary["accuracy"].max() + 0.05))
    ax.set_ylabel("Accuracy media CV")
    ax.set_xlabel("")
    ax.set_title("Accuracy Modelo 2 vs Bet365")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "accuracy_vs_bet365.png", dpi=180, bbox_inches="tight")
    plt.close()


def build_report(df, feature_cols, reg_cv, clf_cv):
    model_acc = clf_cv["accuracy"].mean()
    market_acc = clf_cv["bet365_accuracy"].mean()
    report = f"""# Reporte Modelo 2

## Objetivo

Modelo 2 resuelve dos tareas con validacion temporal honesta:

- Parte A: `Regresion Lineal` para `total_goals`
- Parte B: `Regresion Logistica Multiclase` para `ftr`

## Datos

- Partidos usados: {len(df)}
- Features construidas en tabla maestra: {len(feature_cols)}
- Fuente base: `matches.csv` + agregados de `events_rich.csv` + `xG` del Modelo 1

## Decisiones metodologicas importantes

- No se usan tiros, corners o tarjetas del mismo partido como features directas porque eso seria leakage.
- En su lugar se usan rolling averages pre-match.
- Los datos de eventos se agregan por equipo y partido.
- La fuerza de equipo se estima desde una tabla acumulada pre-match.
- La validacion se hace con `TimeSeriesSplit`, no con split aleatorio.

## Features clave

- Mercado: probabilidades implicitas, spreads de bookmakers, entropia del mercado
- Arbitro: `ref_home_win_rate_pre`, `ref_avg_total_goals_pre`, `ref_card_bias_pre`
- Strength: ratings pre-match del local y visitante
- Forma reciente: goles, tiros a puerta, corners, tarjetas, puntos
- Tactica reciente: `xg`, `pass_accuracy`, `progressive_passes`, `big_chances`, `high_press_pct`, `crosses`

## Seleccion final de variables por tarea

### Regresion lineal

Se quedo con el bloque `referee`, porque fue el mejor subset honesto en CV para `RMSE`.

- Features usadas: {len(REGRESSION_FEATURES)}

### Logistica multiclase

Se quedo con el bloque `odds` solamente, porque fue el mejor subset honesto en CV para accuracy.

- Features usadas: {len(CLASSIFICATION_FEATURES)}

Las demas features si se construyeron y quedan disponibles en la tabla maestra, pero no mejoraron el desempeno CV en esta muestra.

## Parte A - Regresion Lineal (`total_goals`)

- RMSE medio CV: {reg_cv["rmse"].mean():.4f} (+/- {reg_cv["rmse"].std():.4f})
- MAE medio CV: {reg_cv["mae"].mean():.4f} (+/- {reg_cv["mae"].std():.4f})
- R2 medio CV: {reg_cv["r2"].mean():.4f} (+/- {reg_cv["r2"].std():.4f})

## Parte B - Logistica Multiclase (`ftr`)

- Accuracy media CV: {clf_cv["accuracy"].mean():.4f} (+/- {clf_cv["accuracy"].std():.4f})
- F1 macro media CV: {clf_cv["f1_macro"].mean():.4f} (+/- {clf_cv["f1_macro"].std():.4f})
- Log Loss media CV: {clf_cv["log_loss"].mean():.4f} (+/- {clf_cv["log_loss"].std():.4f})
- Accuracy media Bet365 en los mismos folds: {clf_cv["bet365_accuracy"].mean():.4f}

## Benchmark

El taller fija como referencia aproximadamente `49.8%` de accuracy para Bet365.

Comparacion directa en nuestra muestra:

- Modelo 2: {model_acc:.4f}
- Bet365: {market_acc:.4f}

## Lectura

- En esta version honesta del pipeline, el modelo **no supera** a Bet365 en accuracy multiclase.
- Eso no invalida el ejercicio: muestra que el mercado es un baseline muy fuerte y que evitar leakage vuelve el problema realmente dificil.
- La parte de regresion lineal para `total_goals` tambien queda debil, incluso usando el mejor subset honesto encontrado (`referee`), con `R2` negativo.
- El valor tecnico del proyecto esta en tener un pipeline limpio, reproducible y listo para seguir iterando sobre features realmente pre-match.

## Interpretacion metodologica

- Si usaramos estadisticas del mismo partido como tiros, corners o tarjetas, la performance subiria mucho, pero seria leakage.
- Por eso este `Modelo 2` prioriza honestidad metodologica sobre inflar metricas artificialmente.
- La lectura correcta del resultado no es “el modelo falla”, sino “el benchmark del mercado es duro y nuestras features pre-match todavia no capturan suficiente ventaja incremental”.

## Graficas

- `confusion_matrix_multiclass.png`
- `regression_residuals.png`
- `accuracy_vs_bet365.png`
"""
    return report


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df, feature_cols = load_features()
    reg_cv, clf_cv, reg_oof, clf_oof = run_cross_validation(df)
    reg_model, clf_model, reg_coef, logistic_coef = fit_final_models(df)

    reg_cv.to_csv(ARTIFACTS_DIR / "regression_cv_metrics.csv", index=False)
    clf_cv.to_csv(ARTIFACTS_DIR / "classification_cv_metrics.csv", index=False)
    reg_oof.to_csv(ARTIFACTS_DIR / "regression_oof_predictions.csv", index=False)
    clf_oof.to_csv(ARTIFACTS_DIR / "classification_oof_predictions.csv", index=False)
    reg_coef.to_csv(ARTIFACTS_DIR / "linear_coefficients.csv", index=False)
    logistic_coef.to_csv(ARTIFACTS_DIR / "multiclass_coefficients.csv", index=False)

    plot_confusion(clf_oof)
    plot_regression_residuals(reg_oof)
    plot_accuracy_benchmark(clf_cv)

    report = build_report(df, feature_cols, reg_cv, clf_cv)
    (ARTIFACTS_DIR / "report_modelo_2.md").write_text(report, encoding="utf-8")
    (ARTIFACTS_DIR / "feature_list.json").write_text(
        json.dumps(
            {
                "all_features_built": feature_cols,
                "regression_features": REGRESSION_FEATURES,
                "classification_features": CLASSIFICATION_FEATURES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    joblib.dump(reg_model, ARTIFACTS_DIR / "linear_model.joblib")
    joblib.dump(clf_model, ARTIFACTS_DIR / "multiclass_logit_model.joblib")

    print("Modelo 2 entrenado")
    print(f"Accuracy media CV: {clf_cv['accuracy'].mean():.4f}")
    print(f"Bet365 accuracy media: {clf_cv['bet365_accuracy'].mean():.4f}")
    print(f"RMSE medio CV total_goals: {reg_cv['rmse'].mean():.4f}")


if __name__ == "__main__":
    main()
