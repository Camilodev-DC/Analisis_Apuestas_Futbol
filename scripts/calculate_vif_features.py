from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


ROOT = Path(__file__).resolve().parents[1]
FEATURES_FILE = ROOT / "data" / "processed" / "features_modelo1_a_j.csv"
VIF_TABLE_FILE = ROOT / "data" / "processed" / "vif_modelo1_a_j.csv"
REPORT_FILE = ROOT / "INForems" / "vif_con_fuatures.md"


FEATURE_COLUMNS = [
    "distance_to_goal",
    "angle_to_goal",
    "dist_squared",
    "dist_angle",
    "is_in_area",
    "is_central",
    "is_big_chance",
    "is_header",
    "is_right_foot",
    "is_left_foot",
    "is_counter",
    "from_corner",
    "is_penalty",
    "is_volley",
    "first_touch",
    "is_set_piece",
    "defensive_pressure",
    "buildup_passes",
    "buildup_unique_players",
    "buildup_decentralized",
    "shot_quality_index",
    "home_xg_debt_5",
    "ppda",
    "pass_decentralization",
    "momentum",
    "home_win_rate",
    "home_bias",
    "altitude_of_play",
    "clutch_ratio",
]


def status_label(vif_value):
    if np.isinf(vif_value):
        return "Critica"
    if vif_value >= 10:
        return "Alta"
    if vif_value >= 5:
        return "Moderada"
    return "Aceptable"


def main():
    df = pd.read_csv(FEATURES_FILE)

    porteria_cols = sorted(
        col for col in df.columns
        if col.startswith("porteria_zone_") and col != "porteria_zone_unknown"
    )
    if porteria_cols:
        FEATURE_COLUMNS.extend(porteria_cols[:-1])

    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    matrix = df[available].replace([np.inf, -np.inf], np.nan).dropna().copy()

    nunique = matrix.nunique()
    usable = [col for col in matrix.columns if nunique[col] > 1]
    matrix = matrix[usable]

    matrix = add_constant(matrix, has_constant="add")
    vif_data = pd.DataFrame(
        {
            "feature": matrix.columns,
            "VIF": [
                variance_inflation_factor(matrix.values, i)
                for i in range(matrix.shape[1])
            ],
        }
    )
    vif_data = vif_data[vif_data["feature"] != "const"].copy()
    vif_data["estado"] = vif_data["VIF"].apply(status_label)
    vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)
    vif_data.to_csv(VIF_TABLE_FILE, index=False)

    top_lines = []
    for row in vif_data.head(15).itertuples():
        vif_value = "inf" if np.isinf(row.VIF) else f"{row.VIF:.2f}"
        top_lines.append(f"| `{row.feature}` | {vif_value} | {row.estado} |")
    top_table = "\n".join(top_lines)

    high_risk = vif_data[vif_data["VIF"] >= 10]["feature"].tolist()
    moderate = vif_data[(vif_data["VIF"] >= 5) & (vif_data["VIF"] < 10)]["feature"].tolist()

    report = f"""# VIF con fuatures

Este informe resume la multicolinealidad del dataset consolidado `features_modelo1_a_j.csv`, que junta las variables base del Modelo 1 con las features nuevas A-J de la hoja de ruta.

## Metodologia

- Fuente: `data/processed/features_modelo1_a_j.csv`
- Muestra usada para VIF: {len(matrix)} tiros con datos completos
- Regla: VIF < 5 aceptable, 5-10 moderado, > 10 alto
- Para `porteria_zone` se usa codificacion dummy con una categoria de referencia para evitar colinealidad perfecta

## Top 15 VIF

| Feature | VIF | Estado |
| --- | ---: | --- |
{top_table}

## Lectura rapida

- Variables con riesgo alto: {", ".join(f"`{col}`" for col in high_risk[:10]) if high_risk else "ninguna"}
- Variables con riesgo moderado: {", ".join(f"`{col}`" for col in moderate[:10]) if moderate else "ninguna"}
- Features contextuales nuevas mas estables: {", ".join(f"`{col}`" for col in vif_data[vif_data["VIF"] < 5].head(8)["feature"].tolist())}

## Recomendaciones

- Mantener las features contextuales con VIF bajo como candidatas fuertes para el modelo final
- Revisar si `dist_squared`, `dist_angle` o variables muy derivadas deben convivir con `distance_to_goal`
- Si se usa un modelo lineal, considerar eliminar o regularizar las features con VIF alto
- Si se usa arboles o boosting, el riesgo principal es interpretabilidad, no necesariamente rendimiento
"""

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"VIF guardado en: {VIF_TABLE_FILE}")
    print(f"Informe guardado en: {REPORT_FILE}")


if __name__ == "__main__":
    main()
