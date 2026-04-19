# VIF con fuatures

Este informe resume la multicolinealidad del dataset consolidado `features_modelo1_a_j.csv`, que junta las variables base del Modelo 1 con las features nuevas A-J de la hoja de ruta.

## Metodologia

- Fuente: `data/processed/features_modelo1_a_j.csv`
- Muestra usada para VIF: 5725 tiros con datos completos
- Regla: VIF < 5 aceptable, 5-10 moderado, > 10 alto
- Para `porteria_zone` se usa codificacion dummy con una categoria de referencia para evitar colinealidad perfecta

## Top 15 VIF

| Feature | VIF | Estado |
| --- | ---: | --- |
| `distance_to_goal` | inf | Critica |
| `is_in_area` | inf | Critica |
| `is_counter` | inf | Critica |
| `is_big_chance` | inf | Critica |
| `is_central` | inf | Critica |
| `shot_quality_index` | inf | Critica |
| `porteria_zone_center_low` | 82.25 | Alta |
| `is_right_foot` | 55.57 | Alta |
| `is_left_foot` | 48.56 | Alta |
| `porteria_zone_center_high` | 39.20 | Alta |
| `porteria_zone_center_mid` | 35.84 | Alta |
| `is_header` | 33.22 | Alta |
| `dist_angle` | 13.63 | Alta |
| `dist_squared` | 12.47 | Alta |
| `porteria_zone_right_low` | 12.17 | Alta |

## Lectura rapida

- Variables con riesgo alto: `distance_to_goal`, `is_in_area`, `is_counter`, `is_big_chance`, `is_central`, `shot_quality_index`, `porteria_zone_center_low`, `is_right_foot`, `is_left_foot`, `porteria_zone_center_high`
- Variables con riesgo moderado: `angle_to_goal`
- Features contextuales nuevas mas estables: `buildup_unique_players`, `buildup_passes`, `porteria_zone_left_mid`, `buildup_decentralized`, `first_touch`, `porteria_zone_right_high`, `porteria_zone_left_high`, `from_corner`

## Recomendaciones

- Mantener las features contextuales con VIF bajo como candidatas fuertes para el modelo final
- Revisar si `dist_squared`, `dist_angle` o variables muy derivadas deben convivir con `distance_to_goal`
- Si se usa un modelo lineal, considerar eliminar o regularizar las features con VIF alto
- Si se usa arboles o boosting, el riesgo principal es interpretabilidad, no necesariamente rendimiento
