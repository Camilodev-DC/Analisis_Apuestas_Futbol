# 04. Mapa de Fuentes

Este archivo sirve para dejar claro que no se borro informacion: se reorganizo.

## Archivos originales absorbidos aqui

### Contexto y datos

- `INForems/contexto_futbol.md` -> `01_contexto_y_datos.md`
- `INForems/diccionario_datos.md` -> `01_contexto_y_datos.md`
- `INForems/incidencias_data.md` -> `01_contexto_y_datos.md`
- `INForems/guia_colaboracion.md` -> `01_contexto_y_datos.md`

### EDA

- `Research/EDA/EDA.md` -> `02_eda_unificado.md`
- `Research/EDA/events/EDA_events.md` -> `02_eda_unificado.md`
- `Research/EDA/matches/EDA_matches.md` -> `02_eda_unificado.md`
- `Research/EDA/players/EDA_players.md` -> `02_eda_unificado.md`
- `Research/EDA/player_history/EDA_player_history.md` -> `02_eda_unificado.md`
- `INForems/primer_informe.md` -> absorbido parcialmente en `02_eda_unificado.md`
- `INForems/exploracion_avanzada.md` -> absorbido y reemplazado conceptualmente por `02_eda_unificado.md`

### Modelado

- `Research/EDA/hoja_de_ruta_modelamiento.md` -> `03_modelado_unificado.md`
- `Research/MODEL_1/MODEL_1.MD` -> `03_modelado_unificado.md`
- `Research/MODEL_2/MODEL_2.md` -> `03_modelado_unificado.md`
- `INForems/vif_con_fuatures.md` -> referenciado desde `03_modelado_unificado.md`

## Archivos que se conservan como soporte tecnico

Estos no los reemplaza `optimizado`; siguen siendo utiles como evidencia o ejecucion:

- `scripts/feature_engineering.py`
- `scripts/calculate_vif_features.py`
- `modelo 1/train_modelo_1.py`
- `modelo 1/artifacts/`
- `Research/EDA/scripts_EDA/`

## Politica sugerida desde ahora

- leer y mantener `optimizado/` como la capa narrativa oficial
- dejar `Research/` y `INForems/` como respaldo historico o tecnico
- si se actualiza un insight clave, actualizar primero `optimizado/`
