# Optimizado

Esta carpeta es la version compacta del proyecto. Su objetivo es reemplazar la lectura dispersa de `INForems/`, `Research/EDA/` y notas parciales con una ruta corta y consistente.

## Estructura

- `01_contexto_y_datos.md`: contexto de futbol, datasets, diccionario minimo, incidencias y reglas para trabajar con los datos.
- `02_eda_unificado.md`: hallazgos de EDA consolidados, sin repetir el mismo insight en cuatro archivos distintos.
- `03_modelado_unificado.md`: pipeline, features, decisiones de modelado y estado real de Modelo 1 / Modelo 2.
- `04_mapa_fuentes.md`: traduccion entre archivos originales y su equivalente consolidado en esta carpeta.

## Orden recomendado de lectura

1. `01_contexto_y_datos.md`
2. `02_eda_unificado.md`
3. `03_modelado_unificado.md`

## Principios de esta version

- No repetir el mismo insight en multiples archivos.
- Separar claramente contexto, EDA y modelado.
- Conservar solo lo accionable para construir y defender el proyecto.
- Marcar explicitamente limitaciones y riesgos reales del dataset.
- Escribir con estructura reutilizable para una futura pagina web del proyecto.

## Estado del proyecto al consolidar

- Datos raw disponibles: `matches.csv`, `players.csv`, `player_history.csv`, `events.csv`, `events_rich.csv`
- Datos procesados clave: `features_modelo1_a_j.csv`, `vif_modelo1_a_j.csv`, `player_id_map.json`
- Modelo 1 implementado: `modelo 1/train_modelo_1.py`
- Pipeline de features implementado: `scripts/feature_engineering.py`
- Decision vigente para Modelo 1: usar logistica `unweighted` como `xG` final, no la variante `balanced`
