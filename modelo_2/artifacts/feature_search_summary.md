# Busqueda Honesta de Subsets - Modelo 2

## Mejor subset para Regresion Lineal

- Subset: `referee`
- Features: 3
- RMSE medio CV: 1.6077
- MAE medio CV: 1.2651
- R2 medio CV: -0.0335

## Mejor subset para Logistica Multiclase

- Subset: `odds`
- Features: 7
- Accuracy media CV: 0.4958
- F1 macro media CV: 0.3767
- Log Loss media CV: 1.0679
- Bet365 accuracy media CV: 0.5083

## Lectura

- La mejor regresion lineal sale de `referee`, no del bloque completo.
- La mejor clasificacion honesta sale de `odds`, lo cual refuerza que el mercado ya contiene mucha informacion pre-match.
- Agregar demasiadas variables historicas y de eventos no ayudo en esta muestra; aporta riqueza analitica, pero no necesariamente mejora la generalizacion.
