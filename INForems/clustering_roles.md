# **FALTA INFORMACIÓN**
# Visualización Avanzada de Roles (PCA & K-Means)

Una vez entrenado el algoritmo **K-Means con K=5**, hemos utilizado Análisis de Componentes Principales (PCA) para reducir nuestras estadísticas a 2 dimensiones, permitiéndonos visualizar qué tan bien separados están los roles.

## 1. El Mapa de los 'Nuevos Roles'

A continuación vemos cómo la Inteligencia Artificial agrupó a los jugadores basándose **únicamente en su rendimiento per 90 minutos**.

![Clústeres PCA](/home/camilo/proyectos/Ml_Futbol/analysis/plots/pca_clusters.png)

## 2. El Mapa de las Posiciones Clásicas FPL

Si miramos la *misma nube de puntos* pero coloreada según su posición oficial en la Fantasy Premier League (Portero, Defensa, Medio, Delantero), notamos cómo el modelo descubrió sub-roles (por ejemplo, medios defensivos separados de los creativos).

![Posiciones PCA](/home/camilo/proyectos/Ml_Futbol/analysis/plots/pca_positions.png)

## Tablas de Perfiles por Clúster

### Rol 0 (79 jugadores)
**Ejemplos destacados:** Gana, J.Gomes, Potts, Ampadu, André

| Métrica (per 90) | Valor Promedio |
| ---------------- | -------------- |
| goals_scored | 0.03 |
| assists | 0.05 |
| clean_sheets | 0.19 |
| influence | 18.04 |
| creativity | 7.08 |
| threat | 5.40 |
| bps | 11.99 |
| xG | 0.05 |
| xA | 0.04 |

### Rol 1 (44 jugadores)
**Ejemplos destacados:** Haaland, Kroupi.Jr, João Pedro, Richarlison, Palmer

| Métrica (per 90) | Valor Promedio |
| ---------------- | -------------- |
| goals_scored | 0.39 |
| assists | 0.13 |
| clean_sheets | 0.32 |
| influence | 21.35 |
| creativity | 14.41 |
| threat | 27.56 |
| bps | 18.28 |
| xG | 0.37 |
| xA | 0.08 |

### Rol 2 (24 jugadores)
**Ejemplos destacados:** Cherki, B.Fernandes, Bruno G., Foden, Rice

| Métrica (per 90) | Valor Promedio |
| ---------------- | -------------- |
| goals_scored | 0.20 |
| assists | 0.30 |
| clean_sheets | 0.35 |
| influence | 22.82 |
| creativity | 31.80 |
| threat | 21.34 |
| bps | 21.29 |
| xG | 0.21 |
| xA | 0.23 |

### Rol 3 (75 jugadores)
**Ejemplos destacados:** James, Garner, Janelt, L.Miley, Matheus N.

| Métrica (per 90) | Valor Promedio |
| ---------------- | -------------- |
| goals_scored | 0.09 |
| assists | 0.17 |
| clean_sheets | 0.30 |
| influence | 18.01 |
| creativity | 18.36 |
| threat | 10.68 |
| bps | 16.91 |
| xG | 0.10 |
| xA | 0.11 |

### Rol 4 (51 jugadores)
**Ejemplos destacados:** Gabriel, Calafiori, J.Palhinha, Gvardiol, Caicedo

| Métrica (per 90) | Valor Promedio |
| ---------------- | -------------- |
| goals_scored | 0.06 |
| assists | 0.05 |
| clean_sheets | 0.35 |
| influence | 22.85 |
| creativity | 5.55 |
| threat | 5.90 |
| bps | 16.91 |
| xG | 0.05 |
| xA | 0.04 |

