# EDA — `events.csv`

**Registros:** 444,252 eventos | **Columnas:** 20 | **Fuente:** Premier League API (WhoScored)

---

## 1. Calidad de Datos

### Valores Nulos
![Nulos](graficas/01_nulos.png)

Los nulos están en campos estructurales (`player_name`, `player_id`) o específicos de acción (`end_x`, `goal_mouth_z`). Los campos críticos de posición y tipo están completos.

---

## 2. Distribución de Eventos
![Tipos de Evento](graficas/02_tipos_evento.png)

El **Pass** domina el dataset (~60%), seguido de recuperaciones y despejes. Contamos con una base sólida de eventos defensivos y ofensivos.

---

## 3. Outcome y Desbalance
![Outcome](graficas/03_outcome.png)

La tasa de éxito es del **~78%**. El desbalance es natural en el fútbol. Los modelos deben considerar que el fracaso de una acción es el evento minoritario.

---

## 4. Análisis Temporal (Minutos)
![Eventos por Minuto](graficas/04_eventos_minuto.png)

Se observa un flujo constante con picos lógicos en el minuto 45 y 90 (tiempo de adición). La intensidad se mantiene alta durante todo el encuentro.

---

## 5. Mapas de Calor (Densidad)
![Heatmap Todos](graficas/05_heatmap_todos.png)

La actividad se concentra en el carril central y zonas de transición. El juego por bandas es secundario pero relevante para centros.

---

## 6. Análisis de Tiros y Goles
![Heatmap Tiros](graficas/06_heatmap_tiros.png)

Los tiros se concentran en el área grande, pero los goles (**puntos cyan**) se agrupan en la zona de mayor peligro frente al arco.

---

## 7. Efectividad por Zona del Campo
![Efectividad Zona](graficas/07_efectividad_zona.png)

| Zona | % Efectividad |
|---|---|
| Media/Baja (x<33) | <1% |
| Área (>85) | **~18%** |

La efectividad se multiplica por 10 al entrar en el área rival.

---

## 8. Features Geométricas (Taller2)
![Distancia y Ángulo](graficas/08_distancia_angulo_xg.png)

Siguiendo el **Taller2 ML1**, hemos derivado `distance_to_goal` y `angle_to_goal`. Los goles (dorado) tienen una distribución claramente sesgada hacia distancias cortas y ángulos más amplios.

---

## 9. Análisis de Qualifiers (La Mina de Oro)
![Efectividad Qualifiers](graficas/09_efectividad_qualifiers.png)

Al parsear el JSON de `qualifiers`, extraemos features críticas:
- **Penalti**: ~76% de éxito.
- **Big Chance**: ~38% de éxito (feature estrella).
- **Cabeza**: Notablemente menos efectiva que tiros con el pie.

---

## 10. Mapa de Tiros por Tipo de Contacto
![Mapa Tiros Tipo](graficas/10_mapa_tiros_tipo.png)

Visualizamos la especialización: los cabezazos ocurren en el centro del área, mientras que los remates de pie derecho/izquierdo tienen mayor rango. Los **⭐ dorados** marcan los goles reales.

---

## Resumen para Modelado

| Métrica | Valor |
|---|---|
| Total Tiros | ~6,400 |
| Total Goles | ~714 |
| Efectividad Global | 11.2% |
| **AUC-ROC Objetivo** | **> 0.78** |

*Documento consolidado tras actualización de pipeline de descarga (21/03/2026).*
