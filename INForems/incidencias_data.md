# Informe de Incidencias: Integridad y Mapeo de Datos

Este documento detalla los problemas encontrados durante la validación de los datasets descargados y las soluciones implementadas para asegurar la viabilidad del modelo de predicción.

---

## 1. Incidencias Identificadas

### 1.1 Inconsistencia de Identificadores (IDs) — CRÍTICO
Se detectó que el archivo `events.csv` ( WhoScored/Opta) y los archivos `players.csv`/`player_history.csv` (Fantasy Premier League) utilizan sistemas de identificación diferentes para los mismos jugadores.
- **Impacto:** Imposibilidad de unir estadísticas detalladas de eventos con el rendimiento jornada a jornada de FPL mediante un simple `join` de IDs.
- **Ejemplo:** Erling Haaland tiene ID `315227` en eventos y `430` en FPL.

### 1.2 Discrepancia con Reportes Previos
El archivo `events.csv` actual es más reciente y completo que el analizado en el `diccionario_datos.md` inicial.
- **Datos previos:** 398,961 eventos / 714 goles.
- **Datos reales actuales:** 444,252 eventos / 807 goles.
- **Resolución:** Se confirmó que el archivo actual es el correcto ya que coincide con los goles registrados en `matches.csv` para los 291 partidos.

### 1.3 Nombres no Estandarizados
Existen variaciones en el registro de nombres (ej: `Gabriel Magalhães` en eventos vs `Gabriel` en FPL) y en nombres de equipos (ej: `Nott'm Forest` vs `Nottingham Forest`).

---

## 2. Solución Implementada: Script de Mapeo

Para resolver la falta de una llave foránea (FK) común, se ha desarrollado un script de mapeo inteligente.

### 2.1 Estrategia de Mapeo (`scripts/map_players.py`)
El script genera un archivo de traducción `player_id_map.json` siguiendo esta lógica:
1. **Normalización:** Homogeneización de nombres de equipos (ej: Tottenham/Spurs) y eliminación de acentos básicos.
2. **Coincidencia Exacta:** Intento de unir por `Nombre Completo + Equipo`.
3. **Coincidencia por Nombre Corto:** Intento de unir por `Nombre Web (FPL) + Equipo`.
4. **Coincidencia Parcial:** Búsqueda de subcadenas (ej: "Semenyo" dentro de "Antoine Semenyo").

### 2.2 Resultados del Mapeo
De los **518 jugadores específicos** encontrados en el archivo de eventos:
- **490 jugadores** (94.6%) fueron mapeados correctamente con su ID de FPL.
- **28 jugadores** no fueron enlazados (principalmente canteranos con pocos minutos o bajas de invierno).

---

## 3. Acciones Recomendadas

1. **Uso del Mapa:** En cualquier análisis que requiera cruzar eventos con historia de puntos, cargar `/data/processed/player_id_map.json` primero.
2. **Actualización de EDA:** Los informes de `INForems/` deben ser revisados para reflejar los nuevos totales (807 goles).
3. **Mantenimiento:** Volver a ejecutar `python3 scripts/map_players.py` si se descargan nuevos datos de partidos.

---
**Estado final:** Dataset validado y puente de unión entre fuentes establecido en `/data/processed/player_id_map.json`.
