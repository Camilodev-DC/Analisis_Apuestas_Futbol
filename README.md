# Machine Learning I: Modelo de Apuestas Premier League (Taller 2)

## Descripción
Este proyecto implementa una arquitectura modular para predecir resultados de la Premier League utilizando modelos de Regresión Lineal y Logística. Forma parte del **Taller 2 del curso de Machine Learning I** (2026I) en la Universidad Externado de Colombia.

El objetivo es utilizar modelos de regresión para predecir:
1. **Regresión Lineal:** Goles anotados por el equipo local (`fthg`).
2. **Regresión Logística:** Resultado final (`ftr`: Home, Draw, Away).

## Configuración del Entorno

Sigue estos pasos para configurar tu entorno de desarrollo:

1. **Crear Entorno Virtual:**
   ```bash
   python3 -m venv .venv
   ```

2. **Activar Entorno Virtual:**
   - Linux/macOS: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`

3. **Instalar Dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Variables de Envorno:**
   Configura tu `API_BASE_URL` en el archivo `scripts/config.py` o mediante variables de entorno si están habilitadas.

## Estructura
```text
.
├── analysis/           # Reportes y visualizaciones locales
├── artifacts/          # Documentos complementarios
├── data/               # Directorio de datos (ignorado por Git en su mayoría)
│   ├── raw/            # Datos originales (CSV/JSON)
│   └── processed/      # Datos transformados y player_id_map.json
├── INForems/           # Reportes de auditoría, EDA y Diccionarios
├── pdf/                # Guías oficiales del taller
├── scripts/            # Scripts de descarga, modelado y utilidad
├── src/                # Lógica central modular
│   ├── data_loader/    # Extracción de datos desde APIs
│   ├── processing/     # Limpieza de datos y feature engineering
│   ├── models/         # Entrenamiento, validación y predicción
│   └── utils/          # Funciones auxiliares
├── tests/              # Pruebas unitarias
├── main.py             # Pipeline principal de ejecución
├── requirements.txt    # Librerías necesarias
└── Taller_2_Premier_League.ipynb  # Cuaderno principal de experimentación
```

## Guía de Ejecución

1. **Descargar Datos:**
   ```bash
   python3 scripts/download_bulk_data.py
   ```

2. **Entrenamiento de Modelos:**
   Puedes usar los scripts en la carpeta `scripts/` o el cuaderno `Taller_2_Premier_League.ipynb` para ver el flujo completo de experimentación.

## Novedades del Proyecto (Abril 2026)
- **Integración con GitHub:** Sincronización con el repositorio central de `Analisis_Apuestas_Futbol`.
- **Análisis de Roles:** Implementación de Clustering (K-Means) para identificar perfiles de jugadores.
- **Visualización PCA:** Mapas de rendimiento de jugadores per 90 minutos.

---
**Curso:** Machine Learning I (ML1-2026I)  
**Institución:** Universidad Externado de Colombia  
**Docente:** Julián Zuluaga
