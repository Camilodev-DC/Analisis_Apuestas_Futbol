# Machine Learning I: Modelo de Apuestas Premier League

## Descripción
Este proyecto implementa una arquitectura modular para predecir resultados de la Premier League utilizando modelos de Regresión Lineal y Logística. Sigue estándares profesionales de arquitectura de datos y DevOps.

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

4. **Variables de Entorno:**
   Crea un archivo `.env` basado en el ejemplo (si existe) y añade tu `API_BASE_URL`.

## Estructura
```text
.
├── data/               # Directorio de datos
│   ├── raw/            # Datos originales (CSV/JSON) - Ignorados por Git
│   └── processed/      # Datos transformados y player_id_map.json
├── INForems/           # Reportes de auditoría, EDA y Diccionarios
├── scripts/            # Scripts de descarga y mapeo de IDs
├── src/                # Lógica central modular
│   ├── data_loader/    # Extracción de datos desde APIs
│   ├── processing/     # Limpieza de datos y feature engineering
│   ├── models/         # Entrenamiento, validación y predicción
│   └── utils/          # Funciones auxiliares
├── tests/              # Pruebas unitarias
├── main.py             # Pipeline principal de ejecución
└── requirements.txt    # Librerías necesarias
```

## Colaboración
Si eres un nuevo colaborador, consulta la [Guía de Colaboración](INForems/guia_colaboracion.md) para configurar tu entorno y obtener los datasets pesados.

## Guía Rápida de Datos y EDA

Dado que los archivos `.csv` están en `.gitignore` por su peso, un nuevo colaborador debe seguir este orden:

1. **Descargar Datos:**
   ```bash
   python3 scripts/download_bulk_data.py
   ```
   *Nota: Descargará ~444k eventos con qualifiers (tarda ~4 min).*

2. **Generar Reportes EDA:**
   ```bash
   python3 Research/EDA/scripts_EDA/run_all_eda.py
   ```
   *Esto actualizará todas las gráficas en `Research/EDA/*/graficas/`.*

## Novedades del Proyecto (Marzo 2026)
- **Qualifiers en Events**: Ahora el dataset de eventos incluye la metadata completa (BigChance, Shot Type, Contact, etc.).
- **Features Taller2**: Implementación de `distance_to_goal` y `angle_to_goal` en los reportes de eventos.
- **Diccionario Maestro**: Localizado en `INForems/diccionario_datos.md` con el detalle de las +100 categorías de qualifiers.

---
**Senior Data Architect & DevOps Approach**
