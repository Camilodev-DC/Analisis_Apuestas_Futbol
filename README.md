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

## Uso de Scripts
- **Producción:** Ejecuta `python3 main.py` para correr el pipeline completo.
- **Experimentación:** Coloca tus scripts de prueba en `scripts/` para mantener la limpieza.
- **Tests:** Ejecuta `pytest` para correr las pruebas unitarias.

---
**Senior Data Architect & DevOps Approach**
