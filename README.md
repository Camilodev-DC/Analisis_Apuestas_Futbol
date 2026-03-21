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
├── data/               # Datos (ignorado por Git)
│   ├── raw/            # Datos originales de API
│   └── processed/      # Datos transformados
├── scripts/            # Scripts de experimentación y EDA
├── src/                # Lógica central modular
│   ├── data_loader/    # Extracción de datos
│   ├── processing/     # Limpieza y features
│   ├── models/         # Entrenamiento y evaluación
│   └── utils/          # Utilidades comunes
├── tests/              # Pruebas con pytest
├── main.py             # Punto de entrada principal
└── requirements.txt    # Dependencias
```

## Uso de Scripts
- **Producción:** Ejecuta `python3 main.py` para correr el pipeline completo.
- **Experimentación:** Coloca tus scripts de prueba en `scripts/` para mantener la limpieza.
- **Tests:** Ejecuta `pytest` para correr las pruebas unitarias.

---
**Senior Data Architect & DevOps Approach**
