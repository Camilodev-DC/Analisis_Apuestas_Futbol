# Guía para Clona el Proyecto (Colaboradores)

Si acabas de clonar este repositorio, verás que la carpeta `data/` está vacía. Esto es para evitar subir archivos pesados (200MB+) a GitHub. Sigue estos pasos para comenzar:

---

## 1. Configuración del Entorno
Crea y activa un entorno virtual de Python e instala las dependencias:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Obtención de Datos
Necesitas el dataset (`events.csv`, `players.csv`, etc.). Tienes dos opciones:
*   **B. Descargarlos (Recomendado):** Ejecuta el script maestro:
    ```bash
    python3 scripts/download_bulk_data.py
    ```
    Este script descarga todo (players, matches, history) y reconstruye los +444k eventos con **qualifiers** incluidos.

## 3. Mapeo de Identificadores
El archivo `data/processed/player_id_map.json` ya está incluido en el repositorio. Es vital para unir las fuentes de FPL y WhoScored. Si descargas nuevos datos, asegúrate de actualizarlo:
```bash
python3 scripts/map_players.py
```

## 4. Estructura de Trabajo
- `src/`: Lógica central (un solo lugar para el procesamiento).
- `scripts/`: Pruebas rápidas y experimentación. No subir scripts de "usar y tirar".
- `INForems/`: Reportes de integridad y validación (¡Léelos antes de empezar!).

---

**Nota:** No olvides configurar tu usuario de git para que tus commits tengan tu nombre:
`git config user.name "Tu Nombre"`
`git config user.email "tu@correo.com"`
