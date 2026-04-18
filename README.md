# Proyecto: Predicción de Resultados Premier League 2025-26

Este proyecto forma parte del Taller 2 del curso de **Machine Learning I**. El objetivo es utilizar modelos de regresión lineal y logística para predecir el desempeño y los resultados de los partidos de la Premier League utilizando datos en tiempo real de una API personalizada.

## Estructura del Proyecto

- `GUIA_ASISTENTE.md`: Documentación interna con detalles técnicos de la API y modelos.
- `requirements.txt`: Dependencias de Python necesarias.
- `Taller_2_Premier_League.ipynb`: Cuaderno principal de experimentación y modelado.
- `data/`: Directorio reservado para históricos de extracción (ignorado por git).

## Instalación

Para preparar el entorno local:

```bash
# Crear un entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Modelos a Implementar

1. **Regresión Lineal:** Predicción de goles anotados por el equipo local (`fthg`).
2. **Regresión Logística:** Clasificación multiclase del resultado final (`ftr`: Home, Draw, Away).

---
**Curso:** Machine Learning I (ML1-2026I)  
**Institución:** Universidad Externado de Colombia  
**Docente:** Julián Zuluaga
