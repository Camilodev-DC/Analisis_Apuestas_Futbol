# Informe de Multicolinealidad (VIF) — Modelo Expected Goals (xG)

El objetivo de este informe es analizar el **Factor de Inflación de la Varianza (VIF)** entre las variables escogidas para el modelo de predicción de goles (Expected Goals), simulado mediante Regresión Lineal.

El VIF mide cuánto se infla la varianza de un coeficiente de regresión estimado debido a la multicolinealidad con otras predictores.
*   **VIF = 1:** Sin correlación (independiente).
*   **VIF < 5:** Multicolinealidad moderada (Aceptable).
*   **VIF > 5 a 10:** Alta multicolinealidad (Se recomienda remover redundancias).

---

## 📊 Resultados del Análisis VIF 

Los resultados calculados en la muestra en vivo de la temporada actual arrojaron la siguiente distribución de VIF por característica:

| Feature | VIF Score | Estado / Acción |
| :--- | :--- | :--- |
| **is_right_foot** | 48.99 | 🚨 Colinealidad Crítica |
| **is_left_foot** | 42.70 | 🚨 Colinealidad Crítica |
| **is_header** | 30.24 | 🚨 Colinealidad Crítica |
| **distance_to_goal** | 28.46 | ⚠️ Colinealidad Fuerte |
| **dist_angle** | 13.50 | ⚠️ Colinealidad Fuerte |
| **dist_squared** | 11.78 | ⚠️ Colinealidad Fuerte |
| **angle_to_goal**| 6.14 | 🟡 Moderada/Fuerte |
| **is_in_area** | 3.29 | ✅ Aceptable |
| **is_central** | 2.45 | ✅ Aceptable |
| **first_touch** | 1.74 | ✅ Bien Independiente |
| **is_big_chance** | 1.68 | 🟢 **Excelente predictor**|
| **from_corner** | 1.31 | ✅ Bien Independiente |
| **is_volley** | 1.21 | ✅ Bien Independiente |
| **is_penalty**| 1.17 | ✅ Bien Independiente |
| **is_set_piece** | 1.12 | ✅ Bien Independiente |
| **is_counter** | 1.06 | 🟢 **Total Independencia**|

---

## 🧠 Conclusiones y Correcciones Matemáticas

### 1. El Problema de las Partes del Cuerpo (Colinealidad Estructural)
Las variables `is_right_foot`, `is_left_foot` y `is_header` tienen un VIF extremadamente alto (entre 30 y 49). 
*   **Matemáticamente:** Esto sucede por la **"Trampa de la Variable Ficticia" (Dummy Variable Trap)**. Prácticamente el 99% de los tiros se hacen con alguna de estas 3 partes. Al tener las tres en el modelo, actúan como un proyector perfecto, generando redundancia matemática. 
*   **Solución:** Se debe usar una como categoría base (ej. eliminar `is_right_foot` del modelo).

### 2. Redundancia en Variables Geométricas 
Las características `distance_to_goal`, `dist_squared` y `dist_angle` muestran VIF muy por encima del límite aceptable (11 a 28).
*   **Matemáticamente:** Al derivar al cuadrado la distancia, o multiplicarla con el ángulo, estamos proveyendo información sumamente correlacionada entre sí. En modelos de bosque aleatorio (Random Forest o XGBoost), combinar o hacer _engineering_ no hace daño, pero en una **Regresión Lineal**, causa inestabilidad en los coeficientes (podemos obtener pesos negativos no interpretables).
*   **Solución:** Escoger **solo `distance_to_goal` y `angle_to_goal`** (VIF de 6, más aceptable).

### 3. Los Contextos Valen Oro (`is_big_chance`, `is_counter`, `from_corner`)
Todos los clasificadores tácticos tienen VIFs maravillosos (cercanos a 1.0).
*   **En Contexto:** Nos dice que estas features aportan información única e irrepetible. Un contraataque (`is_counter`, VIF 1.06) es un evento sumamente ortogonal al resto de variables. 
*   **La Joya de la Corona:** `is_big_chance` tiene un VIF mínimo (1.68). Al ser la variable más predecible del xG por sí misma, ver que no está fuertemente correlacionada con las construcciones geométricas es positivo; significa que agrega verdadero contexto al modelo.

---
### 🛠️ Configuración Sugerida para el Modelo Lineal Final:
Se aconseja usar solo este set de variables depurado:
`[distance_to_goal, angle_to_goal, is_in_area, is_central, is_big_chance, is_left_foot, is_header, is_penalty, first_touch, is_counter, from_corner, is_volley]`
