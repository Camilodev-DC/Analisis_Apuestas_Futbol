# Cumplimiento del Modelo 1 frente al enunciado

## Veredicto general

Actualmente el `Modelo 1` **cumple la expectativa central del taller**:

- usa todos los tiros como unidad de analisis
- estima `P(gol | tiro)` con `Regresion Logistica`
- usa las variables geometricas obligatorias
- incorpora informacion contextual adicional
- reporta las metricas pedidas y compara contra un baseline naive

## 1. Objetivo del taller

El enunciado pide:

> Entrenar un modelo de regresion logistica que prediga `P(gol | tiro)`.

### Estado

`Cumple`

### Evidencia

- script oficial: [train_modelo_1.py](/home/camilo/proyectos/Ml_Futbol/modelo%201/train_modelo_1.py)
- target: `is_goal`
- salida: probabilidad `xg_logistic`

## 2. Dataset

El enunciado pide:

- todos los eventos con `is_shot = true`
- `7,198` registros

### Estado

`Cumple`

### Evidencia

- dataset base: `data/processed/features_modelo1_a_j.csv`
- filas de tiros usadas: `7,198`

## 3. Features minimas sugeridas

### 3.1 Distancia al arco

`Cumple`

Se usa como `distance_to_goal`.

### 3.2 Angulo al arco

`Cumple`

Se usa como `angle_to_goal`.

### 3.3 Head

`Cumple parcialmente como feature construida, no como variable final independiente`

`is_header` si fue creada en el feature engineering, pero no quedo en el set final del modelo.

#### Por que

- `Head`, `RightFoot` y `LeftFoot` forman un bloque casi mutuamente excluyente
- eso genera redundancia estructural
- en pruebas previas elevaban colinealidad y hacian menos parsimonioso el modelo

#### Que quedo en su lugar

La informacion del tipo de remate se absorbio parcialmente en:

- `is_big_chance`
- `first_touch`
- `defensive_pressure`
- la propia geometria del tiro

### 3.4 RightFoot / LeftFoot

`Cumple parcialmente como feature construida, no como variable final independiente`

Estas variables si existen en la tabla procesada:

- `is_right_foot`
- `is_left_foot`

Pero se descartaron del set final por la misma razon que `is_header`: redundancia y poca ganancia neta frente a un modelo mas estable.

### 3.5 BigChance

`Cumple`

Se usa como `is_big_chance` y es una de las variables mas importantes del modelo.

### 3.6 Penalty

`Cumple parcialmente como feature construida, no como variable final independiente`

`is_penalty` si fue creada en el feature engineering, pero no quedo en el set final porque:

- los penales son pocos y muy especiales
- el set final se optimizo hacia estabilidad y VIF bajo
- la señal de ocasiones claras ya quedaba parcialmente capturada por `is_big_chance`

### 3.7 Zona del disparo: BoxCentre, OutOfBoxCentre, SmallBoxCentre...

`Cumple conceptualmente, no como dummies finales literales`

#### Por que

Las zonas de disparo de ese tipo son, en esencia, una discretizacion de la posicion del tiro. En el proyecto esa informacion ya esta representada por:

- `distance_to_goal`
- `angle_to_goal`
- variables exploratorias como `is_in_area` e `is_central`

Ademas, las zonas Opta dentro de `qualifiers` son otra forma de codificar el mismo espacio que ya modelamos de forma continua con la geometria.

#### Conclusion

- no ignoramos la idea de zona
- la representamos de una forma mas estable y mas rica: geometria continua

## 4. Descubrir cuales qualifiers aportan informacion

El enunciado dice:

> Hay 110 qualifiers, ustedes deben descubrir cuales aportan informacion.

### Estado

`Cumple`

### Evidencia

En el feature engineering se construyeron, entre otras:

- `is_big_chance`
- `is_header`
- `is_right_foot`
- `is_left_foot`
- `is_counter`
- `from_corner`
- `is_penalty`
- `is_volley`
- `first_touch`
- `is_set_piece`

Y el set final del modelo retuvo las mas utiles para el objetivo probabilistico y la estabilidad:

- `is_big_chance`
- `defensive_pressure`
- `buildup_passes`
- `buildup_unique_players`
- `buildup_decentralized`
- `first_touch`

## 5. Evaluacion esperada

### Accuracy, Precision, Recall, F1

`Cumple`

Se calculan en el modelo oficial.

### Matriz de confusion

`Cumple`

Se genera como:

- `modelo 1/artifacts/confusion_matrix.png`

### Curva ROC y AUC

`Cumple`

Se genera la curva ROC y el AUC en:

- `modelo 1/graficas/01_curva_roc.png`

### Comparacion con baseline naive

`Cumple`

El baseline naive es predecir siempre `no gol`.

Se reporta su accuracy para compararlo con el modelo.

## 6. Hallazgo clave del proyecto

El hallazgo mas importante fue que:

- la version `balanced` de la logistica mantenia buen `AUC`
- pero inflaba las probabilidades y sobreestimaba goles

Por eso el modelo oficial quedo como:

- `Logistic Regression unweighted`

Esta variante:

- conserva buena discriminacion
- mejora `Log Loss`
- mejora `Brier Score`
- deja el xG medio mucho mas cerca de la tasa real de gol

## 7. Conclusion final

El `Modelo 1` ya cumple el enunciado en lo metodologicamente importante.

Lo que hicimos no fue ignorar variables del taller, sino depurarlas:

- algunas se construyeron pero no quedaron como features finales independientes
- otras quedaron absorbidas por representaciones mas robustas
- la version final privilegia interpretabilidad, calibracion y estabilidad

En otras palabras:

- `si cumple`
- `cumple mejor` cuando se defiende como modelo de probabilidad y no solo como checklist de columnas
