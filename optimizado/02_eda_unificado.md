# 02. EDA Unificado

Este archivo resume los hallazgos que realmente importan. Todo lo repetido entre `EDA.md`, los EDA por dataset y reportes viejos queda absorbido aqui.

## 1. Hallazgos transversales

### Ventaja local

- el local gana cerca del 42% de los partidos
- produce mas tiros, mas tiros a puerta y mas corners
- a nivel jugador, `was_home` tambien da ventaja en puntos y produccion

Conclusion:

- la localia es una señal real y debe modelarse

### El mercado esta bien calibrado

- las probabilidades implicitas de las cuotas siguen bastante bien los resultados reales
- para Modelo 2, las cuotas son baseline de alta calidad

Conclusion:

- cualquier Modelo 2 serio debe compararse contra las cuotas

### La geometria del tiro explica gran parte del xG

- los goles se concentran en remates cerca del arco y desde zonas centrales
- la efectividad sube fuertemente dentro del area

Conclusion:

- `distance_to_goal` y `angle_to_goal` son obligatorias

### `qualifiers` es la mina de oro del Modelo 1

Las variables de contexto mas utiles del tiro son:

- `is_big_chance`
- `is_penalty`
- `is_header`
- `is_counter`
- `from_corner`
- `first_touch`
- `is_volley`
- `is_set_piece`

Conclusion:

- usar solo geometria deja informacion predictiva importante por fuera

## 2. EDA por dataset, resumida

### `matches.csv`

Lo importante:

- distribucion de resultados desbalanceada hacia victorias locales
- goles con comportamiento compatible con modelos de conteo
- `hst` es mejor predictor que `hs`
- las cuotas son muy informativas

Para modelado:

- Modelo 2 debe partir de cuotas, localia y forma

### `events.csv` / `events_rich.csv`

Lo importante:

- la mayoria de eventos son pases
- los tiros son pocos pero extremadamente informativos
- la efectividad de tiro total ronda el 11%
- los goles se agrupan en zonas cercanas al arco
- la metadata de `qualifiers` agrega contexto tactico del remate

Para modelado:

- aqui vive el Modelo 1

### `players.csv`

Lo importante:

- `minutes`, `xG`, `ict_index` y `price` separan bien jugadores relevantes
- hay buena correlacion entre `xG` y goles, pero con dispersion por finishing skill
- el dataset sirve mas para contexto de plantilla que para xG por tiro

Para modelado:

- util sobre todo para Modelo 2

### `player_history.csv`

Lo importante:

- `minutes` es critico
- `was_home` vuelve a aparecer como ventaja real
- el archivo esta truncado y no representa toda la temporada

Para modelado:

- usar con cautela, especialmente para features rolling largas

## 3. Variables mas defendibles por modelo

### Modelo 1 - xG por tiro

Base:

- `distance_to_goal`
- `angle_to_goal`

Contexto del remate:

- `is_big_chance`
- `is_penalty`
- `is_header`
- `is_counter`
- `first_touch`
- `is_volley`
- `is_set_piece`

Contexto tactico creado en el proyecto:

- `defensive_pressure`
- `buildup_passes`
- `buildup_unique_players`
- `buildup_decentralized`

### Modelo 2 - resultado de partido

- `implied_prob_h`, `implied_prob_d`, `implied_prob_a`
- localia
- forma reciente
- xG agregado por partido desde Modelo 1
- features de fuerza relativa o plantilla

## 4. Riesgos metodologicos

### Leakage

Ejemplos a vigilar:

- usar informacion post-shot en un modelo pre-shot
- usar xG del mismo partido futuro en features rolling

### Multicolinealidad

Pares problematicos:

- `distance_to_goal` con `dist_squared`
- `distance_to_goal` con `dist_angle`
- dummies completas de categorias mutuamente excluyentes

### Desbalance

- en xG, los goles son una minoria
- por eso `class_weight` y metricas como `AUC`, `Log Loss` y `Brier` son preferibles a accuracy simple

## 5. Conclusion ejecutiva del EDA

Si hubiera que resumir todo el EDA en tres frases:

1. La posicion del tiro explica mucho, pero no todo.
2. Los `qualifiers` y el contexto de la jugada son el siguiente gran salto de calidad para el xG.
3. Las cuotas y la localia son la columna vertebral del Modelo 2.
