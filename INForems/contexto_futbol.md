# **FUNCIONAL**
# Contexto de Fútbol para Ciencia de Datos

Este documento explica los conceptos clave del fútbol que necesitas comprender para trabajar con los datos de la Premier League.

---

## 1. La Premier League

La **Premier League** es la primera división del fútbol inglés y una de las ligas más competitivas del mundo. Participan **20 equipos** que juegan todos contra todos (ida y vuelta), dando un total de **380 partidos** por temporada (38 jornadas × 10 partidos cada una).

### Sistema de Puntos
- **Victoria:** 3 puntos
- **Empate:** 1 punto
- **Derrota:** 0 puntos

Los últimos 3 equipos descienden a la Championship (segunda división).

---

## 2. Conceptos Clave del Partido

### Localía (Home/Away)
- **Home (H):** El equipo que juega en su propio estadio. Estadísticamente, el local gana ~42% de los partidos (ventaja de localía).
- **Away (A):** El equipo visitante. Gana ~32% de los partidos.
- **Draw (D):** Empate. Ocurre ~26% de las veces, y es el resultado **más difícil de predecir**.

### Medio Tiempo vs. Tiempo Completo
- **HT (Half Time):** El partido se divide en dos tiempos de 45 minutos. El resultado al medio tiempo puede ser diferente al final.
- **FT (Full Time):** Resultado después de los 90 minutos reglamentarios.

---

## 3. Estadísticas de Juego

### Tiros (Shots)
- **Shots (`hs`/`as_`):** Todos los intentos de gol, incluidos los que van desviados.
- **Shots on Target (`hst`/`ast`):** Solo los tiros que iban a portería (el portero los detuvo o fue gol). Es un mejor indicador de peligro que los "tiros totales".
- **Efectividad:** En la Premier League, aproximadamente **11% de los tiros terminan en gol**.

### Córners (`hc`/`ac`)
Un **córner** se da cuando el balón sale por la línea de fondo y fue tocado por última vez por el equipo defensor. Es una oportunidad de jugada a balón parado. Más córners suele indicar dominio territorial.

### Faltas (`hf`/`af`)
Las **faltas** pueden indicar un equipo agresivo o que el rival es efectivo en el regate. Más faltas del visitante es típico (presión por recuperar balón).

### Tarjetas
- **Amarilla (`hy`/`ay`):** Advertencia. Dos amarillas = expulsión.
- **Roja (`hr`/`ar`):** Expulsión directa. El equipo queda con un jugador menos.

---

## 4. Apuestas y Cuotas (Odds)

### ¿Qué es una cuota?
Las casas de apuestas (como **Bet365**) asignan un número a cada resultado posible. Ejemplo: `b365h = 1.85` significa que si apuestas $1 al equipo local y gana, recibes $1.85.

### Probabilidad Implícita
La cuota se transforma en probabilidad:
```
P = 1 / cuota
```
Ejemplo: `b365h = 1.85` → `P(Home) = 1/1.85 = 54.1%`

### El Margen de la Casa (Overround)
Si sumas las tres probabilidades (H+D+A), obtienes más de 100%. Esa diferencia es la ganancia de la casa de apuestas. Para eliminar ese sesgo, se **normalizan** las probabilidades.

### ¿Por qué importa para ML?
Las cuotas de apuestas son **uno de los mejores predictores existentes** de resultados de fútbol. Son el "baseline" contra el cual debes comparar tu modelo. Si tu modelo no supera las predicciones implícitas de Bet365, no agrega valor.

---

## 5. Tipos de Eventos (Events Data)

### Pases (Pass) — 62.8% de todos los eventos
El tipo de evento más común. Un equipo promedio da ~500 pases por partido. La precisión de pases en la Premier es ~77.8%.

### Duelos Aéreos (Aerial) — 4.2%
Cuando dos jugadores disputan el balón en el aire (normalmente en un centro o un saque largo).

### Regates (TakeOn) — 2.3%
Cuando un jugador intenta superar a un defensor con el balón. Su éxito/fracaso indica habilidad individual.

### Tackle vs. Interception
- **Tackle:** Quitarle el balón al atacante con contacto.
- **Interception:** Cortar un pase sin necesidad de tacklear.

### Clearance — 3.7%
Despejar el balón de una zona peligrosa, normalmente de cabeza o de patada larga. No busca precisión, busca alejar el peligro.

### Dispossessed vs. BallRecovery
- **Dispossessed:** Perder el balón ante la presión del rival.
- **BallRecovery:** Recuperar el balón después de que estaba suelto.

---

## 6. Sistema de Coordenadas Opta

Los datos de `events.csv` usan el **sistema Opta**, donde el campo se normaliza a un rectángulo de 100×100:

```
        0 ────────── x ──────────► 100
        │                           │
   0    │  TU PORTERÍA    PORTERÍA  │
   │    │     ⬛               ⬛    │
   y    │        CAMPO DE JUEGO     │
   │    │     (100 x 100)          │
   ▼    │                           │
  100   │                           │
```

- **x = 0:** Tu propia línea de fondo (defensa).
- **x = 100:** La línea de gol rival (ataque).
- **y = 0:** Banda izquierda.
- **y = 100:** Banda derecha.

Esto permite crear **mapas de tiros**, **mapas de calor** y análisis de zonas de pase.

---

## 7. Insights Clave para tu Modelo

1. **Ventaja local:** ~42.3% de los partidos los gana el equipo de casa.
2. **Empates difíciles:** 26.1% son empates, pero son los más difíciles de clasificar.
3. **Tiros a puerta > Tiros totales:** `hst` es mejor predictor que `hs` para goles.
4. **Cuotas como baseline:** Las probabilidades implícitas de Bet365 son tu punto de comparación obligatorio.
5. **SOTDiff:** La diferencia entre tiros a puerta (`hst - ast`) es una feature poderosa para regresión logística.
