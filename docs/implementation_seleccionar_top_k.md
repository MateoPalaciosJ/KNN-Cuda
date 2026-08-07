# Implementación de seleccionar_top_k

## 1  Objetivo

`seleccionar_top_k` será la operación CPU de referencia para seleccionar los `k` vecinos de menor distancia para cada consulta

La función prioriza corrección, determinismo, claridad y facilidad de prueba. No busca rendimiento competitivo ni introduce estrategias de selección parcial u optimizaciones de memoria

## 2  Contrato

### Entradas

- `distancias` como `numpy.ndarray`
- Forma `[Q, N]`
- Tipo `float32`
- Arreglo bidimensional y no vacío
- Sin valores `NaN` ni infinitos
- `k` como entero mayor o igual a `1` y menor o igual a `N`

### Salidas

- `distancias_seleccionadas` como `numpy.ndarray` con forma `[Q, K]` y tipo `float32`
- `indices_seleccionados` como `numpy.ndarray` con forma `[Q, K]` y tipo `int64`

## 3  Responsabilidad

La función ordena cada fila de `distancias`, selecciona los `k` valores menores y devuelve las distancias seleccionadas junto con sus índices originales

El resultado es determinista y conserva la correspondencia entre cada distancia y su índice de entrenamiento

## 4  Qué no debe hacer

- Calcular distancias nuevas
- Conocer etiquetas
- Realizar votación
- Clasificar
- Aplicar raíz cuadrada
- Modificar la matriz de entrada
- Mantener estado externo
- Usar procesamiento por bloques

## 5  Algoritmo elegido

La función usa `np.argsort` estable sobre cada fila de `distancias`. El orden estable conserva el orden original de los índices cuando dos distancias son iguales

Después de ordenar los índices se seleccionan las primeras `k` posiciones de cada fila y se recuperan las distancias correspondientes

## 6  Justificación

`np.argsort` estable expresa directamente el contrato del módulo de referencia: primero se ordena por distancia y, en caso de empate, se conserva el menor índice original

No se utiliza `np.argpartition` porque esta fase prioriza un orden final completo, fácil de inspeccionar y determinista

## 7  Determinismo

Para cada fila se aplican estas reglas:

- La menor distancia aparece primero
- Las distancias iguales se ordenan por el menor índice de entrenamiento
- Las mismas entradas y el mismo valor de `k` producen exactamente las mismas salidas

El orden estable de `np.argsort` garantiza el desempate porque los índices originales de cada fila ya están en orden ascendente antes de ordenar por distancia

## 8  Complejidad temporal

La ordenación de una fila de `N` distancias tiene complejidad `O(N log N)`

Para `Q` consultas, la complejidad temporal total es `O(Q × N log N)`

## 9  Complejidad espacial

La matriz de índices ordenados requiere `O(Q × N)` espacio temporal

Las salidas requieren `O(Q × K)` espacio para las distancias seleccionadas y los índices seleccionados

## 10  Invariantes

- `distancias` no se modifica
- Las salidas son bidimensionales
- Las salidas tienen forma `[Q, K]`
- `distancias_seleccionadas` tiene tipo `float32`
- `indices_seleccionados` tiene tipo `int64`
- Cada distancia seleccionada corresponde al índice seleccionado en la misma posición
- Cada fila de distancias seleccionadas está en orden ascendente
- Los empates de distancia conservan el menor índice primero

## 11  Validaciones

La función valida directamente:

- Que `distancias` sea `numpy.ndarray`
- Que `distancias` sea bidimensional
- Que `distancias` no esté vacía
- Que `distancias.dtype` sea `float32`
- Que `distancias` no contenga `NaN` ni valores infinitos
- Que `k` sea entero y no sea booleano
- Que `k` sea mayor o igual a `1`
- Que `k` no sea mayor que el número de columnas de `distancias`

Los errores de tipo usan `TypeError` y los errores de valores o dimensiones usan `ValueError`. Cada mensaje identifica la entrada o condición inválida

## 12  Casos límite

- `Q = 1`
- `N = 1`
- `k = 1`
- `k = N`
- Varias consultas
- Distancias repetidas
- Empates completos en una fila
- Distancias ya ordenadas
- Distancias en orden descendente

Los empates deben resolverse siempre por el menor índice original, incluso cuando todas las distancias de una fila son iguales

## 13  Estrategia de pruebas

Las pruebas vivirán en `tests/cpu/test_reference.py` y comprobarán de forma independiente:

- Forma y tipo de ambas salidas
- Casos `k = 1` y `k = N`
- Una consulta, varias consultas y una única muestra de entrenamiento
- Orden ascendente de distancias
- Índices correctos y correspondencia índice distancia
- Empates, distancias repetidas y desempate por índice menor
- Determinismo e inmutabilidad
- Errores de tipo, dimensiones, valores no finitos y valores inválidos de `k`

Los resultados esperados se escribirán de forma explícita sin reutilizar `seleccionar_top_k`

## 14  Criterios de aceptación

La implementación se considera aprobada cuando:

- Respeta el contrato de entrada y salida
- Solo realiza selección y ordenación de vecinos
- Usa `np.argsort` estable
- Mantiene `float32` para distancias e `int64` para índices
- No modifica `distancias`
- Resuelve empates por índice menor
- Todas las pruebas existentes y nuevas pasan
- No agrega dependencias ni funciones auxiliares innecesarias
- Puede revisarse directamente a partir de las reglas de orden definidas

No quedan decisiones pendientes sobre la validación, el orden, el desempate, las salidas ni los casos límite de `seleccionar_top_k`
