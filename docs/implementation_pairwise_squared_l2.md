# Implementación de pairwise_squared_l2

## 1  Objetivo

`pairwise_squared_l2` será la implementación CPU de referencia para calcular distancias euclidianas cuadradas entre un lote de consultas y el conjunto de entrenamiento

Su prioridad será la corrección, el determinismo, la claridad y la facilidad de validación. No busca un rendimiento competitivo ni debe incorporar decisiones de optimización que dificulten la revisión de la lógica

La función formará parte de la fuente primaria de verdad del motor CPU de referencia y servirá para validar las implementaciones posteriores en CUDA

## 2  Contrato

### Entradas

- `X_query` como `numpy.ndarray`
- Forma `[Q, D]`
- Tipo `float32`
- `X_train` como `numpy.ndarray`
- Forma `[N, D]`
- Tipo `float32`
- Ambos arreglos bidimensionales
- Ambos arreglos no vacíos
- Misma cantidad de características
- Sin valores `NaN`
- Sin valores infinitos

### Salida

- `numpy.ndarray`
- Forma `[Q, N]`
- Tipo `float32`
- Cada posición `[q, n]` contiene la distancia euclidiana cuadrada entre `X_query[q]` y `X_train[n]`

### Responsabilidad

Calcular únicamente la matriz de distancias euclidianas cuadradas

### No debe

- Ordenar vecinos
- Seleccionar Top-K
- Aplicar raíz cuadrada
- Conocer etiquetas
- Realizar votación
- Modificar `X_query`
- Modificar `X_train`
- Depender de estado externo

La función debe recibir todos los datos necesarios mediante sus entradas y producir la salida exclusivamente a partir de ellos

## 3  Fórmula matemática

Para una consulta `x` y una muestra de entrenamiento `y`, la distancia euclidiana cuadrada se define como:

`d²(x, y) = suma desde j = 0 hasta D - 1 de (x_j - y_j)²`

La raíz cuadrada no se aplica porque no es necesaria para ordenar vecinos. Si una distancia cuadrada es menor que otra, su distancia euclidiana normal también será menor

La función devuelve distancias cuadradas, no distancias euclidianas normales. Esta decisión coincide con el contrato del motor CPU de referencia y evita realizar una transformación que no participa en esta etapa

## 4  Algoritmo elegido para la referencia CPU

La primera implementación debe priorizar claridad y trazabilidad

La operación calculará las diferencias entre cada consulta y cada muestra de entrenamiento usando NumPy. Debe utilizar una estrategia vectorizada simple y comprensible, con una correspondencia directa entre la operación matemática y las dimensiones de los arreglos

No debe introducir técnicas de optimización complejas ni sustituir la definición matemática por una formulación cuya revisión sea más difícil

La implementación no debe utilizar bibliotecas distintas de NumPy. En particular, no debe usar:

- scipy
- scikit-learn
- PyTorch
- numba
- multiprocessing
- Procesamiento por bloques

Tampoco debe utilizar expresiones algebraicas alternativas como:

`||x||² + ||y||² - 2x·y`

en esta primera referencia

La razón es mantener una correspondencia directa con la definición matemática de la distancia y facilitar la comparación con cálculos manuales

## 5  Estrategia concreta de cálculo

El flujo conceptual será:

1. Recibir `X_query` y `X_train` ya validados
2. Expandir las dimensiones de forma que NumPy pueda comparar cada consulta contra cada muestra de entrenamiento mediante broadcasting
3. Obtener las diferencias por característica
4. Elevar cada diferencia al cuadrado
5. Sumar sobre la dimensión de características
6. Devolver una matriz `[Q, N]` en `float32`

El documento especifica el comportamiento y la secuencia lógica, pero no incluye código fuente

La función no debe ordenar las muestras ni realizar ninguna operación propia de Top-K o de clasificación después de obtener la suma sobre características

## 6  Broadcasting esperado

Las formas intermedias se interpretarán conceptualmente de la siguiente manera:

`X_query`

`[Q, D]`

Se interpreta para broadcasting como:

`[Q, 1, D]`

`X_train`

`[N, D]`

Se interpreta para broadcasting como:

`[1, N, D]`

La resta produce:

`[Q, N, D]`

La suma sobre `D` produce:

`[Q, N]`

Las dimensiones representan:

- `Q`: número de consultas
- `N`: número de muestras de entrenamiento
- `D`: número de características

La dimensión `Q` conserva el orden de `X_query`, la dimensión `N` conserva el orden de `X_train` y la dimensión `D` contiene las diferencias por característica antes de la suma

## 7  Tipos y precisión

- Las entradas deben ser `float32`
- La salida debe permanecer en `float32`
- La función no convierte silenciosamente `float64` a `float32`
- Una entrada con tipo incorrecto debe producir un error claro
- No debe existir promoción intencional a `float64`

Mantener `float32` es importante porque la implementación CUDA futura utilizará el mismo tipo base. La referencia debe detectar un tipo distinto antes del cálculo, en lugar de ocultar la diferencia mediante una conversión implícita

La precisión se evaluará con los criterios definidos para `float32`. Una pequeña diferencia numérica debe analizarse como parte de las pruebas y no corregirse mediante conversiones silenciosas

## 8  Complejidad esperada

### Tiempo

`O(Q × N × D)`

### Memoria de salida

`O(Q × N)`

### Memoria temporal de la estrategia vectorizada

`O(Q × N × D)`

La memoria temporal de la estrategia vectorizada es aceptable en la referencia CPU inicial porque esta fase prioriza claridad y trazabilidad. Esta decisión queda limitada a la referencia y no define el diseño de memoria del motor CUDA

La implementación CUDA final no tendrá esta misma estrategia de memoria. El documento de diseño del motor CUDA establece un procesamiento por lotes y bloques para evitar la necesidad de materializar la matriz completa de distancias

## 9  Invariantes

La implementación debe mantener estos invariantes:

- `X_query` no se modifica
- `X_train` no se modifica
- La salida siempre es bidimensional
- La salida tiene forma `[Q, N]`
- La salida tiene tipo `float32`
- Todas las distancias son mayores o iguales a cero, salvo pequeñas diferencias numéricas inesperadas que deberán investigarse
- Si una consulta es exactamente igual a una muestra de entrenamiento, su distancia debe ser cero
- Las mismas entradas producen la misma salida
- El orden de filas corresponde al orden original de `X_query`
- El orden de columnas corresponde al orden original de `X_train`

Estos invariantes deben comprobarse mediante pruebas o inspección de resultados, no asumirse únicamente por la forma de la operación

## 10  Validaciones

`pairwise_squared_l2` debe validar directamente:

- Que `X_query` sea `numpy.ndarray`
- Que `X_train` sea `numpy.ndarray`
- Que `X_query.ndim == 2`
- Que `X_train.ndim == 2`
- Que `X_query` no esté vacío
- Que `X_train` no esté vacío
- Que `X_query.dtype == float32`
- Que `X_train.dtype == float32`
- Que `X_query.shape[1] == X_train.shape[1]`
- Que no existan valores `NaN`
- Que no existan valores infinitos

Los errores deben indicar claramente qué condición no se cumplió y, cuando sea útil, identificar la entrada afectada

No se agregará un sistema complejo de excepciones personalizadas. Se utilizarán las excepciones estándar de Python cuando corresponda, con mensajes suficientemente claros para identificar el problema

La validación pertenece a esta función porque forma parte de su contrato público. No debe trasladarse a `knn_predict` dejando a `pairwise_squared_l2` sin protección cuando se invoque directamente

## 11  Casos límite

El comportamiento esperado debe estar definido para:

- `Q = 1`
- `N = 1`
- `D = 1`
- Una consulta idéntica a una muestra de entrenamiento
- Todas las muestras idénticas
- Valores negativos
- Valores cero
- Valores muy pequeños representables en `float32`
- Valores grandes pero finitos en `float32`

Los valores negativos deben tratarse mediante la diferencia y su cuadrado, sin asumir que las características son no negativas. Los valores cero deben producir distancia cero cuando las muestras comparadas sean iguales

Un overflow producido por valores extremos no debe ocultarse silenciosamente. Si aparece durante las pruebas, debe detectarse y analizarse para determinar si se trata de una limitación esperada de `float32` o de un error de implementación

## 12  Estrategia de pruebas

Las pruebas vivirán en:

`tests/cpu/test_reference.py`

Como mínimo, deben existir las siguientes pruebas

### Forma y tipo

- Verificar salida `[Q, N]`
- Verificar `dtype` `float32`

### Cálculos manuales

- Un caso de una dimensión calculado manualmente
- Un caso de dos dimensiones calculado manualmente
- Varias consultas contra varias muestras

### Distancia cero

- Consulta idéntica a una muestra

### Duplicados

- Muestras de entrenamiento duplicadas producen distancias iguales

### Valores negativos

- Confirmar que las diferencias al cuadrado se calculan correctamente

### Inmutabilidad

- Verificar que `X_query` no cambia
- Verificar que `X_train` no cambia

### Determinismo

- Ejecutar dos veces y comprobar igualdad

### Errores

- `X_query` no es `numpy.ndarray`
- `X_train` no es `numpy.ndarray`
- Entrada unidimensional
- Entrada tridimensional
- Arreglo vacío
- `float64`
- Entero
- Diferente número de características
- `NaN`
- Infinito

Las pruebas deben comprobar valores, forma, tipo de salida e inmutabilidad. Cuando corresponda, también deben verificar que el mensaje de error identifique la condición inválida

## 13  Comparación manual

Al menos una prueba debe comparar la salida contra valores calculados explícitamente a mano

El resultado esperado no debe construirse con la misma expresión vectorizada de la función. Debe escribirse de forma independiente para evitar que un mismo error aparezca tanto en la implementación como en la prueba

La comparación manual debe cubrir al menos un caso pequeño en el que sea posible revisar cada distancia sin depender de otra biblioteca de cálculo

## 14  Criterios de aceptación

La implementación futura de `pairwise_squared_l2` se considerará aprobada cuando:

- Respete exactamente su contrato
- Todas las pruebas pasen
- No modifique las entradas
- Devuelva siempre `float32`
- Produzca la forma correcta
- Coincida con los casos manuales
- Sea determinista
- No contenga lógica de Top-K
- No contenga lógica de votación
- No introduzca dependencias nuevas
- No tenga lógica duplicada
- Sea fácil de leer y depurar
- Su funcionamiento pueda explicarse directamente a partir de la fórmula matemática

La revisión debe confirmar además que cada validación pertenece a esta función, que no existe una ruta alternativa innecesaria y que la implementación no oculta la operación central mediante abstracciones sin justificación

## 15  Qué no debe hacerse

- No optimizar prematuramente
- No crear clases
- No crear funciones públicas adicionales
- No mezclar validación con otras responsabilidades ajenas al contrato
- No introducir procesamiento por bloques
- No usar CUDA
- No usar PyTorch
- No usar scikit-learn para el cálculo
- No usar scipy
- No usar numba
- No usar paralelismo
- No introducir caché
- No cambiar el contrato definido en `reference_engine.md`

El documento deja definido el comportamiento de `pairwise_squared_l2` sin decisiones pendientes sobre sus entradas, su salida, su cálculo, sus validaciones o sus pruebas

La implementación debe permanecer como una referencia CPU directa y comprensible para las fases posteriores


## 16 Compatibilidad futura

Esta implementación define el comportamiento oficial de pairwise_squared_l2

Cualquier implementación p  osterior, incluyendo CUDA, deberá producir exactamente el mismo resultado dentro de las tolerancias numéricas establecidas

Las optimizaciones futuras no podrán modificar el contrato, la forma de la salida ni las reglas definidas en este documento