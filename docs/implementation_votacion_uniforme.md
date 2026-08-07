# Implementación de votacion_uniforme

## 1  Objetivo

`votacion_uniforme` es la operación CPU de referencia que recibe las etiquetas de los `k` vecinos ya seleccionados y produce una predicción por consulta mediante votación uniforme

La función trabaja con las etiquetas originales del problema, que pueden ser enteras negativas, no consecutivas o de valores grandes

## 2  Contrato

### Entrada

- `etiquetas_vecinos` como `numpy.ndarray`
- Forma `[Q, K]`
- Arreglo bidimensional y no vacío
- Tipo entero de NumPy distinto de booleano
- Etiquetas originales sin necesidad de ser consecutivas

### Salida

- `predicciones` como `numpy.ndarray`
- Forma `[Q]`
- Tipo entero igual al de `etiquetas_vecinos`

## 3  Responsabilidad

- Contar los votos de cada etiqueta por consulta
- Elegir la etiqueta con mayor número de votos
- Resolver empates mediante la etiqueta original numéricamente menor
- Producir una predicción por fila

## 4  Qué no debe hacer

- Calcular distancias
- Ordenar vecinos
- Conocer `datos_consulta`
- Conocer `datos_entrenamiento`
- Acceder a índices de vecinos
- Aplicar ponderación por distancia
- Clasificar mediante distancias
- Modificar `etiquetas_vecinos`
- Mantener estado externo

## 5  Regla de votación

Cada vecino aporta exactamente un voto

Para `etiquetas_vecinos = [5, 2, 5]`, la predicción es `5`

Para `etiquetas_vecinos = [5, 2]`, la predicción es `2` porque ambas etiquetas tienen un voto y gana la etiqueta original menor

## 6  Algoritmo CPU

La función procesa cada fila de `etiquetas_vecinos` con operaciones de NumPy para obtener las etiquetas distintas y sus conteos

Se identifica el conteo máximo y se toma la etiqueta menor entre las que tienen ese conteo. Este paso hace explícita la regla de desempate y no depende de un orden interno incidental

No se usan scipy, scikit-learn, collections.Counter ni una estrategia CUDA

## 7  Determinismo

- Las mismas etiquetas producen exactamente las mismas predicciones
- Los empates de votos siempre se resuelven por la etiqueta original menor
- El resultado no depende del orden interno de una estructura de conteo

## 8  Complejidad

Para una fila con `K` vecinos, obtener etiquetas y conteos tiene complejidad `O(K log K)`

Para `Q` consultas, la complejidad temporal es `O(Q × K log K)`

La memoria temporal por fila es `O(K)` y la salida requiere `O(Q)`

## 9  Invariantes

- Existe una predicción por consulta
- `etiquetas_vecinos` no se modifica
- Cada predicción pertenece a las etiquetas observadas en su fila
- Las predicciones conservan el dtype de entrada
- Los empates se resuelven de forma determinista
- Las mismas entradas producen las mismas salidas

## 10  Validaciones

La función valida directamente:

- Que `etiquetas_vecinos` sea `numpy.ndarray`
- Que `etiquetas_vecinos` sea bidimensional
- Que `etiquetas_vecinos` no esté vacío
- Que `etiquetas_vecinos.dtype` sea entero
- Que `etiquetas_vecinos.dtype` no sea booleano

Los errores de tipo usan `TypeError` y los errores de forma o tamaño usan `ValueError`. Los mensajes identifican la entrada inválida

## 11  Casos límite

- Una consulta
- Varias consultas
- Una sola etiqueta por fila
- Una única clase
- Etiquetas negativas
- Etiquetas no consecutivas
- Etiquetas enteras grandes
- Empate entre dos clases
- Empate entre más de dos clases

## 12  Estrategia de pruebas

Las pruebas en `tests/cpu/test_reference.py` verifican forma, dtype, clasificación binaria y multiclase, una sola clase, `k = 1`, etiquetas negativas, etiquetas no consecutivas, etiquetas grandes, empates, determinismo, inmutabilidad y validaciones

Los resultados esperados se definen explícitamente sin reutilizar `votacion_uniforme`

## 13  Criterios de aceptación

La implementación se considera aprobada cuando:

- Respeta el contrato de entrada y salida
- Conserva el dtype entero de las etiquetas
- No modifica las entradas
- Cuenta un voto por vecino
- Resuelve todos los empates por la etiqueta menor
- No calcula distancias ni conoce índices
- Todas las pruebas pasan
- No agrega dependencias ni abstracciones innecesarias

No quedan decisiones pendientes sobre la validación, los conteos, el desempate ni las salidas de `votacion_uniforme`
