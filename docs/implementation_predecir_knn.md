# Implementación de predecir_knn

## 1  Objetivo

`predecir_knn` coordina el pipeline CPU completo de clasificación KNN exacta y devuelve una predicción original por consulta

La función no implementa cálculos especializados propios. Reutiliza las operaciones de distancias, selección y votación ya definidas por el motor CPU de referencia

## 2  Contrato

### Entradas

- `datos_entrenamiento` como `numpy.ndarray` con forma `[N, D]` y dtype `float32`
- `etiquetas_entrenamiento` como `numpy.ndarray` con forma `[N]` y dtype entero distinto de booleano
- `datos_consulta` como `numpy.ndarray` con forma `[Q, D]` y dtype `float32`
- `k` como entero válido entre `1` y `N`

### Salida

- `predicciones` como `numpy.ndarray`
- Forma `[Q]`
- Etiquetas originales con un dtype entero compatible con `etiquetas_entrenamiento`

## 3  Responsabilidad

La función coordina este flujo:

```text
datos_consulta + datos_entrenamiento
                ↓
      distancias_l2_cuadradas
                ↓
        seleccionar_top_k
                ↓
      indices_seleccionados
                ↓
etiquetas_entrenamiento[indices_seleccionados]
                ↓
        votacion_uniforme
                ↓
          predicciones
```

## 4  Qué no debe hacer

- Reimplementar el cálculo de distancias
- Reimplementar la selección Top-K
- Reimplementar la votación
- Crear estado o caché
- Aplicar raíz cuadrada
- Usar CUDA o PyTorch
- Procesar por bloques
- Usar búsqueda aproximada
- Modificar cualquiera de las entradas

## 5  Validaciones específicas

`predecir_knn` valida las condiciones exclusivas del pipeline:

- Que `etiquetas_entrenamiento` sea `numpy.ndarray`
- Que `etiquetas_entrenamiento` sea unidimensional y no esté vacía
- Que `etiquetas_entrenamiento.dtype` sea entero y no booleano
- Que la cantidad de etiquetas coincida con el número de filas de `datos_entrenamiento`

Las funciones especializadas validan sus propios contratos de datos, distancias y valor de `k`. Esta separación evita repetir validaciones que ya pertenecen a otra responsabilidad

## 6  Determinismo

El pipeline conserva las reglas de determinismo de cada operación:

- Las distancias se calculan de forma determinista
- Los empates de distancia se resuelven por el menor índice de entrenamiento
- Los empates de votos se resuelven por la etiqueta original menor
- Las mismas entradas producen exactamente las mismas predicciones

## 7  Complejidad

El cálculo de distancias requiere `O(Q × N × D)` tiempo y `O(Q × N × D)` memoria temporal en la referencia CPU

La selección requiere `O(Q × N log N)` tiempo y `O(Q × N)` memoria temporal

La votación requiere `O(Q × K log K)` tiempo y `O(K)` memoria temporal por consulta

La complejidad dominante depende del tamaño relativo de `D` y `log N`, sin optimizaciones de memoria ni procesamiento por bloques

## 8  Limitaciones

- Es una referencia CPU orientada a corrección
- No optimiza memoria
- No procesa por bloques
- No usa CUDA
- No usa PyTorch
- No usa búsqueda aproximada
- No usa ponderación por distancia

## 9  Estrategia de pruebas

Las pruebas de integración en `tests/cpu/test_reference.py` cubren `k = 1`, `k = N`, clasificación binaria y multiclase, una sola clase, consultas únicas y múltiples, etiquetas no consecutivas y negativas, empates de distancia, empates de voto, duplicados, casos manuales, varias características, determinismo, inmutabilidad y propagación de errores

Una comparación secundaria con scikit-learn podrá añadirse en una fase posterior sin convertirla en una dependencia obligatoria. La fuente de verdad actual son los casos manuales y las operaciones CPU aprobadas

## 10  Criterios de aceptación

La implementación se considera aprobada cuando:

- Reutiliza `distancias_l2_cuadradas`, `seleccionar_top_k` y `votacion_uniforme`
- No duplica la lógica de esas operaciones
- Devuelve una predicción original por consulta
- Mantiene las entradas inmutables
- Conserva los desempates definidos por el proyecto
- Propaga los errores de contratos especializados cuando corresponde
- Todas las pruebas existentes y nuevas pasan
- No agrega dependencias, estado ni funcionalidad fuera del alcance

No quedan decisiones pendientes sobre el flujo, las validaciones específicas, los desempates ni las limitaciones de `predecir_knn`
