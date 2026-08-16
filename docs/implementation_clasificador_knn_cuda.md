# Implementación de ClasificadorKNNCUDA

## 1  Objetivo

`ClasificadorKNNCUDA` proporciona la interfaz pública de alto nivel del proyecto

Su responsabilidad es conservar el estado del clasificador, verificar que esté ajustado y delegar el cálculo en el backend nativo mediante PyTorch Dispatcher

## 2  Integración nativa actual

Durante la Fase 1, `ClasificadorKNNCUDA` utilizó las funciones de `referencia.py` basadas en NumPy como backend operativo

Ese diseño se conserva como contexto histórico, pero el clasificador actual utiliza los operadores nativos CPU o CUDA registrados bajo los mismos esquemas del Dispatcher

NumPy continúa como referencia funcional primaria y no funciona como fallback operativo oculto

## 3  Constructor

El constructor recibe `numero_vecinos=5` y `dispositivo="cpu"`

`numero_vecinos` debe ser un entero distinto de booleano y mayor o igual a `1`

`dispositivo` acepta `"cpu"`, `"cuda"` y `"auto"`

El constructor conserva ambos valores y establece `ajustado_` en `False`

No crea datos de entrenamiento, etiquetas ni metadatos ficticios

## 4  Estado

Después de `ajustar()` existen `datos_entrenamiento_`, `etiquetas_entrenamiento_`, `datos_entrenamiento_tensor_`, `etiquetas_entrenamiento_tensor_`, `numero_muestras_entrenamiento_`, `numero_caracteristicas_`, `dispositivo_efectivo_` y `ajustado_`

Los datos y las etiquetas públicas se conservan como copias NumPy independientes y los tensores internos se preparan una sola vez en el dispositivo efectivo

La clase no conserva resultados intermedios ni duplica la lógica KNN

## 5  ajustar

`ajustar(datos_entrenamiento, etiquetas_entrenamiento)` prepara el estado necesario para realizar consultas posteriores y devuelve `self`

Valida que los datos y las etiquetas sean arreglos NumPy no vacíos con dimensiones y dtype compatibles, características finitas y una etiqueta entera por muestra de entrenamiento

No calcula distancias, no selecciona vecinos, no vota y no ejecuta entrenamiento de parámetros

Resuelve `"cpu"`, `"cuda"` o `"auto"`, comprueba los kernels requeridos y establece `dispositivo_efectivo_` solo después de completar el ajuste correctamente

## 6  predecir

`predecir(datos_consulta)` exige un clasificador ajustado, valida la consulta NumPy y delega directamente en `torch.ops.knn_cuda.predecir_knn`

Usa `numero_vecinos` almacenado y devuelve un `numpy.ndarray` con las etiquetas originales predichas

No reimplementa distancias, selección de vecinos ni votación

## 7  vecinos_mas_cercanos

`vecinos_mas_cercanos(datos_consulta, numero_vecinos=None, devolver_distancias=True)` exige un clasificador ajustado

Cuando `numero_vecinos` es `None`, utiliza el valor almacenado en la instancia

Un valor temporal válido puede sustituir el número de vecinos de una consulta sin modificar `self.numero_vecinos`

La búsqueda delega el cálculo en `torch.ops.knn_cuda.distancias_l2_cuadradas` y la selección en `torch.ops.knn_cuda.seleccionar_top_k` para el dispositivo efectivo

Cuando `devolver_distancias` es verdadero, devuelve distancias euclidianas normales `float32` e índices `int64`

La raíz cuadrada se aplica solo a las distancias de los vecinos seleccionados

Cuando `devolver_distancias` es falso, devuelve únicamente los índices seleccionados

## 8  Contratos y errores

`predecir()` y `vecinos_mas_cercanos()` generan `RuntimeError` con un mensaje claro si se invocan antes de `ajustar()`

El constructor y la sobrescritura temporal de `numero_vecinos` rechazan valores no enteros, booleanos o menores que `1`

`vecinos_mas_cercanos()` rechaza un número de vecinos mayor que el número de muestras de entrenamiento

Las entradas públicas deben ser `numpy.ndarray`, las características deben usar `float32` y las etiquetas deben usar dtype entero distinto de booleano

La solicitud explícita de CUDA falla claramente si el runtime o los kernels CUDA no están disponibles, mientras `"auto"` utiliza CPU cuando no puede utilizar CUDA

## 9  Invariantes

- `numero_vecinos` no cambia durante una consulta temporal
- Las entradas no se modifican
- El orden de los vecinos conserva el desempate por menor índice de entrenamiento
- La votación conserva el desempate por etiqueta original menor
- Las mismas entradas y configuración producen los mismos resultados
- La clase no contiene una segunda implementación de KNN

## 10  Limitaciones

- La API pública acepta únicamente `numpy.ndarray`
- No acepta `torch.Tensor` como entrada pública
- No divide automáticamente los datos en lotes ni bloques de memoria
- No implementa regresión, búsqueda aproximada ni ponderación por distancia
- No añade conversión automática de dtype ni escalado de características
- No implementa selección explícita de una GPU ni ejecución multi-GPU

## 11  Estrategia de pruebas

Las pruebas en `tests/cpu/test_clasificador.py` cubren el constructor, el ciclo de ajuste, el estado, las predicciones, las consultas de vecinos, los desempates, la inmutabilidad y los errores

Las pruebas de integración CPU se ejecutan junto con `tests/cpu/test_referencia.py` y las pruebas CUDA viven en `tests/cuda/test_clasificador_cuda.py`

Ambas rutas comparan resultados contra la referencia NumPy y conservan los contratos públicos

## 12  Criterios de aceptación

- Expone `ClasificadorKNNCUDA` como API pública del paquete
- Conserva únicamente el estado necesario
- Delega cada operación de cálculo en el backend nativo mediante el Dispatcher
- Devuelve distancias euclidianas solo en la API pública de vecinos
- Mantiene el comportamiento determinista del motor de referencia
- No agrega dependencias, estado global ni lógica KNN duplicada
- Conserva entradas y salidas públicas `numpy.ndarray`
- Permite `"cpu"`, `"cuda"` y `"auto"` sin fallback oculto cuando CUDA se solicita explícitamente

## 13  Estado de integración CPU y CUDA

`ClasificadorKNNCUDA` está integrado con las implementaciones CPU C++ y CUDA mediante PyTorch Dispatcher

Los métodos `ajustar()`, `predecir()` y `vecinos_mas_cercanos()` conservan una única API pública y seleccionan el backend mediante `dispositivo` sin trasladar detalles de los kernels al usuario
