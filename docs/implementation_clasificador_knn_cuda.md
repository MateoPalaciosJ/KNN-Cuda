# Implementación de ClasificadorKNNCUDA

## 1  Objetivo

`ClasificadorKNNCUDA` proporciona la interfaz pública de alto nivel de la Fase 1

Su responsabilidad es conservar el estado del clasificador, verificar que esté ajustado y delegar el cálculo en el motor CPU de referencia

## 2  Rol temporal CPU

Durante la Fase 1, `ClasificadorKNNCUDA` utiliza las funciones de `referencia.py` basadas en NumPy

La clase no ejecuta CUDA, no contiene operaciones matemáticas propias y no afirma usar GPU

La interfaz pública se mantendrá estable cuando el backend CPU se sustituya por el backend CUDA

## 3  Constructor

El constructor recibe `numero_vecinos=5`

`numero_vecinos` debe ser un entero distinto de booleano y mayor o igual a `1`

El constructor conserva `numero_vecinos` y establece `ajustado_` en `False`

No crea datos de entrenamiento, etiquetas ni metadatos ficticios

## 4  Estado

Después de `ajustar()` existen `datos_entrenamiento_`, `etiquetas_entrenamiento_`, `numero_muestras_entrenamiento_`, `numero_caracteristicas_` y `ajustado_`

Los datos y las etiquetas almacenados conservan su contenido original y no se convierten ni modifican durante el ajuste

La clase no conserva caché, resultados intermedios ni estado de CUDA

## 5  ajustar

`ajustar(datos_entrenamiento, etiquetas_entrenamiento)` prepara el estado necesario para realizar consultas posteriores y devuelve `self`

Valida que los datos y las etiquetas sean arreglos NumPy no vacíos con dimensiones compatibles y que exista una etiqueta por muestra de entrenamiento

No calcula distancias, no selecciona vecinos, no vota y no ejecuta entrenamiento de parámetros

Las validaciones de dtype, valores finitos y otras precondiciones del cálculo permanecen en las funciones especializadas del motor CPU

## 6  predecir

`predecir(datos_consulta)` exige un clasificador ajustado y delega directamente en `predecir_knn`

Usa `numero_vecinos` almacenado y devuelve las etiquetas originales predichas

No reimplementa distancias, selección de vecinos ni votación

## 7  vecinos_mas_cercanos

`vecinos_mas_cercanos(datos_consulta, numero_vecinos=None, devolver_distancias=True)` exige un clasificador ajustado

Cuando `numero_vecinos` es `None`, utiliza el valor almacenado en la instancia

Un valor temporal válido puede sustituir el número de vecinos de una consulta sin modificar `self.numero_vecinos`

La búsqueda delega el cálculo en `distancias_l2_cuadradas` y la selección en `seleccionar_top_k`

Cuando `devolver_distancias` es verdadero, devuelve distancias euclidianas normales `float32` e índices `int64`

La raíz cuadrada se aplica solo a las distancias de los vecinos seleccionados

Cuando `devolver_distancias` es falso, devuelve únicamente los índices seleccionados

## 8  Contratos y errores

`predecir()` y `vecinos_mas_cercanos()` generan `RuntimeError` con un mensaje claro si se invocan antes de `ajustar()`

El constructor y la sobrescritura temporal de `numero_vecinos` rechazan valores no enteros, booleanos o menores que `1`

`vecinos_mas_cercanos()` rechaza un número de vecinos mayor que el número de muestras de entrenamiento

Los errores de los contratos de características y etiquetas se propagan desde el motor CPU cuando la operación correspondiente los valida

## 9  Invariantes

- `numero_vecinos` no cambia durante una consulta temporal
- Las entradas no se modifican
- El orden de los vecinos conserva el desempate por menor índice de entrenamiento
- La votación conserva el desempate por etiqueta original menor
- Las mismas entradas y configuración producen los mismos resultados
- La clase no contiene una segunda implementación de KNN

## 10  Limitaciones

- Usa CPU y NumPy durante la Fase 1
- No usa CUDA, PyTorch ni GPU
- No procesa por lotes ni bloques
- No implementa regresión, búsqueda aproximada ni ponderación por distancia
- No añade conversión automática de dtype ni escalado de características

## 11  Estrategia de pruebas

Las pruebas en `tests/cpu/test_clasificador.py` cubren el constructor, el ciclo de ajuste, el estado, las predicciones, las consultas de vecinos, los desempates, la inmutabilidad y los errores

Las pruebas de integración se ejecutan junto con `tests/cpu/test_referencia.py` para comprobar que la clase no modifica el comportamiento del motor CPU aprobado

## 12  Criterios de aceptación

- Expone `ClasificadorKNNCUDA` como API pública del paquete
- Conserva únicamente el estado necesario
- Delega cada operación de cálculo en el motor CPU existente
- Devuelve distancias euclidianas solo en la API pública de vecinos
- Mantiene el comportamiento determinista del motor de referencia
- No agrega dependencias, estado global ni lógica KNN duplicada
- Todas las pruebas CPU pasan sin modificar las pruebas del motor de referencia

## 13  Transición futura hacia CUDA

El reemplazo del backend CPU por CUDA conservará los métodos `ajustar()`, `predecir()` y `vecinos_mas_cercanos()`

La implementación futura podrá introducir operadores C++ y kernels CUDA detrás de esta interfaz sin trasladar detalles de dispositivo a la API pública
