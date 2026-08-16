# Diseño de software de KNN-Cuda

Este documento define el diseño técnico actual de KNN-Cuda

Complementa la arquitectura y los requisitos del proyecto

Su principio central es demostrar primero la corrección del software y después optimizar el rendimiento

## 1. Producto y API pública

El paquete Python se llama `knn_cuda`. La clase pública principal es `ClasificadorKNNCUDA`, orientada a clasificación KNN exacta

La clase proporciona las operaciones públicas `__init__()`, `ajustar()`, `vecinos_mas_cercanos()` y `predecir()`

La API pública vigente es `ClasificadorKNNCUDA(numero_vecinos=5, dispositivo="cpu")`

- `numero_vecinos`: número de vecinos que se utilizarán

`dispositivo` acepta `"cpu"`, `"cuda"` y `"auto"`, sin introducir parámetros públicos de lotes o bloques

El comportamiento de la API es el siguiente:

- `ajustar()` recibe el conjunto de entrenamiento y sus etiquetas, prepara el estado del clasificador y devuelve `self` para permitir encadenamiento
- `vecinos_mas_cercanos()` recibe consultas y devuelve distancias e índices de los vecinos, respetando el orden determinista definido en este documento
- `predecir()` busca los vecinos, aplica votación uniforme y devuelve las etiquetas en su representación original

El estado del clasificador, incluyendo el conjunto de entrenamiento, las etiquetas originales y los metadatos, permanece gestionado por Python. La API debe ocultar al usuario la coordinación de la extensión C++ y de los kernels CUDA

## 1.1 Política de dispositivo de Fase 4

El valor predeterminado `"cpu"` preserva un comportamiento reproducible entre máquinas y no activa aceleración implícitamente

- `"cpu"` utiliza explícitamente el backend C++ CPU y exige que el Dispatcher tenga los kernels CPU requeridos
- `"cuda"` exige `torch.cuda.is_available()` y los kernels CUDA requeridos. No degrada silenciosamente a CPU
- `"auto"` utiliza CUDA solo si el runtime y los kernels CUDA requeridos están disponibles. En cualquier otro caso utiliza CPU

`dispositivo_efectivo_` es el único indicador visible del dispositivo seleccionado y se establece durante un `ajustar()` exitoso como `torch.device("cpu")` o el dispositivo CUDA actual

La integración no expone tensores PyTorch en la API pública ni añade métodos públicos solo para consultar el backend

## 2. Integración tecnológica

La estrategia nativa oficial utiliza la infraestructura de extensiones de PyTorch y operadores registrados mediante su dispatcher

La Fase 2 implementó operadores CPU mediante `CppExtension`

La Fase 3 añadió implementaciones CUDA mediante `CUDAExtension` sin reemplazar la ruta de integración ni modificar la API pública

No se utilizará pybind11 como estrategia independiente ni existirá un segundo sistema de binding

La división de responsabilidades es:

- Python conserva el estado, valida las entradas públicas y coordina las llamadas
- C++ valida las precondiciones críticas, recibe tensores y ejecuta operadores CPU
- CUDA ejecuta operaciones sin estado sobre tensores CUDA

Los operadores C++/CUDA no conservan estado entre invocaciones ni conocen la instancia de `ClasificadorKNNCUDA`. Reciben tensores con las etiquetas originales cuando la operación lo requiere y no necesitan metadatos externos de clases

La compilación oficial es anticipada, como parte del flujo normal de preparación del paquete o del entorno de ejecución. La compilación JIT no forma parte del mecanismo oficial de distribución ni de ejecución estable

## 3. Contratos de datos

Los tensores de características respetan estos contratos:

- `datos_entrenamiento` tiene forma `[N, D]`, donde `N` es el número de muestras y `D` el número de características
- `datos_consulta` tiene forma `[Q, D]`, donde `Q` es el número de consultas
- `datos_entrenamiento` y `datos_consulta` usan `float32`
- Las etiquetas usan un dtype entero compatible y conservan su representación original
- Los índices de vecinos usan `int64`
- Los operadores nativos aceptan vistas no contiguas válidas y crean una representación contigua interna solo cuando es necesaria
- Los tensores que participan en una operación nativa comparten un mismo dispositivo, CPU o CUDA según el Dispatcher

La API pública vigente recibe únicamente `numpy.ndarray` y rechaza dtype incompatibles sin convertirlos silenciosamente

La integración CPU convierte explícitamente las entradas NumPy válidas a tensores PyTorch CPU y convierte las salidas a `numpy.ndarray` para conservar el contrato público

`vecinos_mas_cercanos()` devuelve distancias `float32` e índices `int64` y `predecir()` devuelve las etiquetas originales

## 4. Etiquetas

La primera versión acepta etiquetas enteras originales sin necesidad de que sean consecutivas

Las etiquetas pueden ser negativas o tener valores grandes y se entregan directamente a `votacion_uniforme` tanto en CPU como en CUDA

El flujo de etiquetas es:

1. `seleccionar_top_k` obtiene `indices_seleccionados`
2. La capa que coordina el pipeline recupera `etiquetas_vecinos` desde `etiquetas_entrenamiento`
3. `votacion_uniforme(etiquetas_vecinos)` recibe las etiquetas originales y devuelve predicciones originales

Las etiquetas de texto quedan fuera del alcance inicial

No existe codificación de etiquetas, `clases_`, `numero_clases` ni decodificación posterior para `votacion_uniforme`

## 5. Distancias

Los kernels calculan la distancia euclidiana cuadrada entre cada consulta y cada muestra de entrenamiento. La búsqueda no calcula la raíz cuadrada, porque la transformación es monótona y no cambia el orden de los vecinos

`vecinos_mas_cercanos()` devuelve distancia euclidiana normal para los `k` resultados finales, por tanto la raíz cuadrada se aplica únicamente después de completar la selección de los vecinos, nunca durante el recorrido de búsqueda ni para los candidatos descartados

`predecir()` con votación uniforme no necesita calcular raíces: la clasificación depende del orden de los vecinos y de sus etiquetas, no de la escala transformada de la distancia

## 6. Validaciones

La capa Python valida las entradas públicas antes de solicitar operaciones nativas. Como mínimo, comprueba:

- Dimensiones correctas para datos de entrenamiento y consultas
- Tipos compatibles sin conversiones automáticas de dtype
- Conjuntos no vacíos cuando la operación requiera datos
- Coincidencia entre el número de muestras de entrenamiento y el número de etiquetas
- Coincidencia entre el número de características de entrenamiento y consultas
- Ausencia de valores `NaN` e infinito en las características
- Valor válido de `k`: entero positivo y no mayor que el número de muestras de entrenamiento
- Disponibilidad del kernel CPU cuando se solicite CPU
- Disponibilidad de runtime CUDA y del kernel CUDA cuando se solicite CUDA
- Coincidencia de dispositivo entre los tensores que participan en una operación nativa

Python debe generar errores comprensibles, indicando la condición incumplida y, cuando sea útil, los valores observados y esperados. C++ volverá a comprobar las precondiciones críticas relacionadas con tamaños, tipos, dispositivos y parámetros antes de lanzar kernels. Esta segunda comprobación protege la frontera nativa y evita depender exclusivamente de la validación de alto nivel

## 7. Flujo de `ajustar()`

La integración de Fase 4 sigue este flujo:

1. Validar las dimensiones, dtype, valores finitos, tamaños y correspondencia entre muestras y etiquetas sin alterar el contrato NumPy
2. Conservar `datos_entrenamiento_` y `etiquetas_entrenamiento_` como copias NumPy independientes
3. Resolver el dispositivo efectivo según `"cpu"`, `"cuda"` o `"auto"`
4. Preparar una representación Tensor del entrenamiento una sola vez en el dispositivo efectivo
5. Establecer `dispositivo_efectivo_` durante `ajustar()`

`ajustar()` no ejecuta entrenamiento: KNN no aprende parámetros numéricos. La operación prepara y conserva el conjunto de referencia para consultas posteriores

Cuando el dispositivo efectivo sea CUDA, el conjunto de entrenamiento completo debe caber en la VRAM disponible y no se contempla una representación parcial ni procesamiento out-of-core durante `ajustar()`

## 8. Flujo de `vecinos_mas_cercanos()`

La integración reutiliza `distancias_l2_cuadradas` y `seleccionar_top_k` bajo el Dispatcher para el dispositivo efectivo

La capa Python aplica la raíz cuadrada únicamente a las distancias seleccionadas y devuelve `numpy.ndarray` para preservar el contrato público

La ordenación conserva distancia ascendente y menor índice de entrenamiento en empate

## 9. Flujo de `predecir()`

En la referencia CPU actual, `predecir_knn` sigue este flujo:

1. Obtener `indices_seleccionados` mediante `seleccionar_top_k`
2. Recuperar `etiquetas_vecinos = etiquetas_entrenamiento[indices_seleccionados]`
3. Invocar `votacion_uniforme(etiquetas_vecinos)`
4. Devolver las predicciones con etiquetas originales

`votacion_uniforme` recibe únicamente `etiquetas_vecinos`, realiza votación uniforme y resuelve los empates por la etiqueta original menor

La integración invoca `knn_cuda::predecir_knn` mediante el Dispatcher y conserva las mismas etiquetas originales y reglas deterministas que la referencia NumPy

`predecir()` evita calcular raíces cuadradas y realiza únicamente las transferencias necesarias para obtener las etiquetas finales, la búsqueda sigue siendo exacta porque los valores de distancia no participan en la votación

## 10. Reglas de desempate

Las reglas de orden y desempate son parte del contrato funcional:

- Los vecinos se ordenan por distancia ascendente
- Si dos vecinos tienen la misma distancia, se prioriza el menor índice de entrenamiento
- Si dos o más etiquetas tienen el mismo número de votos, gana la etiqueta original menor
- El comportamiento completo debe ser determinista bajo las mismas entradas y configuración

El orden por índice debe aplicarse al seleccionar resultados finales. No se debe depender de un orden incidental producido por el paralelismo de CUDA

## 11. Capas

La separación vigente de módulos es la siguiente:

- `clasificador.py` conserva el estado del clasificador y presenta la API pública
- `referencia.py` contiene la referencia CPU en NumPy para pruebas y comparación
- `src/cpp/operadores.cpp` implementa los operadores CPU
- `src/cpp/registro.cpp` define los esquemas y registra implementaciones por dispositivo
- `src/cuda/` contiene las implementaciones CUDA de los operadores

CUDA no conoce Python ni conserva estado entre invocaciones. C++ tampoco conserva el estado del clasificador y recibe tensores y parámetros de una operación. `votacion_uniforme` recibe directamente las etiquetas originales y devuelve la predicción original

## 12. Operadores internos

Los operadores internos vigentes son:

### `distancias_l2_cuadradas`

- **Entradas:** `datos_consulta` y `datos_entrenamiento` con formas compatibles `[Q, D]` y `[N, D]`, ambos en `float32` y en el mismo dispositivo CPU o CUDA
- **Salida:** distancias euclidianas cuadradas con forma `[Q, N]`
- **Responsabilidad:** calcular la suma de diferencias al cuadrado por dimensión. No ordena vecinos, no aplica raíz cuadrada y no conoce etiquetas

### `seleccionar_top_k`

- **Entradas:** matriz de distancias `[Q, N]` en `float32` y `k`
- **Salidas:** distancias seleccionadas `float32` e índices `int64` con forma `[Q, k]`
- **Responsabilidad:** seleccionar vecinos por distancia ascendente y menor índice original en empate

### `votacion_uniforme`

- **Entradas:** `etiquetas_vecinos` con forma `[Q, K]`, dtype entero compatible y dispositivo CPU o CUDA
- **Salida:** `predicciones` con forma `[Q]`, el mismo dtype entero y el dispositivo de entrada
- **Responsabilidad:** realizar la votación uniforme sobre etiquetas originales y resolver empates por la etiqueta numéricamente menor

`votacion_uniforme` no recibe índices de vecinos, clases, `numero_clases`, etiquetas codificadas ni accede al conjunto completo de etiquetas de entrenamiento. La recuperación de `etiquetas_vecinos` a partir de los índices ocurre antes de invocar la operación

El Dispatcher resuelve la implementación CPU C++ o CUDA de `votacion_uniforme` bajo el mismo contrato lógico

Estos operadores son sin estado. Python conserva las copias NumPy de entrenamiento y etiquetas, prepara tensores internos durante `ajustar()` y los mantiene en el dispositivo efectivo para reutilizarlos

## 13. Estrategia progresiva

La implementación se completó en etapas verificables:

1. **Referencia CPU NumPy:** define el comportamiento correcto, las formas, la ordenación y los desempates
2. **Backend C++ CPU:** implementa y valida los cuatro operadores bajo el Dispatcher
3. **Backend CUDA:** implementa y valida los mismos cuatro operadores bajo los mismos esquemas
4. **Integración de Fase 4:** conecta `ClasificadorKNNCUDA` con CPU C++ y CUDA sin ampliar sus tipos públicos de entrada
5. **Optimización de Fase 5:** incorporó tiling 2D con shared memory en distancias y reducción híbrida por warps en top-k después de validar corrección y rendimiento

La progresión mantiene una referencia funcional disponible en todas las etapas. El orden de prioridad es primero corrección y después rendimiento

Las optimizaciones adicionales quedan fuera del alcance de esta versión

## 14. Pruebas

Las pruebas se dividen entre entornos locales y CUDA:

- Las pruebas locales deben ejecutarse sin GPU y utilizar la referencia CPU en NumPy
- Las pruebas CUDA se ejecutan en cualquier entorno CUDA compatible y Google Colab es una opción de validación, no una dependencia
- La comparación primaria es contra NumPy, que define los valores esperados de distancia, índices y predicciones bajo las reglas del proyecto
- La comparación secundaria con scikit-learn queda fuera del alcance de esta versión y no forma parte de sus dependencias

La matriz de casos debe incluir, como mínimo:

- Clasificación binaria
- Clasificación multiclase
- Una sola clase
- Empate de distancias
- Empate de votos
- Una dimensión
- Muchas características
- Puntos duplicados
- Varias consultas
- Una sola consulta
- `NaN`
- Infinito
- `k = 1`
- `k = N`
- `k = 0`
- `k > N`
- Formas incompatibles
- Número de etiquetas distinto del número de muestras
- Diferente cantidad de características entre entrenamiento y consultas
- Entradas inválidas y mensajes de error

La tolerancia para valores `float32` es `rtol = 1e-4` y `atol = 1e-5`. Los índices y las predicciones deben coincidir exactamente según las reglas de orden y desempate definidas. La tolerancia numérica aplica a las distancias, no a la identidad de los vecinos ni a las etiquetas resultantes

## 15. Benchmarks

Los benchmarks deben distinguir las siguientes mediciones:

- Tiempo de kernel CUDA
- Tiempo del motor, incluyendo la coordinación de operaciones en el dispositivo
- Tiempo de extremo a extremo desde la API Python
- Transferencias de entrada y salida medidas por separado

Cada benchmark debe incluir calentamiento antes de registrar resultados, realizar varias repeticiones y reportar al menos la mediana. Las mediciones de CUDA deben usar sincronización explícita en los puntos necesarios para que la asincronía no oculte el tiempo real de ejecución

Los benchmarks reportan mediana, promedio, desviación y *speedup* respecto a la referencia CPU elegida. Los experimentos varían tamaños de dataset, dimensionalidad, número de consultas y valores de `k`

Las consultas por segundo quedan fuera del alcance de esta versión

Cada resultado debe registrar hardware, versión de CUDA, versión de Python, versión y configuración de PyTorch, configuración de compilación, tamaños de entrada y parámetros del experimento. No se afirmará una mejora sin benchmarks reproducibles y resultados revisados funcionalmente

## 16. Limitaciones iniciales

La versión actual tiene estas limitaciones explícitas:

- El conjunto de entrenamiento completo debe caber en VRAM
- No habrá procesamiento out-of-core
- No habrá búsqueda aproximada
- No habrá regresión
- No habrá ejecución multi-GPU
- No habrá métricas arbitrarias; se utilizará la distancia euclidiana cuadrada definida
- No habrá etiquetas de texto
- No habrá autograd ni integración de gradientes
- La extensión nativa conserva implementaciones CPU C++ y CUDA bajo los mismos esquemas del Dispatcher

El desarrollo local sin GPU puede utilizar el backend CPU C++ y las pruebas CPU sin requerir CUDA

## 17. Estructura actual del repositorio

La estructura actual es:

```text
KNN-Cuda/
├── .gitignore
├── src/
│   ├── python/knn_cuda/
│   │   ├── __init__.py
│   │   ├── _backend_nativo.py
│   │   ├── clasificador.py
│   │   └── referencia.py
│   ├── cpp/
│   │   ├── include/knn_cuda/operadores.h
│   │   ├── operadores.cpp
│   │   └── registro.cpp
│   └── cuda/
│       ├── include/knn_cuda/operadores_cuda.h
│       ├── distancias_l2_cuadradas.cu
│       ├── seleccionar_top_k.cu
│       ├── votacion_uniforme.cu
│       └── predecir_knn.cu
├── tests/
│   ├── cpu/
│   │   ├── test_referencia.py
│   │   ├── test_clasificador.py
│   │   └── test_backend_cpp.py
│   └── cuda/
│       ├── test_distancias_l2_cuadradas_cuda.py
│       ├── test_seleccionar_top_k_cuda.py
│       ├── test_votacion_uniforme_cuda.py
│       ├── test_predecir_knn_cuda.py
│       └── test_clasificador_cuda.py
├── benchmarks/
│   ├── __init__.py
│   ├── medicion.py
│   ├── perfilado.py
│   ├── benchmark_operadores.py
│   ├── benchmark_pipeline.py
│   ├── benchmark_clasificador.py
│   ├── perfilar_distancias_cuda.py
│   ├── perfilar_seleccionar_top_k_cuda.py
│   └── perfilar_pipeline_cuda.py
├── docs/
├── LICENSE
├── README.md
├── pyproject.toml
├── requirements.txt
└── setup.py
```

La estructura separa la API Python, la extensión C++, los kernels CUDA, las pruebas CPU, las pruebas CUDA y la documentación. Los nombres de archivos reflejan responsabilidades

## 18. Reglas arquitectónicas

- La corrección tiene prioridad sobre el rendimiento
- Cada capa tiene una única responsabilidad y debe comunicarse mediante contratos claros
- Ninguna optimización se acepta sin evidencia de mejora mediante benchmarks reproducibles
- El procesamiento por bloques no puede modificar los resultados frente al recorrido completo
- Los operadores C++/CUDA no conservan estado
- El código propio del proyecto se escribe en español latinoamericano
- Los identificadores propios se escriben sin tildes ni ñ
- Los nombres externos, las APIs de terceros y los términos obligatorios se conservan en su forma original
- La documentación se escribe en español
- Cada cambio debe tener pruebas y revisión

Estas reglas se aplican tanto a nuevas funcionalidades como a optimizaciones internas. Una modificación que mejore el tiempo pero altere la ordenación, los desempates, las distancias dentro de la tolerancia o las etiquetas predichas no cumple el diseño

## Filosofía del proyecto
KNN-Cuda prioriza la corrección antes que el rendimiento

Cada etapa debe ser verificable

Cada optimización debe demostrar su beneficio mediante benchmarks reproducibles

La arquitectura debe favorecer la claridad y la mantenibilidad antes que la complejidad :)
