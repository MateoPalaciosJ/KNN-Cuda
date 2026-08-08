# Diseño de software de KNN-Cuda

Este documento define el diseño técnico inicial de KNN-Cuda

Complementa la arquitectura y los requisitos del proyecto

Su principio central es demostrar primero la corrección del software y después optimizar el rendimiento

## 1. Producto y API pública

El paquete Python se llamará `knn_cuda`. La clase pública principal será `ClasificadorKNNCUDA`, orientada a clasificación KNN exacta

La clase proporcionará las operaciones públicas `__init__()`, `ajustar()`, `vecinos_mas_cercanos()` y `predecir()`

Durante la Fase 1, el constructor inicial será `ClasificadorKNNCUDA(numero_vecinos=5)` y solo recibirá este parámetro

- `numero_vecinos`: número de vecinos que se utilizarán

`tamano_lote_consultas` y `tamano_bloque_entrenamiento` pertenecen al backend CUDA futuro para controlar el procesamiento por lotes y bloques, no son parámetros obligatorios del constructor CPU inicial y solo podrán incorporarse a la API pública cuando la integración CUDA lo requiera

El comportamiento de la API será el siguiente:

- `ajustar()` recibe el conjunto de entrenamiento y sus etiquetas, prepara el estado del clasificador y devuelve `self` para permitir encadenamiento
- `vecinos_mas_cercanos()` recibe consultas y devuelve distancias e índices de los vecinos, respetando el orden determinista definido en este documento
- `predecir()` busca los vecinos, aplica votación uniforme y devuelve las etiquetas en su representación original

El estado del clasificador, incluyendo el conjunto de entrenamiento, las etiquetas originales y los metadatos, permanece gestionado por Python. La API debe ocultar al usuario la coordinación de la extensión C++ y de los kernels CUDA

## 2. Integración tecnológica

La estrategia nativa oficial utiliza la infraestructura de extensiones de PyTorch y operadores registrados mediante su dispatcher

Durante la Fase 2, la extensión se compilará con `CppExtension` y expondrá operadores CPU sin utilizar CUDA ni GPU

Durante la Fase 3, la misma arquitectura evolucionará a `CUDAExtension` para incorporar implementaciones CUDA sin reemplazar la ruta de integración ni modificar la API pública

No se utilizará pybind11 como estrategia independiente ni existirá un segundo sistema de binding

La división de responsabilidades será:

- Python conserva el estado, valida las entradas públicas y coordina las llamadas
- C++ valida las precondiciones críticas, recibe tensores y ejecuta operadores CPU en la Fase 2 u operaciones CUDA en la Fase 3
- CUDA ejecutará operaciones sin estado sobre los tensores proporcionados a partir de la Fase 3

Los operadores C++/CUDA no conservarán estado entre invocaciones ni conocerán la instancia de `ClasificadorKNNCUDA`. Reciben tensores con las etiquetas originales cuando la operación lo requiere y no necesitan metadatos externos de clases

La compilación oficial será anticipada, como parte del flujo normal de preparación del paquete o del entorno de ejecución. La compilación JIT se utilizará únicamente para experimentación, prototipos y validaciones tempranas; no será el mecanismo oficial de distribución ni de ejecución estable

## 3. Contratos de datos

Los tensores de características respetarán estos contratos:

- `datos_entrenamiento` tiene forma `[N, D]`, donde `N` es el número de muestras y `D` el número de características
- `datos_consulta` tiene forma `[Q, D]`, donde `Q` es el número de consultas
- `datos_entrenamiento` y `datos_consulta` usan `float32`
- Las etiquetas usan un dtype entero compatible y conservan su representación original
- Los índices de vecinos usan `int64`
- Los tensores internos son contiguos
- Todos los tensores internos se encuentran en el mismo dispositivo CUDA

La API Python puede recibir `numpy.ndarray` o `torch.Tensor`. La capa Python convierte las entradas al formato interno requerido: tensor PyTorch, tipo `float32` para características, tipo `int64` para índices, dtype entero original para etiquetas, disposición contigua y dispositivo CUDA. La conversión debe ser explícita y validable para que el usuario reciba errores comprensibles cuando una entrada no sea compatible

Las salidas conservan el tipo de la entrada pública correspondiente: si la entrada pública es un `torch.Tensor`, la salida correspondiente también será un `torch.Tensor`; si la entrada pública es un `numpy.ndarray`, la salida correspondiente también será un `numpy.ndarray`. Internamente, todas las operaciones producen tensores CUDA y la capa Python realiza la conversión final cuando corresponde

`vecinos_mas_cercanos()` devuelve distancias `float32` e índices `int64` y `predecir()` devuelve las etiquetas originales

## 4. Etiquetas

La primera versión acepta etiquetas enteras originales sin necesidad de que sean consecutivas

Las etiquetas pueden ser negativas o tener valores grandes y se entregan directamente a `votacion_uniforme` tanto en CPU como en CUDA

El flujo de etiquetas será:

1. `seleccionar_top_k` obtiene `indices_seleccionados`
2. La capa que coordina el pipeline recupera `etiquetas_vecinos` desde `etiquetas_entrenamiento`
3. `votacion_uniforme(etiquetas_vecinos)` recibe las etiquetas originales y devuelve predicciones originales

Las etiquetas de texto quedan fuera del alcance inicial

No existe codificación de etiquetas, `clases_`, `numero_clases` ni decodificación posterior para `votacion_uniforme`

## 5. Distancias

Los kernels calculan la distancia euclidiana cuadrada entre cada consulta y cada muestra de entrenamiento. La búsqueda no calcula la raíz cuadrada, porque la transformación es monótona y no cambia el orden de los vecinos

`vecinos_mas_cercanos()` devolverá distancia euclidiana normal para los `k` resultados finales, por tanto la raíz cuadrada se aplicará únicamente después de completar la selección de los vecinos, nunca durante el recorrido de búsqueda ni para los candidatos descartados

`predecir()` con votación uniforme no necesita calcular raíces: la clasificación depende del orden de los vecinos y de sus etiquetas, no de la escala transformada de la distancia

## 6. Validaciones

La capa Python validará las entradas públicas antes de solicitar operaciones nativas. Como mínimo, comprobará:

- Dimensiones correctas para datos de entrenamiento y consultas
- Tipos compatibles y conversiones permitidas
- Conjuntos no vacíos cuando la operación requiera datos
- Coincidencia entre el número de muestras de entrenamiento y el número de etiquetas
- Coincidencia entre el número de características de entrenamiento y consultas
- Ausencia de valores `NaN` e infinito en las características
- Valor válido de `k`: entero positivo y no mayor que el número de muestras de entrenamiento
- Disponibilidad de CUDA para las operaciones del motor
- Contigüidad de los tensores internos después de la conversión
- Coincidencia de dispositivo entre los tensores que participan en una operación

Python debe generar errores comprensibles, indicando la condición incumplida y, cuando sea útil, los valores observados y esperados. C++ volverá a comprobar las precondiciones críticas relacionadas con tamaños, tipos, dispositivos y parámetros antes de lanzar kernels. Esta segunda comprobación protege la frontera nativa y evita depender exclusivamente de la validación de alto nivel

## 7. Flujo de `ajustar()`

`ajustar()` seguirá este flujo cuando se integre el backend CUDA:

1. Validar las dimensiones, tipos, valores finitos, tamaños y correspondencia entre muestras y etiquetas
2. Convertir las características a `float32`, obtener una disposición contigua y preparar el dispositivo CUDA
3. Conservar las etiquetas enteras originales con su dtype compatible
4. Transferir a GPU los tensores que deban ser utilizados por el motor
5. Almacenar en el objeto Python los tensores preparados, las etiquetas originales y los metadatos necesarios

`ajustar()` no ejecuta entrenamiento: KNN no aprende parámetros numéricos. La operación prepara y conserva el conjunto de referencia para consultas posteriores

En la versión inicial con CUDA, el conjunto de entrenamiento completo debe caber en la VRAM disponible, no se contempla una representación parcial ni procesamiento out-of-core durante `ajustar()`

## 8. Flujo de `vecinos_mas_cercanos()`

En el backend CUDA futuro, la búsqueda exacta se organizará en dos niveles de partición:

1. Python divide `datos_consulta` en lotes de tamaño limitado por el parámetro interno o futuro `tamano_lote_consultas`
2. `buscar_vecinos_knn` recibe únicamente un lote de consultas
3. C++ y CUDA recorren `datos_entrenamiento` en bloques de tamaño limitado por `tamano_bloque_entrenamiento`
4. Para cada bloque se calculan las distancias euclidianas cuadradas
5. Se obtiene un Top-K local por consulta dentro del bloque
6. El Top-K local se fusiona con el Top-K global acumulado para esa consulta
7. Se repite el proceso hasta revisar exactamente todas las muestras de entrenamiento
8. Al terminar cada lote, se ordenan los `k` vecinos finales y se convierten sus distancias cuadradas a distancia euclidiana normal para la salida pública de `vecinos_mas_cercanos()`
9. Python concatena los resultados de todos los lotes y mantiene el orden original de `datos_consulta`

La división por lotes y bloques no puede cambiar los resultados. La fusión debe conservar el orden por distancia ascendente y, en igualdad de distancia, por menor índice de entrenamiento. La matriz completa `Q x N` no debe ser necesaria en la versión final, porque el consumo temporal de memoria debe estar acotado por el tamaño de los lotes, los bloques y los candidatos Top-K

## 9. Flujo de `predecir()`

En la referencia CPU actual, `predecir_knn` sigue este flujo:

1. Obtener `indices_seleccionados` mediante `seleccionar_top_k`
2. Recuperar `etiquetas_vecinos = etiquetas_entrenamiento[indices_seleccionados]`
3. Invocar `votacion_uniforme(etiquetas_vecinos)`
4. Devolver las predicciones con etiquetas originales

`votacion_uniforme` recibe únicamente `etiquetas_vecinos`, realiza votación uniforme y resuelve los empates por la etiqueta original menor

En el backend CUDA, `predecir()` reutilizará la búsqueda de vecinos y recuperará las etiquetas originales correspondientes a los índices antes de invocar `votacion_uniforme`, igual que el backend CPU

Cuando sea posible, `predecir()` debe evitar calcular raíces cuadradas, conservar matrices temporales que no necesita y realizar únicamente las transferencias necesarias para obtener las etiquetas finales, la búsqueda seguirá siendo exacta aunque se omitan de la ruta de predicción los datos de distancia que no participan en la votación

## 10. Reglas de desempate

Las reglas de orden y desempate son parte del contrato funcional:

- Los vecinos se ordenan por distancia ascendente
- Si dos vecinos tienen la misma distancia, se prioriza el menor índice de entrenamiento
- Si dos o más etiquetas tienen el mismo número de votos, gana la etiqueta original menor
- El comportamiento completo debe ser determinista bajo las mismas entradas y configuración

El orden por índice debe aplicarse tanto al ordenar resultados finales como al fusionar candidatos parciales. No se debe depender de un orden incidental producido por el paralelismo de CUDA

## 11. Capas

La separación inicial de módulos será la siguiente:

- `clasificador.py` conserva el estado del clasificador y presenta la API pública
- `_validacion.py` valida las entradas públicas y los parámetros
- `_operaciones.py` carga y encapsula los operadores registrados de PyTorch
- `referencia.py` contiene la referencia CPU en NumPy para pruebas y comparación
- `operaciones_knn.cpp` registra y valida los operadores C++/CUDA y coordina sus lanzamientos
- `distancias_kernel.cu` calcula las distancias euclidianas cuadradas
- `seleccion_top_k_kernel.cu` selecciona y fusiona vecinos locales y globales
- `votacion_uniforme.cu` realiza la votación sobre etiquetas originales

CUDA no conoce Python ni conserva estado entre invocaciones. C++ tampoco conserva el estado del clasificador y recibe tensores y parámetros de una operación. `votacion_uniforme` recibe directamente las etiquetas originales y devuelve la predicción original

## 12. Operadores internos

Los operadores internos iniciales serán:

### `distancias_l2_cuadradas`

- **Entradas:** `datos_consulta` y `datos_entrenamiento` contiguos con formas compatibles `[Q, D]` y `[N, D]`, ambos en `float32` y en el mismo dispositivo CUDA
- **Salida:** distancias euclidianas cuadradas para el bloque de consultas y el bloque de entrenamiento que se estén procesando
- **Responsabilidad:** calcular la suma de diferencias al cuadrado por dimensión. No ordena vecinos, no aplica raíz cuadrada y no conoce etiquetas

### `buscar_vecinos_knn`

- **Entradas:** un lote de `datos_consulta`, `datos_entrenamiento`, `k` y `tamano_bloque_entrenamiento`, con los tensores contiguos en `float32` y en el mismo dispositivo CUDA
- **Salidas:** índices `int64` y distancias cuadradas de los `k` vecinos por consulta, ordenados según las reglas de desempate
- **Responsabilidad:** recorrer `datos_entrenamiento` por bloques, calcular distancias, obtener Top-K local, fusionar Top-K global y garantizar que ninguna muestra quede sin revisar

### `votacion_uniforme`

- **Entradas:** `etiquetas_vecinos` con forma `[Q, K]`, dtype entero compatible y dispositivo CPU o CUDA
- **Salida:** `predicciones` con forma `[Q]`, el mismo dtype entero y el dispositivo de entrada
- **Responsabilidad:** realizar la votación uniforme sobre etiquetas originales y resolver empates por la etiqueta numéricamente menor

`votacion_uniforme` no recibe índices de vecinos, clases, `numero_clases`, etiquetas codificadas ni accede al conjunto completo de etiquetas de entrenamiento. La recuperación de `etiquetas_vecinos` a partir de los índices ocurre antes de invocar la operación

El Dispatcher resuelve la implementación CPU C++ o CUDA de `votacion_uniforme` bajo el mismo contrato lógico

Estos operadores son sin estado. Python conserva las referencias a los tensores de entrenamiento y etiquetas, y los pasa en cada llamada cuando corresponde

## 13. Estrategia progresiva

La implementación se desarrollará en etapas verificables:

1. **Referencia CPU NumPy:** definir el comportamiento correcto, las formas, la ordenación y los desempates sin depender de CUDA
2. **Extensión C++ CPU de PyTorch:** registrar operadores propios, validar el intercambio Python → PyTorch → C++ y comparar sus salidas con la referencia NumPy sin utilizar CUDA
3. **Matriz CUDA completa con selección temporal:** construir una versión sencilla que materialice temporalmente la matriz de distancias para validar la integración y los kernels
4. **Selección CUDA propia:** sustituir la selección temporal por una selección paralela de candidatos con reglas deterministas
5. **Procesamiento por bloques:** introducir `tamano_lote_consultas` y `tamano_bloque_entrenamiento`, fusionar resultados parciales y eliminar la necesidad de la matriz completa `Q x N`
6. **Optimización avanzada:** mejorar acceso a memoria, ocupación, uso de memoria compartida y organización de kernels únicamente cuando existan pruebas de corrección y benchmarks que justifiquen el cambio

La progresión mantiene una referencia funcional disponible en todas las etapas. El orden de prioridad es primero corrección y después rendimiento

## 14. Pruebas

Las pruebas se dividirán entre entornos locales y CUDA:

- Las pruebas locales deben ejecutarse sin GPU y utilizar la referencia CPU en NumPy
- Las pruebas CUDA se ejecutarán en Google Colab con GPU NVIDIA
- La comparación primaria será contra NumPy, que definirá los valores esperados de distancia, índices y predicciones bajo las reglas del proyecto
- La comparación secundaria será contra scikit-learn en casos controlados compatibles con su configuración

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

La tolerancia inicial para valores `float32` será `rtol = 1e-4` y `atol = 1e-5`. Los índices y las predicciones deben coincidir exactamente según las reglas de orden y desempate definidas. La tolerancia numérica aplica a las distancias, no a la identidad de los vecinos ni a las etiquetas resultantes

## 15. Benchmarks

Los benchmarks deben distinguir las siguientes mediciones:

- Tiempo de kernel CUDA
- Tiempo del motor, incluyendo la coordinación de operaciones en el dispositivo
- Tiempo de extremo a extremo desde la API Python
- Transferencias de entrada y salida medidas por separado

Cada benchmark debe incluir calentamiento antes de registrar resultados, realizar varias repeticiones y reportar al menos la mediana. Las mediciones de CUDA deben usar sincronización explícita en los puntos necesarios para que la asincronía no oculte el tiempo real de ejecución

Se reportarán consultas por segundo y *speedup* respecto a la referencia CPU elegida. Los experimentos variarán tamaños de dataset, dimensionalidad, número de consultas y valores de `k`

Cada resultado debe registrar hardware, versión de CUDA, versión de Python, versión y configuración de PyTorch, configuración de compilación, tamaños de entrada y parámetros del experimento. No se afirmará una mejora sin benchmarks reproducibles y resultados revisados funcionalmente

## 16. Limitaciones iniciales

La primera versión tendrá estas limitaciones explícitas:

- El conjunto de entrenamiento completo debe caber en VRAM
- No habrá procesamiento out-of-core
- No habrá búsqueda aproximada
- No habrá regresión
- No habrá ejecución multi-GPU
- No habrá métricas arbitrarias; se utilizará la distancia euclidiana cuadrada definida
- No habrá etiquetas de texto
- No habrá autograd ni integración de gradientes
- No habrá backend CPU dentro de la extensión CUDA

El desarrollo local sin GPU se apoyará en la referencia NumPy y en las pruebas que no requieran la extensión CUDA. Esto no convierte la extensión CUDA en un motor CPU alternativo

## 17. Arquitectura objetivo de carpetas

La estructura objetivo completa será:

```text
KNN-Cuda/
├── .github/
├── src/
│   ├── python/
│   │   └── knn_cuda/
│   │       ├── __init__.py       # API pública del paquete
│   │       ├── clasificador.py   # ClasificadorKNNCUDA y estado Python
│   │       ├── _validacion.py    # Validación de entradas y parámetros
│   │       ├── _operaciones.py   # Carga y encapsulado de operadores
│   │       └── referencia.py     # Referencia CPU en NumPy
│   ├── cpp/
│   │   ├── include/
│   │   │   └── knn_cuda/
│   │   │       └── operaciones.h # Declaraciones de operadores
│   │   └── operaciones_knn.cpp   # Registro y validación de operadores
│   └── cuda/
│       ├── include/
│       │   └── knn_cuda/
│       │       └── kernels.cuh   # Declaraciones de kernels
│       ├── distancias_kernel.cu       # Distancia euclidiana cuadrada
│       ├── seleccion_top_k_kernel.cu  # Selección y fusión Top-K
│       └── votacion_kernel.cu         # Votación uniforme
├── tests/
│   ├── cpu/
│   │   ├── test_referencia.py    # Pruebas de la referencia NumPy
│   │   ├── test_validacion.py    # Pruebas de validación
│   │   └── test_etiquetas.py     # Pruebas de etiquetas
│   └── cuda/
│       ├── test_distancias.py    # Pruebas de distancias
│       ├── test_seleccion_top_k.py # Pruebas de selección Top-K
│       ├── test_votacion.py      # Pruebas de votación
│       └── test_clasificador.py  # Pruebas del clasificador
├── benchmarks/
│   ├── medicion_cpu.py           # Mediciones de referencia CPU
│   ├── medicion_cuda.py          # Mediciones CUDA
│   └── configuraciones.py        # Configuraciones de medición
├── data/                         # Datos pequeños o referencias reproducibles
├── docs/
│   ├── architecture.md           # Arquitectura inicial
│   ├── requirements.md           # Requisitos iniciales
│   └── software_design.md        # Diseño de software
├── examples/                     # Ejemplos de uso de la API
├── notebooks/                    # Exploración y ejecución en Google Colab
├── pyproject.toml                # Configuración del paquete y compilación
├── requirements.txt              # Dependencias del entorno
├── .gitignore                    # Archivos excluidos del control de versiones
├── LICENSE                       # Licencia del proyecto
└── README.md                     # Introducción y guía de inicio
```

La estructura separa la API Python, la extensión C++, los kernels CUDA, las pruebas CPU, las pruebas CUDA, los benchmarks, la documentación, los ejemplos y los notebooks. Los nombres de archivos reflejan responsabilidades; no implican que deban implementarse todos los módulos en una única etapa

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
