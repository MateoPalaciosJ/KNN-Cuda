# Diseño de software de KNN-Cuda

Este documento define el diseño técnico inicial de KNN-Cuda. Complementa la arquitectura y los requisitos del proyecto y mantiene como principio rector que la corrección debe demostrarse antes de optimizar el rendimiento.

## 1. Producto y API pública

El paquete Python se llamará `knn_cuda`. La clase pública principal será `CUDAKNNClassifier`, orientada a clasificación KNN exacta.

La clase proporcionará las operaciones públicas `__init__()`, `fit()`, `kneighbors()` y `predict()`.

Los parámetros iniciales del constructor serán:

- `n_neighbors`: número de vecinos que se utilizarán.
- `query_batch_size`: número máximo de consultas procesadas por lote.
- `train_block_size`: número de muestras de entrenamiento procesadas por bloque.

El comportamiento de la API será el siguiente:

- `fit()` recibe el conjunto de entrenamiento y sus etiquetas, prepara el estado del clasificador y devuelve `self` para permitir encadenamiento.
- `kneighbors()` recibe consultas y devuelve distancias e índices de los vecinos, respetando el orden determinista definido en este documento.
- `predict()` busca los vecinos, aplica votación uniforme y devuelve las etiquetas en su representación original.

El estado del clasificador, incluyendo el conjunto de entrenamiento, las etiquetas codificadas y los metadatos, permanece gestionado por Python. La API debe ocultar al usuario la coordinación de la extensión C++ y de los kernels CUDA.

## 2. Integración tecnológica

La implementación utilizará una extensión C++/CUDA integrada con PyTorch. La extensión se compilará con `CUDAExtension` y expondrá operadores registrados mediante el dispatcher de PyTorch.

La división de responsabilidades será:

- Python conserva el estado, valida las entradas públicas y coordina las llamadas.
- C++ valida las precondiciones críticas, recibe tensores y lanza las operaciones CUDA.
- CUDA ejecuta operaciones sin estado sobre los tensores proporcionados.

Los operadores C++/CUDA no conservarán estado entre invocaciones ni conocerán la instancia de `CUDAKNNClassifier`. Tampoco conocerán las etiquetas originales; trabajarán únicamente con tensores y etiquetas codificadas.

La compilación oficial será anticipada, como parte del flujo normal de preparación del paquete o del entorno de ejecución. La compilación JIT se utilizará únicamente para experimentación, prototipos y validaciones tempranas; no será el mecanismo oficial de distribución ni de ejecución estable.

## 3. Contratos de datos

Los tensores de características respetarán estos contratos:

- `X_train` tiene forma `[N, D]`, donde `N` es el número de muestras y `D` el número de características.
- `X_query` tiene forma `[Q, D]`, donde `Q` es el número de consultas.
- `X_train` y `X_query` usan `float32`.
- Las etiquetas codificadas usan `int64`.
- Los índices de vecinos usan `int64`.
- Los tensores internos son contiguos.
- Todos los tensores internos se encuentran en el mismo dispositivo CUDA.

La API Python puede recibir `numpy.ndarray` o `torch.Tensor`. La capa Python convierte las entradas al formato interno requerido: tensor PyTorch, tipo `float32` para características, tipo `int64` para índices y etiquetas codificadas, disposición contigua y dispositivo CUDA. La conversión debe ser explícita y validable para que el usuario reciba errores comprensibles cuando una entrada no sea compatible.

Las salidas públicas conservarán las convenciones de la API: las distancias se devolverán como valores numéricos asociados a cada vecino y los índices como valores enteros. La implementación debe documentar el dispositivo y el tipo de cada salida pública cuando se concrete la interfaz de empaquetado.

## 4. Etiquetas

La primera versión aceptará etiquetas enteras. Antes de invocar a CUDA, Python ordenará las etiquetas distintas y las codificará como valores contiguos desde cero.

Por ejemplo, si las clases originales ordenadas son `2`, `5` y `9`, la representación interna será `0`, `1` y `2`, respectivamente. Esta transformación permite que los kernels trabajen con un dominio compacto y conocido.

El flujo de etiquetas será:

1. Python obtiene las clases distintas y las almacena ordenadas en `classes_`.
2. Python transforma cada etiqueta original en su código entero contiguo.
3. CUDA utiliza únicamente las etiquetas codificadas para la votación.
4. Python recupera las etiquetas originales mediante `classes_` antes de devolver las predicciones.

Las etiquetas de texto quedan fuera del alcance inicial. La clase `classes_` debe conservar el orden de las etiquetas originales ordenadas para que la decodificación sea determinista y para que el menor valor de etiqueta resuelva empates de votos.

## 5. Distancias

Los kernels calculan la distancia euclidiana cuadrada entre cada consulta y cada muestra de entrenamiento. La búsqueda no calcula la raíz cuadrada, porque la transformación es monótona y no cambia el orden de los vecinos.

`kneighbors()` devolverá distancia euclidiana normal para los `k` resultados finales. Por tanto, la raíz cuadrada se aplicará únicamente después de completar la selección de los vecinos, nunca durante el recorrido de búsqueda ni para los candidatos descartados.

`predict()` con votación uniforme no necesita calcular raíces: la clasificación depende del orden de los vecinos y de sus etiquetas, no de la escala transformada de la distancia.

## 6. Validaciones

La capa Python validará las entradas públicas antes de solicitar operaciones nativas. Como mínimo, comprobará:

- Dimensiones correctas para datos de entrenamiento y consultas.
- Tipos compatibles y conversiones permitidas.
- Conjuntos no vacíos cuando la operación requiera datos.
- Coincidencia entre el número de muestras de entrenamiento y el número de etiquetas.
- Coincidencia entre el número de características de entrenamiento y consultas.
- Ausencia de valores `NaN` e infinito en las características.
- Valor válido de `k`: entero positivo y no mayor que el número de muestras de entrenamiento.
- Disponibilidad de CUDA para las operaciones del motor.
- Contigüidad de los tensores internos después de la conversión.
- Coincidencia de dispositivo entre los tensores que participan en una operación.

Python debe generar errores comprensibles, indicando la condición incumplida y, cuando sea útil, los valores observados y esperados. C++ volverá a comprobar las precondiciones críticas relacionadas con tamaños, tipos, dispositivos y parámetros antes de lanzar kernels. Esta segunda comprobación protege la frontera nativa y evita depender exclusivamente de la validación de alto nivel.

## 7. Flujo de `fit()`

`fit()` seguirá este flujo:

1. Validar las dimensiones, tipos, valores finitos, tamaños y correspondencia entre muestras y etiquetas.
2. Convertir las características a `float32`, obtener una disposición contigua y preparar el dispositivo CUDA.
3. Ordenar y codificar las etiquetas enteras como valores `int64` contiguos desde cero.
4. Transferir a GPU los tensores que deban ser utilizados por el motor.
5. Almacenar en el objeto Python los tensores preparados, las etiquetas codificadas, `classes_` y los metadatos necesarios.

`fit()` no ejecuta entrenamiento: KNN no aprende parámetros numéricos. La operación prepara y conserva el conjunto de referencia para consultas posteriores.

En la versión inicial, el conjunto de entrenamiento completo debe caber en la VRAM disponible. No se contempla una representación parcial ni procesamiento out-of-core durante `fit()`.

## 8. Flujo de `kneighbors()`

La búsqueda exacta se organizará en dos niveles de partición:

1. Las consultas se procesan por lotes de tamaño limitado por `query_batch_size`.
2. Para cada lote de consultas, el conjunto de entrenamiento se recorre en bloques de tamaño limitado por `train_block_size`.
3. Para cada bloque se calculan las distancias euclidianas cuadradas.
4. Se obtiene un Top-K local por consulta dentro del bloque.
5. El Top-K local se fusiona con el Top-K global acumulado para esa consulta.
6. Se repite el proceso hasta revisar exactamente todas las muestras de entrenamiento.
7. Al terminar, se ordenan los `k` vecinos finales y se convierten sus distancias cuadradas a distancia euclidiana normal para la salida pública.

La división por lotes y bloques no puede cambiar los resultados. La fusión debe conservar el orden por distancia ascendente y, en igualdad de distancia, por menor índice de entrenamiento. La matriz completa `Q x N` no debe ser necesaria en la versión final, porque el consumo temporal de memoria debe estar acotado por el tamaño de los lotes, los bloques y los candidatos Top-K.

## 9. Flujo de `predict()`

`predict()` reutilizará la búsqueda de vecinos y seguirá este flujo:

1. Buscar los índices de los vecinos con `kneighbors()` o mediante la misma operación interna sin materializar resultados innecesarios.
2. Recuperar las etiquetas codificadas correspondientes a esos índices.
3. Realizar votación uniforme sobre las etiquetas de cada conjunto de vecinos.
4. Resolver los empates con las reglas deterministas definidas en este documento.
5. Recuperar las etiquetas originales mediante `classes_`.
6. Devolver las etiquetas predichas en la representación original.

Cuando sea posible, `predict()` debe evitar calcular raíces cuadradas, conservar matrices temporales que no necesita y realizar únicamente las transferencias necesarias para obtener las etiquetas finales. La búsqueda seguirá siendo exacta aunque se omitan de la ruta de predicción los datos de distancia que no participan en la votación.

## 10. Reglas de desempate

Las reglas de orden y desempate son parte del contrato funcional:

- Los vecinos se ordenan por distancia ascendente.
- Si dos vecinos tienen la misma distancia, se prioriza el menor índice de entrenamiento.
- Si dos o más etiquetas tienen el mismo número de votos, gana la etiqueta original menor.
- `classes_` se almacena ordenado.
- El comportamiento completo debe ser determinista bajo las mismas entradas y configuración.

El orden por índice debe aplicarse tanto al ordenar resultados finales como al fusionar candidatos parciales. No se debe depender de un orden incidental producido por el paralelismo de CUDA.

## 11. Capas

La separación inicial de módulos será la siguiente:

- `classifier.py` conserva el estado del clasificador y presenta la API pública.
- `_validation.py` valida las entradas públicas y los parámetros.
- `_labels.py` codifica y decodifica etiquetas y mantiene la correspondencia con `classes_`.
- `_ops.py` carga y encapsula los operadores registrados de PyTorch.
- `reference.py` contiene la referencia CPU en NumPy para pruebas y comparación.
- `knn_ops.cpp` registra y valida los operadores C++/CUDA y coordina sus lanzamientos.
- `distance_kernel.cu` calcula las distancias euclidianas cuadradas.
- `topk_kernel.cu` selecciona y fusiona vecinos locales y globales.
- `vote_kernel.cu` realiza la votación sobre etiquetas codificadas.

CUDA no conoce Python ni las etiquetas originales. C++ tampoco debe conservar el estado del clasificador; recibe tensores, parámetros y metadatos de una operación y devuelve resultados. Python es la única capa responsable del estado de alto nivel y de la traducción entre etiquetas originales y codificadas.

## 12. Operadores internos

Los operadores internos iniciales serán:

### `pairwise_squared_l2`

- **Entradas:** tensores contiguos de consultas y entrenamiento con formas compatibles `[Q, D]` y `[N, D]`, ambos en `float32` y en el mismo dispositivo CUDA.
- **Salida:** distancias euclidianas cuadradas para el bloque de consultas y el bloque de entrenamiento que se estén procesando.
- **Responsabilidad:** calcular la suma de diferencias al cuadrado por dimensión. No ordena vecinos, no aplica raíz cuadrada y no conoce etiquetas.

### `knn_search`

- **Entradas:** consultas, entrenamiento, `k`, `query_batch_size` y `train_block_size`, junto con los parámetros necesarios para recorrer los bloques.
- **Salidas:** índices `int64` y distancias cuadradas de los `k` vecinos por consulta, ordenados según las reglas de desempate.
- **Responsabilidad:** coordinar el recorrido exacto de todos los bloques, obtener Top-K local, fusionar Top-K global y garantizar que ninguna muestra quede sin revisar.

### `knn_vote`

- **Entradas:** índices de vecinos, etiquetas codificadas `int64`, `k` y el número de clases.
- **Salida:** etiquetas predichas codificadas `int64` por consulta.
- **Responsabilidad:** realizar votación uniforme, resolver empates mediante el menor valor de etiqueta codificada —equivalente al menor valor original porque `classes_` está ordenado— y producir una salida determinista.

Estos operadores son sin estado. Python conserva las referencias a los tensores de entrenamiento y etiquetas, y los pasa en cada llamada cuando corresponde.

## 13. Estrategia progresiva

La implementación se desarrollará en etapas verificables:

1. **Referencia CPU NumPy:** definir el comportamiento correcto, las formas, la ordenación y los desempates sin depender de CUDA.
2. **Matriz CUDA completa con Top-K temporal:** construir una versión sencilla que materialice temporalmente la matriz de distancias para validar la integración y los kernels.
3. **Top-K CUDA propio:** sustituir la selección temporal por una selección paralela de candidatos con reglas deterministas.
4. **Procesamiento por bloques:** introducir `query_batch_size` y `train_block_size`, fusionar resultados parciales y eliminar la necesidad de la matriz completa `Q x N`.
5. **Optimización avanzada:** mejorar acceso a memoria, ocupación, uso de memoria compartida y organización de kernels únicamente cuando existan pruebas de corrección y benchmarks que justifiquen el cambio.

La progresión mantiene una referencia funcional disponible en todas las etapas. El orden de prioridad es primero corrección y después rendimiento.

## 14. Pruebas

Las pruebas se dividirán entre entornos locales y CUDA:

- Las pruebas locales deben ejecutarse sin GPU y utilizar la referencia CPU en NumPy.
- Las pruebas CUDA se ejecutarán en Google Colab con GPU NVIDIA.
- La comparación primaria será contra NumPy, que definirá los valores esperados de distancia, índices y predicciones bajo las reglas del proyecto.
- La comparación secundaria será contra scikit-learn en casos controlados compatibles con su configuración.

La matriz de casos debe incluir, como mínimo:

- `k = 1`.
- `k = N`.
- Clasificación multiclase.
- Puntos duplicados.
- Empates de distancia.
- Varias consultas.
- Una sola consulta.
- Entradas inválidas y mensajes de error.

La tolerancia inicial para valores `float32` será `rtol = 1e-4` y `atol = 1e-5`. Los índices y las predicciones deben coincidir exactamente según las reglas de orden y desempate definidas. La tolerancia numérica aplica a las distancias, no a la identidad de los vecinos ni a las etiquetas resultantes.

## 15. Benchmarks

Los benchmarks deben distinguir las siguientes mediciones:

- Tiempo de kernel CUDA.
- Tiempo del motor, incluyendo la coordinación de operaciones en el dispositivo.
- Tiempo de extremo a extremo desde la API Python.
- Transferencias de entrada y salida medidas por separado.

Cada benchmark debe incluir calentamiento antes de registrar resultados, realizar varias repeticiones y reportar al menos la mediana. Las mediciones de CUDA deben usar sincronización explícita en los puntos necesarios para que la asincronía no oculte el tiempo real de ejecución.

Se reportarán consultas por segundo y *speedup* respecto a la referencia CPU elegida. Los experimentos variarán tamaños de dataset, dimensionalidad, número de consultas y valores de `k`.

Cada resultado debe registrar hardware, versión de CUDA, versión de Python, versión y configuración de PyTorch, configuración de compilación, tamaños de entrada y parámetros del experimento. No se afirmará una mejora sin benchmarks reproducibles y resultados revisados funcionalmente.

## 16. Limitaciones iniciales

La primera versión tendrá estas limitaciones explícitas:

- El conjunto de entrenamiento completo debe caber en VRAM.
- No habrá procesamiento out-of-core.
- No habrá búsqueda aproximada.
- No habrá regresión.
- No habrá ejecución multi-GPU.
- No habrá métricas arbitrarias; se utilizará la distancia euclidiana cuadrada definida.
- No habrá etiquetas de texto.
- No habrá autograd ni integración de gradientes.
- No habrá backend CPU dentro de la extensión CUDA.

El desarrollo local sin GPU se apoyará en la referencia NumPy y en las pruebas que no requieran la extensión CUDA. Esto no convierte la extensión CUDA en un motor CPU alternativo.

## 17. Arquitectura objetivo de carpetas

La estructura objetivo completa será:

```text
KNN-Cuda/
├── src/
│   ├── python/
│   │   └── knn_cuda/
│   │       ├── __init__.py       # API pública del paquete
│   │       ├── classifier.py     # CUDAKNNClassifier y estado Python
│   │       ├── _validation.py    # Validación de entradas y parámetros
│   │       ├── _labels.py        # Codificación y decodificación de etiquetas
│   │       ├── _ops.py           # Carga y encapsulado de operadores
│   │       └── reference.py      # Referencia CPU en NumPy
│   ├── cpp/
│   │   └── knn_ops.cpp           # Registro y validación de operadores
│   └── cuda/
│       ├── distance_kernel.cu    # Distancia euclidiana cuadrada
│       ├── topk_kernel.cu        # Selección y fusión Top-K
│       └── vote_kernel.cu        # Votación uniforme
├── tests/
│   ├── cpu/                      # Pruebas sin GPU y referencia NumPy
│   └── cuda/                     # Pruebas de extensión y kernels CUDA
├── benchmarks/                   # Medición, configuración y resultados
├── data/                         # Datos pequeños o referencias reproducibles
├── docs/
│   ├── architecture.md           # Arquitectura inicial
│   ├── requirements.md           # Requisitos iniciales
│   └── software_design.md        # Diseño de software
├── examples/                     # Ejemplos de uso de la API
├── notebooks/                    # Exploración y ejecución en Google Colab
└── README.md                     # Introducción y guía de inicio
```

La estructura separa la API Python, la extensión C++, los kernels CUDA, las pruebas CPU, las pruebas CUDA, los benchmarks, la documentación, los ejemplos y los notebooks. Los nombres de archivos reflejan responsabilidades; no implican que deban implementarse todos los módulos en una única etapa.

## 18. Reglas arquitectónicas

- La corrección tiene prioridad sobre el rendimiento.
- Cada capa tiene una única responsabilidad y debe comunicarse mediante contratos claros.
- Ninguna optimización se acepta sin evidencia de mejora mediante benchmarks reproducibles.
- El procesamiento por bloques no puede modificar los resultados frente al recorrido completo.
- Los operadores C++/CUDA no conservan estado.
- El código se escribe en inglés.
- La documentación se escribe en español.
- Cada cambio debe tener pruebas y revisión.

Estas reglas se aplican tanto a nuevas funcionalidades como a optimizaciones internas. Una modificación que mejore el tiempo pero altere la ordenación, los desempates, las distancias dentro de la tolerancia o las etiquetas predichas no cumple el diseño.
