# Backend CUDA de la Fase 3

## Objetivo

La Fase 3 añade implementaciones CUDA reales detrás de los esquemas de operadores ya aprobados en CPU

Las implementaciones son `distancias_l2_cuadradas`, `seleccionar_top_k`, `votacion_uniforme` y `predecir_knn` y conservan NumPy como referencia primaria de corrección

## Arquitectura

La ruta nativa se mantiene como Python → PyTorch → `CUDAExtension` → dispatcher de PyTorch → implementación CUDA

Los esquemas `knn_cuda::distancias_l2_cuadradas`, `knn_cuda::seleccionar_top_k`, `knn_cuda::votacion_uniforme` y `knn_cuda::predecir_knn` son únicos y el dispatcher selecciona la implementación CPU o CUDA según el dispositivo de los tensores de entrada

La implementación CPU C++ continúa disponible y no se modifica al añadir CUDA

## Estructura

- `src/cuda/include/knn_cuda/operadores_cuda.h` declara los contratos internos de CUDA
- `src/cuda/distancias_l2_cuadradas.cu` contiene el launcher y el kernel CUDA
- `src/cuda/seleccionar_top_k.cu` contiene el launcher y el kernel CUDA de selección
- `src/cuda/votacion_uniforme.cu` contiene el launcher y el kernel CUDA de votación
- `src/cuda/predecir_knn.cu` coordina el pipeline CUDA completo
- `src/cpp/registro.cpp` registra la implementación CUDA del mismo esquema cuando la extensión se compila con CUDA
- `setup.py` elige `CppExtension` o `CUDAExtension` según la disponibilidad de un toolkit compatible

## Kernel de distancias

El kernel usa una baldosa bidimensional de 16 consultas por 16 muestras con un bloque CUDA de 16 por 16 threads

Cada thread representa una salida `[q, n]` dentro de la baldosa y conserva un acumulador `float32` privado

El grid usa `ceil(N / 16)` bloques en X y `ceil(Q / 16)` bloques en Y, por lo que cubre dimensiones arbitrarias con comprobaciones de borde

Cada iteración carga un segmento de 32 características en dos arreglos de shared memory con forma conceptual `[32, 16]`, uno para consultas y otro para entrenamiento

Los 256 threads cargan las filas globales de cada entrada de forma contigua cuando el borde lo permite, reutilizan la baldosa de consultas entre 16 muestras y la baldosa de entrenamiento entre 16 consultas

La disposición de shared memory usa característica como primera dimensión para que los threads que recorren muestras lean posiciones contiguas y evita conflictos evitables durante el cálculo

Se usan dos `__syncthreads()` por segmento, uno después de cargar shared memory y otro antes de sobrescribirla en el siguiente segmento

Cada thread acumula segmentos y características en orden ascendente, no usa reducciones de D, no crea temporales globales `[Q, N, D]` y no usa Tensor Cores ni sincronización global

## Memoria y contigüidad

La salida se crea con PyTorch en el mismo dispositivo CUDA y tiene forma `[Q, N]`

El kernel acumula cada distancia directamente y no crea un temporal `[Q, N, D]`

La implementación acepta vistas no contiguas mediante `contiguous()` explícito para cada entrada solo cuando PyTorch requiere una copia, sin modificar los tensores originales

## Selección Top-K

`seleccionar_top_k` es el segundo operador KNN implementado para CUDA y recibe una matriz CUDA `float32` con forma `[Q, N]` junto con `k`

Devuelve distancias seleccionadas `float32` e índices `int64` con forma `[Q, k]` y conserva el orden ascendente de distancia

El kernel asigna un bloque de 256 threads a cada consulta y repite una reducción paralela para cada posición de salida

Cada thread recorre una parte de la fila y propone su mejor candidato local, después cada warp aplica un árbol fijo de `__shfl_down_sync` sobre pares distancia e índice

Los ocho candidatos de warp se almacenan en shared memory y el warp cero aplica la misma reducción fija para obtener el ganador final

Un candidato es mejor cuando su distancia es menor o cuando la distancia es igual y su índice original es menor, por lo que los empates son deterministas

Un tensor temporal booleano `[Q, N]` registra los índices ya elegidos y evita seleccionar el mismo vecino dos veces sin modificar `distancias`

La reducción usa 32 bytes para ocho distancias y 64 bytes para ocho índices `int64`, en lugar de los arreglos compartidos de 256 candidatos de la primera versión

La implementación acepta vistas no contiguas mediante `contiguous()` interno cuando es necesario y lanza el kernel en el stream CUDA actual

La operación rechaza tensores fuera de CUDA, dtype distinto de `float32`, dimensiones distintas de dos, matrices vacías, valores no finitos y valores de `k` fuera del intervalo de `1` a `N`

La selección todavía explora la fila completa por cada uno de los `k` vecinos y mantiene la dependencia secuencial de la máscara, por lo que la complejidad global no cambia

Para casos experimentales compatibles, la selección usa una ruta alternativa de un bloque por consulta con top-k local por warp en shared memory y un merge final dentro del mismo bloque

La ruta experimental requiere `k <= 32`, al menos `k` elementos por cada uno de los ocho segmentos y una reserva dinámica de shared memory no mayor que 16 KiB ni que el límite del dispositivo CUDA actual

Cada distancia se carga una vez a shared memory, cada warp selecciona sus candidatos locales y el warp cero fusiona los ocho conjuntos de candidatos con el mismo comparador distancia e índice

La ruta evita la máscara global `[Q, N]` y usa una máscara local por bloque, sin temporales globales adicionales

Los tamaños que no cumplen esas condiciones usan el kernel anterior con máscara global, incluidos inicialmente los casos como `k = N`

## Votación uniforme

`votacion_uniforme` es el tercer operador KNN implementado para CUDA y recibe etiquetas enteras originales con forma `[Q, K]`

El kernel asigna un bloque de 256 threads a cada consulta y cada thread cuenta las apariciones de las etiquetas que procesa dentro de la fila

La reducción de shared memory compara explícitamente los pares conteo y etiqueta para elegir mayor conteo y menor etiqueta en empate

Las posiciones repetidas de una misma etiqueta pueden proponer el mismo candidato y no alteran el resultado porque conservan el mismo conteo y etiqueta

No se codifican etiquetas ni se reserva memoria según el valor de las etiquetas, por lo que se admiten valores negativos, no consecutivos y grandes

La operación admite dtypes enteros compatibles, incluidos `int32` e `int64`, y conserva el dtype de entrada en las predicciones

No crea temporales globales adicionales, acepta vistas no contiguas mediante `contiguous()` interno cuando es necesario y usa el stream CUDA actual

Esta primera versión tiene complejidad `O(Q × K²)` y prioriza corrección, determinismo y claridad antes de optimizaciones futuras

## Pipeline KNN CUDA

`predecir_knn` completa el pipeline KNN CUDA y reutiliza `distancias_l2_cuadradas`, `seleccionar_top_k` y `votacion_uniforme` sin reimplementar sus responsabilidades

Primero valida las etiquetas originales, su correspondencia con las muestras, `k` y la coincidencia de dispositivos CUDA antes de calcular distancias

Después calcula distancias `[Q, N]`, selecciona índices `[Q, k]`, obtiene `etiquetas_vecinos` mediante indexado ATen en GPU y aplica la votación uniforme

El indexado conserva el dtype entero de `etiquetas_entrenamiento` y no realiza transferencias CPU o GPU ni requiere un kernel auxiliar de gather

Los operadores se encadenan en el stream CUDA actual y los temporales naturales son distancias, selección e `etiquetas_vecinos`

El pipeline hereda el desempate de distancia por índice menor y el desempate de votos por etiqueta menor

NumPy sigue siendo la referencia primaria y las pruebas comparan las predicciones CUDA con NumPy y con C++ CPU

## Streams y errores

El launcher usa el stream CUDA actual de PyTorch para el dispositivo de entrada y no llama a `cudaDeviceSynchronize`

Después del lanzamiento usa `C10_CUDA_KERNEL_LAUNCH_CHECK` para informar errores de lanzamiento sin ocultarlos

La comprobación de valores finitos forma parte de la validación del contrato y puede requerir sincronización de ese tensor antes del kernel

## Validaciones

La implementación CUDA requiere tensores CUDA `float32`, bidimensionales, no vacíos, finitos, en el mismo dispositivo y con la misma cantidad de características

No convierte dtype ni mueve datos entre CPU y CUDA de forma automática

## Instalación y build portable

PyTorch es una dependencia de runtime y también de build porque `setup.py` utiliza `CppExtension` y `CUDAExtension`

Una extensión nativa de PyTorch debe compilarse con la misma distribución de PyTorch que se utilizará durante la ejecución para conservar compatibilidad ABI

El aislamiento PEP 517 crea un entorno temporal y puede resolver otro PyTorch, incluso cuando el entorno principal ya tiene una distribución compatible

Para evitar esa mezcla, la instalación oficial desde fuente requiere instalar primero el PyTorch apropiado y compilar KNN-Cuda contra ese mismo entorno con `--no-build-isolation`

`build-system.requires` contiene solo setuptools para que pip no instale un PyTorch temporal durante el build aislado

Si se intenta compilar desde fuente sin PyTorch instalado, `setup.py` falla de forma clara antes de construir una extensión incompatible

La configuración compara la versión de `nvcc` encontrada desde `CUDA_HOME` con `torch.version.cuda` del PyTorch preinstalado

Si ambas versiones coinciden, `setup.py` usa `CUDAExtension`, compila la fuente `.cu` y registra el kernel CUDA

Si PyTorch no tiene soporte CUDA, usa `CppExtension` para conservar el backend CPU sin requerir CUDA Toolkit ni GPU

Si PyTorch tiene soporte CUDA pero no existe un toolkit utilizable, informa explícitamente que compilará solo CPU

Si las versiones CUDA de PyTorch y del toolkit no coinciden, el build falla de forma explícita y no genera una extensión que pueda aparentar compatibilidad

La detección de build no depende de `torch.cuda.is_available()`, que indica disponibilidad de runtime y no de toolkit

### Instalación CPU

La instalación CPU no requiere GPU ni CUDA Toolkit

```text
instalar una distribución CPU de PyTorch compatible
python -m pip install --no-build-isolation -e ".[test]"
python -m pytest tests/cpu -q
```

La instalación editable deja el paquete disponible sin modificar `PYTHONPATH`

### Instalación CUDA

Para compilar CUDA, el entorno de build debe resolver una distribución de PyTorch compatible con el CUDA Toolkit instalado

```text
instalar una distribución CUDA de PyTorch compatible con el sistema
python -m pip install --no-build-isolation -e ".[test]"
python -m pytest tests/cuda -q
```

Este flujo no depende de Google Colab y funciona en cualquier sistema que tenga un PyTorch y un CUDA Toolkit compatibles

La distribución futura mediante wheels podrá publicar artefactos compilados para combinaciones concretas de sistema operativo, Python, PyTorch y CUDA sin requerir compilación local

Un modo explícito de build `AUTO`, `CPU` y `CUDA` es una mejora futura recomendable para evitar que una intención explícita de CUDA termine en un build CPU

## Pruebas

Las pruebas CUDA viven en `tests/cuda/` y se omiten limpiamente cuando el runtime CUDA no está disponible

En un entorno CUDA válido comparan el kernel con la referencia NumPy y con la implementación C++ CPU usando `rtol=1e-4` y `atol=1e-5`

Para validar en Google Colab o en cualquier máquina CUDA compatible se debe comprobar `torch.version.cuda`, `torch.cuda.is_available()` y `CUDA_HOME`, reinstalar el proyecto editable y ejecutar `python -m pytest tests/cuda -q`

Google Colab es un entorno de desarrollo y validación opcional, no una dependencia del código ni de la instalación

## Limitaciones pendientes

`distancias_l2_cuadradas`, `seleccionar_top_k`, `votacion_uniforme` y `predecir_knn` tienen implementación CUDA

`ClasificadorKNNCUDA` usa `"cpu"`, `"cuda"` y `"auto"` mediante el Dispatcher. `"cuda"` exige runtime y kernels CUDA disponibles, mientras `"auto"` elige CUDA solo cuando ambos existen y usa CPU en cualquier otro caso

`dispositivo_efectivo_` conserva el dispositivo resuelto durante `ajustar()` y evita cambiar de backend en cada predicción

Los benchmarks y las optimizaciones de kernel permanecen pendientes
