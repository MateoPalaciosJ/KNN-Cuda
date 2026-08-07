# Backend CUDA de la Fase 3

## Objetivo

La Fase 3 añade implementaciones CUDA reales detrás de los esquemas de operadores ya aprobados en CPU

La primera implementación es `distancias_l2_cuadradas` y conserva NumPy como referencia primaria de corrección

## Arquitectura

La ruta nativa se mantiene como Python → PyTorch → `CUDAExtension` → dispatcher de PyTorch → implementación CUDA

El esquema `knn_cuda::distancias_l2_cuadradas` es único y el dispatcher selecciona la implementación CPU o CUDA según el dispositivo de los tensores de entrada

La implementación CPU C++ continúa disponible y no se modifica al añadir CUDA

## Estructura

- `src/cuda/include/knn_cuda/operadores_cuda.h` declara los contratos internos de CUDA
- `src/cuda/distancias_l2_cuadradas.cu` contiene el launcher y el kernel CUDA
- `src/cpp/registro.cpp` registra la implementación CUDA del mismo esquema cuando la extensión se compila con CUDA
- `setup.py` elige `CppExtension` o `CUDAExtension` según la disponibilidad de un toolkit compatible

## Kernel de distancias

El kernel asigna un hilo a cada par consulta y muestra de entrenamiento

Un índice lineal se transforma en `indice_consulta` e `indice_muestra`, el hilo recorre las características, acumula la diferencia al cuadrado en `float32` y escribe una sola posición de salida `[Q, N]`

La primera versión usa 256 hilos por bloque y calcula el número de bloques como el redondeo hacia arriba de `Q × N / 256`

No utiliza shared memory, tiling, vectorización, Tensor Cores ni sincronización global

## Memoria y contigüidad

La salida se crea con PyTorch en el mismo dispositivo CUDA y tiene forma `[Q, N]`

El kernel acumula cada distancia directamente y no crea un temporal `[Q, N, D]`

La implementación acepta vistas no contiguas mediante `contiguous()` explícito para cada entrada solo cuando PyTorch requiere una copia, sin modificar los tensores originales

## Streams y errores

El launcher usa el stream CUDA actual de PyTorch para el dispositivo de entrada y no llama a `cudaDeviceSynchronize`

Después del lanzamiento usa `C10_CUDA_KERNEL_LAUNCH_CHECK` para informar errores de lanzamiento sin ocultarlos

La comprobación de valores finitos forma parte de la validación del contrato y puede requerir sincronización de ese tensor antes del kernel

## Validaciones

La implementación CUDA requiere tensores CUDA `float32`, bidimensionales, no vacíos, finitos, en el mismo dispositivo y con la misma cantidad de características

No convierte dtype ni mueve datos entre CPU y CUDA de forma automática

## Build portable

La compilación comprueba `CUDA_HOME` y que la distribución de PyTorch tenga soporte CUDA para detectar un toolkit apto para compilar

Si ambos requisitos existen, `setup.py` usa `CUDAExtension`, compila la fuente `.cu` y registra el kernel CUDA

Si faltan, usa `CppExtension` y conserva el backend CPU sin requerir CUDA Toolkit ni GPU

La detección de build no depende de `torch.cuda.is_available()`, que indica disponibilidad de runtime y no de toolkit

## Pruebas

Las pruebas CUDA viven en `tests/cuda/test_distancias_l2_cuadradas_cuda.py` y se omiten limpiamente cuando el runtime CUDA no está disponible

En un entorno CUDA válido comparan el kernel con la referencia NumPy y con la implementación C++ CPU usando `rtol=1e-4` y `atol=1e-5`

Para validar en Google Colab o en cualquier máquina CUDA compatible se debe comprobar `torch.version.cuda`, `torch.cuda.is_available()` y `CUDA_HOME`, reinstalar el proyecto editable y ejecutar `python -m pytest tests/cuda -q`

Google Colab es un entorno de desarrollo y validación opcional, no una dependencia del código ni de la instalación

## Limitaciones pendientes

Solo `distancias_l2_cuadradas` tiene implementación CUDA

`seleccionar_top_k`, `votacion_uniforme`, `predecir_knn`, la integración de `ClasificadorKNNCUDA`, los benchmarks y las optimizaciones de kernel permanecen pendientes
