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

Las pruebas CUDA viven en `tests/cuda/test_distancias_l2_cuadradas_cuda.py` y se omiten limpiamente cuando el runtime CUDA no está disponible

En un entorno CUDA válido comparan el kernel con la referencia NumPy y con la implementación C++ CPU usando `rtol=1e-4` y `atol=1e-5`

Para validar en Google Colab o en cualquier máquina CUDA compatible se debe comprobar `torch.version.cuda`, `torch.cuda.is_available()` y `CUDA_HOME`, reinstalar el proyecto editable y ejecutar `python -m pytest tests/cuda -q`

Google Colab es un entorno de desarrollo y validación opcional, no una dependencia del código ni de la instalación

## Limitaciones pendientes

Solo `distancias_l2_cuadradas` tiene implementación CUDA

`seleccionar_top_k`, `votacion_uniforme`, `predecir_knn`, la integración de `ClasificadorKNNCUDA`, los benchmarks y las optimizaciones de kernel permanecen pendientes
