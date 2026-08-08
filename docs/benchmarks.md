# Benchmarks y profiling

## Objetivo

Esta infraestructura establece una linea base reproducible antes de optimizar los backends CPU o CUDA

Los resultados miden implementaciones existentes y no sustituyen las pruebas funcionales

## Ejecucion

Desde una instalacion editable del proyecto se pueden ejecutar los scripts con

```powershell
python -m benchmarks.benchmark_operadores --escenario pequeno
python -m benchmarks.benchmark_pipeline --escenario pequeno
python -m benchmarks.benchmark_clasificador --escenario pequeno
```

`--escenario todos` ejecuta los tres tamaños y `--solo-cpu` evita medir CUDA en el benchmark de operadores

Los scripts detectan CUDA en runtime junto con los kernels CUDA registrados y omiten sus filas CUDA cuando alguna condicion no se cumple

## Escenarios reproducibles

Todos los datos se generan con la semilla fija `2026` y etiquetas enteras originales no consecutivas

| Escenario | N | Q | D | k |
| --- | ---: | ---: | ---: | ---: |
| pequeno | 128 | 16 | 8 | 5 |
| mediano | 512 | 64 | 32 | 10 |
| grande | 2048 | 128 | 64 | 20 |

N representa muestras de entrenamiento, Q consultas, D caracteristicas y k vecinos

## Metodologia CPU

Las mediciones CPU usan `perf_counter_ns`, tres iteraciones de calentamiento y quince repeticiones por defecto

Cada fila informa mediana, promedio y desviacion poblacional en milisegundos

La preparacion de datos y las comprobaciones de correccion ocurren fuera de las repeticiones medidas

## Metodologia CUDA

Las operaciones CUDA con tensores ya residentes se miden con `torch.cuda.Event` para respetar la ejecucion asincrona

Las transferencias y el tiempo extremo a extremo se miden con reloj de pared y sincronizacion antes y despues de cada repeticion

Las sincronizaciones existen solo en los scripts de benchmark y no forman parte de los operadores productivos

Cada escenario CUDA realiza calentamiento antes de registrar tiempos para excluir inicializacion y compilacion inicial de la medicion principal

## Cobertura

`benchmark_operadores` mide `distancias_l2_cuadradas`, `seleccionar_top_k`, `votacion_uniforme` y `predecir_knn` para NumPy, C++ CPU y CUDA cuando esta disponible

`benchmark_pipeline` separa preparacion de Tensor CPU, transferencia CPU a GPU, calculo CUDA residente, transferencia GPU a CPU y calculo con salida

`benchmark_clasificador` separa `ajustar` de `predecir` y mide el tiempo publico extremo a extremo de `ClasificadorKNNCUDA`

Antes de medir cada pipeline los scripts comparan una salida nativa contra la referencia NumPy

## Profiling de distancias CUDA

`perfilar_distancias_cuda` captura una llamada real a `torch.ops.knn_cuda.distancias_l2_cuadradas` despues de calentamiento y usa por defecto el escenario grande de la linea base

```powershell
python -m benchmarks.perfilar_distancias_cuda
python -m benchmarks.perfilar_distancias_cuda --traza distancias_cuda.json
```

El perfil muestra tablas de actividades CPU y CUDA ordenadas por tiempo propio junto con llamadas, memoria CUDA asignada, pico adicional y contigüidad de las entradas

La tabla puede mostrar las operaciones ATen de validacion como `isfinite`, `all` e `item` y el kernel propio con el nombre disponible en la traza

La opcion `--traza` exporta una traza compatible con Chrome Trace o Perfetto y no requiere ninguna dependencia adicional del proyecto

El profiler mide la llamada completa del operador y permite separar eventos registrados, pero no sustituye Nsight Compute para conocer ocupacion, registros por thread, spills, eficiencia de coalescing, contadores de memoria o conflictos de shared memory

## Memoria CUDA

El benchmark de operadores informa memoria CUDA asignada y pico adicional mediante las APIs de memoria de PyTorch

Las formas relevantes son la matriz de distancias `[Q, N]` y la mascara temporal `[Q, N]` utilizada por `seleccionar_top_k`

La medicion de memoria no debe interpretarse como un perfil completo del asignador ni como uso total del proceso

## Lectura de resultados

La salida muestra escenario, backend, etapa, mediana, promedio, desviacion y memoria CUDA cuando corresponde

Las aceleraciones comparan la mediana CUDA residente contra la mediana C++ CPU de la misma etapa

El tiempo de kernel no representa por si solo el tiempo percibido por usuarios que entregan y reciben `numpy.ndarray`

Todo resultado debe reportar sistema operativo, CPU, GPU si corresponde, RAM, version de Python, version de PyTorch, version de CUDA y configuracion de build

No deben compararse resultados de maquinas distintas sin ese contexto ni usarse estas mediciones como prueba de correccion

## Portabilidad

Los scripts no contienen rutas, hardware ni configuracion especifica de Google Colab

Colab puede utilizarse como entorno temporal de validacion CUDA y cualquier maquina con PyTorch y CUDA compatibles puede ejecutar las mismas ordenes
