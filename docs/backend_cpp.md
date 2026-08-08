# Backend C++ de la Fase 2

## Objetivo

La Fase 2 crea la primera infraestructura nativa del proyecto mediante extensiones C++ de PyTorch

Su objetivo es validar la cadena Python → PyTorch → C++ con operadores CPU registrados en el dispatcher de PyTorch

NumPy conserva el papel de referencia primaria de corrección y `ClasificadorKNNCUDA` utiliza el backend C++ CPU desde la primera integración de Fase 4

## Arquitectura

La ruta nativa oficial es Python → PyTorch → `CppExtension` → operadores propios → implementación CPU C++

La Fase 3 añadió implementaciones CUDA dentro de esta misma arquitectura sin crear una segunda estrategia de binding ni cambiar la API pública

El módulo interno compilado se llama `knn_cuda._backend_cpp` y su importación registra los operadores de esta fase

## Archivos del backend

- `src/cpp/include/knn_cuda/operadores.h` declara los contratos de los operadores C++
- `src/cpp/operadores.cpp` implementa los operadores CPU y sus validaciones
- `src/cpp/registro.cpp` registra los esquemas e implementaciones en el dispatcher de PyTorch
- `setup.py` configura `CppExtension` y `BuildExtension`

La implementación matemática no se mezcla con el registro de operadores

## Operadores de infraestructura

`verificar_backend_cpu` devuelve `True` y confirma que la extensión compilada puede importarse y registrar un operador sin argumentos

`sumar_vectores` recibe dos tensores CPU `float32` unidimensionales, no vacíos y de la misma longitud, y devuelve un tensor `float32` con la suma elemento a elemento

`sumar_vectores` no forma parte de KNN y existe solo para validar recepción de tensores, lectura C++, creación de la salida, retorno hacia Python y errores de contrato

## Primera operación KNN

`distancias_l2_cuadradas` es la primera operación KNN real implementada en C++ durante la Fase 2

Recibe `datos_consulta` con forma `[Q, D]` y `datos_entrenamiento` con forma `[N, D]`, ambos tensores CPU `float32` bidimensionales, no vacíos y con valores finitos

Devuelve un tensor CPU `float32` con forma `[Q, N]` donde cada elemento es la suma de las diferencias al cuadrado entre una consulta y una muestra de entrenamiento

La implementación utiliza operaciones ATen para expresar directamente la resta con broadcasting, el cuadrado y la suma sobre las características, sin aplicar raíz cuadrada ni requerir contigüidad

NumPy sigue siendo la referencia primaria de corrección y las pruebas comparan esta operación con `distancias_l2_cuadradas` del motor de referencia

El esquema se define una sola vez y actualmente tiene implementaciones CPU y CUDA bajo el mismo Dispatcher

La implementación CPU no incorpora optimizaciones, bloques ni paralelismo manual

`seleccionar_top_k` es la segunda operación KNN real implementada en C++ durante la Fase 2

Recibe `distancias` con forma `[Q, N]` y un valor entero `k`, y devuelve `distancias_seleccionadas` `float32` e `indices_seleccionados` `int64`, ambos con forma `[Q, k]`

La operación ordena cada fila de forma ascendente mediante una ordenación estable de ATen y conserva el índice original menor cuando existen distancias iguales

La implementación se ejecuta solo en CPU, no requiere contigüidad y se compara con `seleccionar_top_k` de NumPy como referencia primaria

CUDA ya registra su propia implementación bajo el mismo esquema

`votacion_uniforme` es la tercera operación KNN real implementada en C++ durante la Fase 2

Recibe directamente `etiquetas_vecinos` originales con forma `[Q, K]`, aplica un voto por vecino y devuelve una predicción por consulta con el mismo dtype entero de entrada

La operación ordena las etiquetas de cada consulta, cuenta sus valores distintos y resuelve los empates por la etiqueta original numéricamente menor

NumPy sigue siendo la referencia primaria y CPU C++ convive con CUDA bajo los mismos esquemas

## Validaciones

`sumar_vectores` rechaza tensores fuera de CPU, dtype distinto de `float32`, dimensiones distintas de una, tensores vacíos y longitudes diferentes

`distancias_l2_cuadradas` rechaza tensores fuera de CPU, dtype distinto de `float32`, dimensiones distintas de dos, tensores vacíos, valores no finitos y cantidades de características diferentes

`seleccionar_top_k` rechaza tensores fuera de CPU, dtype distinto de `float32`, tensores que no sean bidimensionales, tensores vacíos, valores `NaN`, valores infinitos, `k` menor que `1` y `k` mayor que el número de columnas

`seleccionar_top_k` no convierte tipos silenciosamente, no modifica la matriz de distancias y no requiere contigüidad

Los empates de `seleccionar_top_k` se resuelven preservando el menor índice original mediante ordenación estable

`votacion_uniforme` rechaza tensores fuera de CPU, tensores que no sean bidimensionales, tensores vacíos, dtype no entero y dtype booleano

`votacion_uniforme` no convierte tipos silenciosamente, no modifica `etiquetas_vecinos` y no requiere contigüidad

No convierte tipos de forma silenciosa ni modifica los tensores de entrada

## Pipeline KNN C++ CPU

`predecir_knn` completa el pipeline KNN funcional interno en C++ CPU

Recibe `datos_entrenamiento`, `etiquetas_entrenamiento`, `datos_consulta` y `k`, valida anticipadamente las etiquetas, su correspondencia con las muestras y el rango de `k` antes de calcular distancias

Después reutiliza `distancias_l2_cuadradas`, `seleccionar_top_k` y `votacion_uniforme` sin reimplementar sus responsabilidades

Las etiquetas de vecinos se obtienen mediante indexado de `etiquetas_entrenamiento` con `indices_seleccionados`

La operación conserva los desempates deterministas de distancia y votación, prioriza la corrección sobre el rendimiento y se compara de forma integral con la referencia NumPy

El pipeline C++ continúa siendo interno y `ClasificadorKNNCUDA` lo invoca mediante el Dispatcher para el dispositivo CPU

La implementación CUDA actual reutiliza el mismo esquema de operador

## Compilación desde fuente

Una extensión de PyTorch debe compilarse con la misma distribución de PyTorch que la cargará durante la ejecución para conservar compatibilidad ABI

Por esta razón, PyTorch debe instalarse antes de compilar KNN-Cuda desde fuente y la instalación editable oficial usa el entorno actual sin aislamiento de build

La instalación CPU se ejecuta desde la raíz con `python -m pip install --no-build-isolation -e ".[test]"`

Esta política conserva `CppExtension` para sistemas sin CUDA y evita compilar la extensión contra un PyTorch temporal diferente

## Pruebas

Las pruebas del backend viven en `tests/cpu/test_backend_cpp.py`

La verificación completa ejecuta primero esas pruebas y después `python -m pytest tests/cpu -q` para confirmar que la Fase 1 no presenta regresiones

## Alcance pendiente

`distancias_l2_cuadradas` ya está implementada en C++ CPU

`seleccionar_top_k`, `votacion_uniforme` y `predecir_knn` ya están implementados en C++ CPU

`ClasificadorKNNCUDA` utiliza el backend C++ CPU y mantiene entradas y salidas públicas NumPy

Los operadores nativos actuales tienen implementaciones CUDA validadas. La integración CUDA y `"auto"` de `ClasificadorKNNCUDA` continúan pendientes

El temporal conceptual `[Q, N, D]` de `distancias_l2_cuadradas` CPU continúa siendo un riesgo de memoria pendiente para fases de optimización

No existe una API pública adicional

La Fase 3 reutilizó el registro de operadores, la configuración de extensión y la separación entre contratos, implementación y registro para añadir implementaciones CUDA
