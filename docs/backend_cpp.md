# Backend C++ de la Fase 2

## Objetivo

La Fase 2 crea la primera infraestructura nativa del proyecto mediante extensiones C++ de PyTorch

Su objetivo es validar la cadena Python → PyTorch → C++ con operadores CPU registrados en el dispatcher de PyTorch

NumPy conserva el papel de referencia primaria de corrección y el clasificador público continúa utilizando el motor de referencia durante esta fase

## Arquitectura

La ruta nativa oficial es Python → PyTorch → `CppExtension` → operadores propios → implementación CPU C++

La Fase 3 añadirá implementaciones CUDA dentro de esta misma arquitectura sin crear una segunda estrategia de binding ni cambiar la API pública

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

## Validaciones

`sumar_vectores` rechaza tensores fuera de CPU, dtype distinto de `float32`, dimensiones distintas de una, tensores vacíos y longitudes diferentes

No convierte tipos de forma silenciosa ni modifica los tensores de entrada

## Compilación

La extensión se compila durante la instalación editable mediante `CppExtension` y las herramientas oficiales de extensiones C++ de PyTorch

La instalación se ejecuta desde la raíz con `python -m pip install -e ".[test]"`

## Pruebas

Las pruebas del backend viven en `tests/cpu/test_backend_cpp.py`

La verificación completa ejecuta primero esas pruebas y después `python -m pytest tests/cpu -q` para confirmar que la Fase 1 no presenta regresiones

## Alcance pendiente

No se implementan operaciones KNN, integración de `ClasificadorKNNCUDA`, CUDA, kernels, GPU ni una API pública adicional

La Fase 3 reutilizará el registro de operadores, la configuración de extensión y la separación entre contratos, implementación y registro para añadir implementaciones CUDA
