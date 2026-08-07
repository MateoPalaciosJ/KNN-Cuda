#include "knn_cuda/operadores.h"

#include <torch/library.h>

TORCH_LIBRARY(knn_cuda, modulo) {
    modulo.def(
        "verificar_backend_cpu() -> bool",
        TORCH_FN(knn_cuda::verificar_backend_cpu)
    );
    modulo.def("sumar_vectores(Tensor primer_vector, Tensor segundo_vector) -> Tensor");
}

TORCH_LIBRARY_IMPL(knn_cuda, CPU, modulo) {
    modulo.impl("sumar_vectores", &knn_cuda::sumar_vectores);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, modulo) {}
