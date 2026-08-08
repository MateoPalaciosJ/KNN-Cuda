#include "knn_cuda/operadores.h"

#ifdef KNN_CUDA_CON_CUDA
#include "knn_cuda/operadores_cuda.h"
#endif

#include <torch/library.h>

TORCH_LIBRARY(knn_cuda, modulo) {
    modulo.def(
        "verificar_backend_cpu() -> bool",
        TORCH_FN(knn_cuda::verificar_backend_cpu)
    );
    modulo.def("sumar_vectores(Tensor primer_vector, Tensor segundo_vector) -> Tensor");
    modulo.def(
        "distancias_l2_cuadradas(Tensor datos_consulta, Tensor datos_entrenamiento) -> Tensor"
    );
    modulo.def("seleccionar_top_k(Tensor distancias, int k) -> (Tensor, Tensor)");
    modulo.def("votacion_uniforme(Tensor etiquetas_vecinos) -> Tensor");
    modulo.def(
        "predecir_knn(Tensor datos_entrenamiento, Tensor etiquetas_entrenamiento, Tensor datos_consulta, int k) -> Tensor"
    );
}

TORCH_LIBRARY_IMPL(knn_cuda, CPU, modulo) {
    modulo.impl("sumar_vectores", &knn_cuda::sumar_vectores);
    modulo.impl("distancias_l2_cuadradas", &knn_cuda::distancias_l2_cuadradas);
    modulo.impl("seleccionar_top_k", &knn_cuda::seleccionar_top_k);
    modulo.impl("votacion_uniforme", &knn_cuda::votacion_uniforme);
    modulo.impl("predecir_knn", &knn_cuda::predecir_knn);
}

#ifdef KNN_CUDA_CON_CUDA
TORCH_LIBRARY_IMPL(knn_cuda, CUDA, modulo) {
    modulo.impl("distancias_l2_cuadradas", &knn_cuda::distancias_l2_cuadradas_cuda);
    modulo.impl("seleccionar_top_k", &knn_cuda::seleccionar_top_k_cuda);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, modulo) {}
