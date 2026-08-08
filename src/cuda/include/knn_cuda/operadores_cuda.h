#pragma once

#include <tuple>

#include <torch/extension.h>

namespace knn_cuda {

torch::Tensor distancias_l2_cuadradas_cuda(
    const torch::Tensor& datos_consulta,
    const torch::Tensor& datos_entrenamiento
);

std::tuple<torch::Tensor, torch::Tensor> seleccionar_top_k_cuda(
    const torch::Tensor& distancias,
    int64_t k
);

torch::Tensor votacion_uniforme_cuda(const torch::Tensor& etiquetas_vecinos);

torch::Tensor predecir_knn_cuda(
    const torch::Tensor& datos_entrenamiento,
    const torch::Tensor& etiquetas_entrenamiento,
    const torch::Tensor& datos_consulta,
    int64_t k
);

}
