#pragma once

#include <torch/extension.h>

namespace knn_cuda {

torch::Tensor distancias_l2_cuadradas_cuda(
    const torch::Tensor& datos_consulta,
    const torch::Tensor& datos_entrenamiento
);

}
