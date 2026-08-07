#pragma once

#include <torch/extension.h>

namespace knn_cuda {

bool verificar_backend_cpu();

torch::Tensor sumar_vectores(
    const torch::Tensor& primer_vector,
    const torch::Tensor& segundo_vector
);

torch::Tensor distancias_l2_cuadradas(
    const torch::Tensor& datos_consulta,
    const torch::Tensor& datos_entrenamiento
);

}
