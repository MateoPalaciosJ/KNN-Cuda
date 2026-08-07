#pragma once

#include <torch/extension.h>

namespace knn_cuda {

bool verificar_backend_cpu();

torch::Tensor sumar_vectores(
    const torch::Tensor& primer_vector,
    const torch::Tensor& segundo_vector
);

}
