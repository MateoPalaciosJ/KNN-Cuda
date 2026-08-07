#include "knn_cuda/operadores.h"

#include <torch/torch.h>

namespace {

void validar_vector(const torch::Tensor& vector, const char* nombre) {
    TORCH_CHECK(vector.device().is_cpu(), nombre, " debe estar en CPU");
    TORCH_CHECK(vector.scalar_type() == torch::kFloat32, nombre, " debe tener dtype float32");
    TORCH_CHECK(vector.dim() == 1, nombre, " debe ser unidimensional");
    TORCH_CHECK(vector.numel() > 0, nombre, " no debe estar vacio");
}

}

namespace knn_cuda {

bool verificar_backend_cpu() {
    return true;
}

torch::Tensor sumar_vectores(
    const torch::Tensor& primer_vector,
    const torch::Tensor& segundo_vector
) {
    validar_vector(primer_vector, "primer_vector");
    validar_vector(segundo_vector, "segundo_vector");
    TORCH_CHECK(
        primer_vector.size(0) == segundo_vector.size(0),
        "primer_vector y segundo_vector deben tener la misma longitud"
    );

    return torch::add(primer_vector, segundo_vector);
}

}
