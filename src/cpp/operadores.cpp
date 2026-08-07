#include "knn_cuda/operadores.h"

#include <torch/torch.h>

namespace {

void validar_tensor_cpu_float32(const torch::Tensor& tensor, const char* nombre) {
    TORCH_CHECK(tensor.device().is_cpu(), nombre, " debe estar en CPU");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, nombre, " debe tener dtype float32");
}

void validar_vector(const torch::Tensor& vector, const char* nombre) {
    validar_tensor_cpu_float32(vector, nombre);
    TORCH_CHECK(vector.dim() == 1, nombre, " debe ser unidimensional");
    TORCH_CHECK(vector.numel() > 0, nombre, " no debe estar vacio");
}

void validar_matriz(const torch::Tensor& matriz, const char* nombre) {
    validar_tensor_cpu_float32(matriz, nombre);
    TORCH_CHECK(matriz.dim() == 2, nombre, " debe ser bidimensional");
    TORCH_CHECK(matriz.numel() > 0, nombre, " no debe estar vacio");
    TORCH_CHECK(torch::isfinite(matriz).all().item<bool>(), nombre, " debe contener valores finitos");
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

torch::Tensor distancias_l2_cuadradas(
    const torch::Tensor& datos_consulta,
    const torch::Tensor& datos_entrenamiento
) {
    validar_matriz(datos_consulta, "datos_consulta");
    validar_matriz(datos_entrenamiento, "datos_entrenamiento");
    TORCH_CHECK(
        datos_consulta.size(1) == datos_entrenamiento.size(1),
        "datos_consulta y datos_entrenamiento deben tener la misma cantidad de caracteristicas"
    );

    const auto diferencias = datos_consulta.unsqueeze(1) - datos_entrenamiento.unsqueeze(0);
    return (diferencias * diferencias).sum(2);
}

}
