#include "knn_cuda/operadores_cuda.h"

#include <c10/cuda/CUDAGuard.h>

namespace {

void validar_etiquetas_entrenamiento_cuda(
    const torch::Tensor& etiquetas_entrenamiento
) {
    TORCH_CHECK(
        etiquetas_entrenamiento.is_cuda(),
        "etiquetas_entrenamiento debe estar en CUDA"
    );
    TORCH_CHECK(
        etiquetas_entrenamiento.scalar_type() != torch::kBool,
        "etiquetas_entrenamiento no debe tener dtype booleano"
    );
    TORCH_CHECK(
        c10::isIntegralType(etiquetas_entrenamiento.scalar_type(), false),
        "etiquetas_entrenamiento debe tener dtype entero"
    );
    TORCH_CHECK(
        etiquetas_entrenamiento.dim() == 1,
        "etiquetas_entrenamiento debe ser unidimensional"
    );
    TORCH_CHECK(
        etiquetas_entrenamiento.numel() > 0,
        "etiquetas_entrenamiento no debe estar vacio"
    );
}

}

namespace knn_cuda {

torch::Tensor predecir_knn_cuda(
    const torch::Tensor& datos_entrenamiento,
    const torch::Tensor& etiquetas_entrenamiento,
    const torch::Tensor& datos_consulta,
    int64_t k
) {
    validar_etiquetas_entrenamiento_cuda(etiquetas_entrenamiento);
    TORCH_CHECK(
        datos_entrenamiento.dim() >= 1,
        "datos_entrenamiento debe tener al menos una dimension"
    );
    TORCH_CHECK(
        datos_entrenamiento.is_cuda(),
        "datos_entrenamiento debe estar en CUDA"
    );
    TORCH_CHECK(datos_consulta.is_cuda(), "datos_consulta debe estar en CUDA");
    TORCH_CHECK(
        datos_entrenamiento.device() == etiquetas_entrenamiento.device(),
        "datos_entrenamiento y etiquetas_entrenamiento deben estar en el mismo dispositivo CUDA"
    );
    TORCH_CHECK(
        datos_entrenamiento.device() == datos_consulta.device(),
        "datos_entrenamiento y datos_consulta deben estar en el mismo dispositivo CUDA"
    );
    TORCH_CHECK(
        etiquetas_entrenamiento.size(0) == datos_entrenamiento.size(0),
        "etiquetas_entrenamiento debe coincidir con las muestras de datos_entrenamiento"
    );
    TORCH_CHECK(k >= 1, "k debe ser mayor o igual a 1");
    TORCH_CHECK(
        k <= datos_entrenamiento.size(0),
        "k no debe ser mayor que el numero de muestras de datos_entrenamiento"
    );

    const c10::cuda::CUDAGuard guard(datos_entrenamiento.device());
    const auto distancias = distancias_l2_cuadradas_cuda(
        datos_consulta, datos_entrenamiento
    );
    const auto resultado_seleccion = seleccionar_top_k_cuda(distancias, k);
    const auto& indices_seleccionados = std::get<1>(resultado_seleccion);
    const auto etiquetas_vecinos = etiquetas_entrenamiento.index({indices_seleccionados});

    return votacion_uniforme_cuda(etiquetas_vecinos);
}

}
