#include "knn_cuda/operadores_cuda.h"

#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int hilos_por_bloque = 256;

void validar_matriz_cuda(const torch::Tensor& matriz, const char* nombre) {
    TORCH_CHECK(matriz.is_cuda(), nombre, " debe estar en CUDA");
    TORCH_CHECK(matriz.scalar_type() == torch::kFloat32, nombre, " debe tener dtype float32");
    TORCH_CHECK(matriz.dim() == 2, nombre, " debe ser bidimensional");
    TORCH_CHECK(matriz.numel() > 0, nombre, " no debe estar vacio");
    TORCH_CHECK(torch::isfinite(matriz).all().item<bool>(), nombre, " debe contener valores finitos");
}

__global__ void calcular_distancias_l2_cuadradas_kernel(
    const float* datos_consulta,
    const float* datos_entrenamiento,
    float* distancias_cuadradas,
    int64_t numero_consultas,
    int64_t numero_muestras,
    int64_t numero_caracteristicas
) {
    const int64_t indice_lineal =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t numero_distancias = numero_consultas * numero_muestras;

    if (indice_lineal >= numero_distancias) {
        return;
    }

    const int64_t indice_consulta = indice_lineal / numero_muestras;
    const int64_t indice_muestra = indice_lineal % numero_muestras;
    float distancia_cuadrada = 0.0F;

    for (
        int64_t indice_caracteristica = 0;
        indice_caracteristica < numero_caracteristicas;
        ++indice_caracteristica
    ) {
        const float diferencia =
            datos_consulta[indice_consulta * numero_caracteristicas + indice_caracteristica] -
            datos_entrenamiento[indice_muestra * numero_caracteristicas + indice_caracteristica];
        distancia_cuadrada += diferencia * diferencia;
    }

    distancias_cuadradas[indice_lineal] = distancia_cuadrada;
}

}

namespace knn_cuda {

torch::Tensor distancias_l2_cuadradas_cuda(
    const torch::Tensor& datos_consulta,
    const torch::Tensor& datos_entrenamiento
) {
    validar_matriz_cuda(datos_consulta, "datos_consulta");
    validar_matriz_cuda(datos_entrenamiento, "datos_entrenamiento");
    TORCH_CHECK(
        datos_consulta.device() == datos_entrenamiento.device(),
        "datos_consulta y datos_entrenamiento deben estar en el mismo dispositivo CUDA"
    );
    TORCH_CHECK(
        datos_consulta.size(1) == datos_entrenamiento.size(1),
        "datos_consulta y datos_entrenamiento deben tener la misma cantidad de caracteristicas"
    );

    const int64_t numero_consultas = datos_consulta.size(0);
    const int64_t numero_muestras = datos_entrenamiento.size(0);
    const int64_t numero_caracteristicas = datos_consulta.size(1);
    TORCH_CHECK(
        numero_consultas <= std::numeric_limits<int64_t>::max() / numero_muestras,
        "el numero de distancias excede el rango admitido"
    );
    const int64_t numero_distancias = numero_consultas * numero_muestras;
    const int64_t numero_bloques =
        (numero_distancias - 1) / hilos_por_bloque + 1;
    TORCH_CHECK(
        numero_bloques <= std::numeric_limits<unsigned int>::max(),
        "el numero de bloques excede el rango admitido"
    );

    const c10::cuda::CUDAGuard guard(datos_consulta.device());
    const auto datos_consulta_contiguos = datos_consulta.contiguous();
    const auto datos_entrenamiento_contiguos = datos_entrenamiento.contiguous();
    auto distancias_cuadradas = torch::empty(
        {numero_consultas, numero_muestras}, datos_consulta.options()
    );

    calcular_distancias_l2_cuadradas_kernel<<<
        static_cast<unsigned int>(numero_bloques),
        hilos_por_bloque,
        0,
        c10::cuda::getCurrentCUDAStream().stream()
    >>>(
        datos_consulta_contiguos.data_ptr<float>(),
        datos_entrenamiento_contiguos.data_ptr<float>(),
        distancias_cuadradas.data_ptr<float>(),
        numero_consultas,
        numero_muestras,
        numero_caracteristicas
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return distancias_cuadradas;
}

}
