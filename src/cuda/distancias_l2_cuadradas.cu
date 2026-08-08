#include "knn_cuda/operadores_cuda.h"

#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int tamanio_baldosa_consultas = 16;
constexpr int tamanio_baldosa_muestras = 16;
constexpr int caracteristicas_por_baldosa = 32;
constexpr int hilos_por_bloque =
    tamanio_baldosa_consultas * tamanio_baldosa_muestras;

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
    __shared__ float consultas_compartidas[
        caracteristicas_por_baldosa
    ][tamanio_baldosa_consultas];
    __shared__ float entrenamiento_compartido[
        caracteristicas_por_baldosa
    ][tamanio_baldosa_muestras];

    const int indice_hilo = threadIdx.y * blockDim.x + threadIdx.x;
    const int64_t indice_consulta =
        static_cast<int64_t>(blockIdx.y) * tamanio_baldosa_consultas + threadIdx.y;
    const int64_t indice_muestra =
        static_cast<int64_t>(blockIdx.x) * tamanio_baldosa_muestras + threadIdx.x;
    const bool salida_valida =
        indice_consulta < numero_consultas && indice_muestra < numero_muestras;
    float distancia_cuadrada = 0.0F;

    for (
        int64_t indice_caracteristica_base = 0;
        indice_caracteristica_base < numero_caracteristicas;
        indice_caracteristica_base += caracteristicas_por_baldosa
    ) {
        const int64_t caracteristicas_restantes =
            numero_caracteristicas - indice_caracteristica_base;
        const int caracteristicas_en_baldosa =
            caracteristicas_restantes < caracteristicas_por_baldosa
            ? static_cast<int>(caracteristicas_restantes)
            : caracteristicas_por_baldosa;

        for (
            int indice_carga = indice_hilo;
            indice_carga < tamanio_baldosa_consultas * caracteristicas_en_baldosa;
            indice_carga += hilos_por_bloque
        ) {
            const int indice_consulta_local =
                indice_carga / caracteristicas_en_baldosa;
            const int indice_caracteristica_local =
                indice_carga % caracteristicas_en_baldosa;
            const int64_t indice_consulta_global =
                static_cast<int64_t>(blockIdx.y) * tamanio_baldosa_consultas +
                indice_consulta_local;
            const int64_t indice_caracteristica_global =
                indice_caracteristica_base + indice_caracteristica_local;

            consultas_compartidas[indice_caracteristica_local][indice_consulta_local] =
                indice_consulta_global < numero_consultas
                ? datos_consulta[
                    indice_consulta_global * numero_caracteristicas +
                    indice_caracteristica_global
                ]
                : 0.0F;
        }

        for (
            int indice_carga = indice_hilo;
            indice_carga < tamanio_baldosa_muestras * caracteristicas_en_baldosa;
            indice_carga += hilos_por_bloque
        ) {
            const int indice_muestra_local =
                indice_carga / caracteristicas_en_baldosa;
            const int indice_caracteristica_local =
                indice_carga % caracteristicas_en_baldosa;
            const int64_t indice_muestra_global =
                static_cast<int64_t>(blockIdx.x) * tamanio_baldosa_muestras +
                indice_muestra_local;
            const int64_t indice_caracteristica_global =
                indice_caracteristica_base + indice_caracteristica_local;

            entrenamiento_compartido[indice_caracteristica_local][indice_muestra_local] =
                indice_muestra_global < numero_muestras
                ? datos_entrenamiento[
                    indice_muestra_global * numero_caracteristicas +
                    indice_caracteristica_global
                ]
                : 0.0F;
        }

        __syncthreads();

        if (salida_valida) {
            for (
                int indice_caracteristica_local = 0;
                indice_caracteristica_local < caracteristicas_en_baldosa;
                ++indice_caracteristica_local
            ) {
                const float diferencia =
                    consultas_compartidas[indice_caracteristica_local][threadIdx.y] -
                    entrenamiento_compartido[indice_caracteristica_local][threadIdx.x];
                distancia_cuadrada += diferencia * diferencia;
            }
        }

        __syncthreads();
    }

    if (salida_valida) {
        distancias_cuadradas[indice_consulta * numero_muestras + indice_muestra] =
            distancia_cuadrada;
    }
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
    const int64_t numero_bloques_consultas =
        (numero_consultas - 1) / tamanio_baldosa_consultas + 1;
    const int64_t numero_bloques_muestras =
        (numero_muestras - 1) / tamanio_baldosa_muestras + 1;
    TORCH_CHECK(
        numero_bloques_consultas <= std::numeric_limits<unsigned int>::max() &&
            numero_bloques_muestras <= std::numeric_limits<unsigned int>::max(),
        "el numero de bloques por dimension excede el rango admitido"
    );

    const c10::cuda::CUDAGuard guard(datos_consulta.device());
    const auto datos_consulta_contiguos = datos_consulta.contiguous();
    const auto datos_entrenamiento_contiguos = datos_entrenamiento.contiguous();
    auto distancias_cuadradas = torch::empty(
        {numero_consultas, numero_muestras}, datos_consulta.options()
    );

    calcular_distancias_l2_cuadradas_kernel<<<
        dim3(
            static_cast<unsigned int>(numero_bloques_muestras),
            static_cast<unsigned int>(numero_bloques_consultas)
        ),
        dim3(tamanio_baldosa_muestras, tamanio_baldosa_consultas),
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
