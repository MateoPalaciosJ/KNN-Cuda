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

__device__ bool candidato_es_mejor(
    float primera_distancia,
    int64_t primer_indice,
    float segunda_distancia,
    int64_t segundo_indice
) {
    if (primer_indice < 0) {
        return false;
    }
    if (segundo_indice < 0) {
        return true;
    }
    return primera_distancia < segunda_distancia ||
        (primera_distancia == segunda_distancia && primer_indice < segundo_indice);
}

__global__ void seleccionar_top_k_kernel(
    const float* distancias,
    bool* seleccionados,
    float* distancias_seleccionadas,
    int64_t* indices_seleccionados,
    int64_t numero_muestras,
    int64_t k
) {
    const int64_t indice_consulta = static_cast<int64_t>(blockIdx.x);
    const int64_t inicio_fila = indice_consulta * numero_muestras;
    const int64_t inicio_salida = indice_consulta * k;

    __shared__ float distancias_candidatas[hilos_por_bloque];
    __shared__ int64_t indices_candidatos[hilos_por_bloque];

    for (int64_t posicion_seleccionada = 0;
         posicion_seleccionada < k;
         ++posicion_seleccionada) {
        float distancia_mejor = 0.0F;
        int64_t indice_mejor = -1;

        for (int64_t indice_muestra = threadIdx.x;
             indice_muestra < numero_muestras;
             indice_muestra += blockDim.x) {
            if (seleccionados[inicio_fila + indice_muestra]) {
                continue;
            }

            const float distancia = distancias[inicio_fila + indice_muestra];
            if (candidato_es_mejor(
                    distancia,
                    indice_muestra,
                    distancia_mejor,
                    indice_mejor
                )) {
                distancia_mejor = distancia;
                indice_mejor = indice_muestra;
            }
        }

        distancias_candidatas[threadIdx.x] = distancia_mejor;
        indices_candidatos[threadIdx.x] = indice_mejor;
        __syncthreads();

        for (int desplazamiento = blockDim.x / 2;
             desplazamiento > 0;
             desplazamiento /= 2) {
            if (threadIdx.x < desplazamiento) {
                const int indice_companero = threadIdx.x + desplazamiento;
                if (candidato_es_mejor(
                        distancias_candidatas[indice_companero],
                        indices_candidatos[indice_companero],
                        distancias_candidatas[threadIdx.x],
                        indices_candidatos[threadIdx.x]
                    )) {
                    distancias_candidatas[threadIdx.x] =
                        distancias_candidatas[indice_companero];
                    indices_candidatos[threadIdx.x] =
                        indices_candidatos[indice_companero];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            const int64_t indice_ganador = indices_candidatos[0];
            distancias_seleccionadas[inicio_salida + posicion_seleccionada] =
                distancias_candidatas[0];
            indices_seleccionados[inicio_salida + posicion_seleccionada] =
                indice_ganador;
            seleccionados[inicio_fila + indice_ganador] = true;
        }
        __syncthreads();
    }
}

}

namespace knn_cuda {

std::tuple<torch::Tensor, torch::Tensor> seleccionar_top_k_cuda(
    const torch::Tensor& distancias,
    int64_t k
) {
    validar_matriz_cuda(distancias, "distancias");
    TORCH_CHECK(k >= 1, "k debe ser mayor o igual a 1");
    TORCH_CHECK(
        k <= distancias.size(1),
        "k no debe ser mayor que el numero de columnas de distancias"
    );

    const int64_t numero_consultas = distancias.size(0);
    const int64_t numero_muestras = distancias.size(1);
    TORCH_CHECK(
        numero_consultas <= std::numeric_limits<unsigned int>::max(),
        "el numero de consultas excede el rango admitido"
    );

    const c10::cuda::CUDAGuard guard(distancias.device());
    const auto distancias_contiguas = distancias.contiguous();
    auto distancias_seleccionadas = torch::empty(
        {numero_consultas, k}, distancias.options()
    );
    auto indices_seleccionados = torch::empty(
        {numero_consultas, k},
        distancias.options().dtype(torch::kInt64)
    );
    auto seleccionados = torch::zeros(
        {numero_consultas, numero_muestras},
        distancias.options().dtype(torch::kBool)
    );

    seleccionar_top_k_kernel<<<
        static_cast<unsigned int>(numero_consultas),
        hilos_por_bloque,
        0,
        c10::cuda::getCurrentCUDAStream().stream()
    >>>(
        distancias_contiguas.data_ptr<float>(),
        seleccionados.data_ptr<bool>(),
        distancias_seleccionadas.data_ptr<float>(),
        indices_seleccionados.data_ptr<int64_t>(),
        numero_muestras,
        k
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(distancias_seleccionadas, indices_seleccionados);
}

}
