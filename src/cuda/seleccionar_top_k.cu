#include "knn_cuda/operadores_cuda.h"

#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

namespace {

constexpr int hilos_por_bloque = 256;
constexpr int hilos_por_warp = 32;
constexpr int numero_warps_por_bloque = hilos_por_bloque / hilos_por_warp;
constexpr int64_t maximo_vecinos_ruta_warp = 32;
constexpr size_t limite_bytes_memoria_compartida_ruta_warp = 16 * 1024;

struct Candidato {
    float distancia;
    int64_t indice;
};

static_assert(sizeof(Candidato) == 16);

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

__host__ __device__ size_t alinear_bytes(
    size_t bytes,
    size_t alineacion
) {
    return (bytes + alineacion - 1) & ~(alineacion - 1);
}

size_t calcular_bytes_memoria_compartida_ruta_warp(
    int64_t numero_muestras,
    int64_t k
) {
    size_t bytes = static_cast<size_t>(numero_muestras) * sizeof(float);
    bytes += static_cast<size_t>(numero_muestras) * sizeof(bool);
    bytes = alinear_bytes(bytes, alignof(Candidato));
    bytes += static_cast<size_t>(numero_warps_por_bloque) *
        static_cast<size_t>(k) * sizeof(Candidato);
    return bytes;
}

bool usar_ruta_warp(
    int64_t numero_muestras,
    int64_t k,
    int indice_dispositivo
) {
    if (
        k > maximo_vecinos_ruta_warp ||
        numero_muestras / numero_warps_por_bloque < k
    ) {
        return false;
    }

    const size_t bytes_candidatos =
        static_cast<size_t>(numero_warps_por_bloque) *
        static_cast<size_t>(k) * sizeof(Candidato);
    const size_t bytes_alineacion = alignof(Candidato) - 1;
    if (bytes_candidatos + bytes_alineacion >
        limite_bytes_memoria_compartida_ruta_warp) {
        return false;
    }
    const size_t bytes_disponibles_muestras =
        limite_bytes_memoria_compartida_ruta_warp -
        bytes_candidatos -
        bytes_alineacion;
    const size_t bytes_por_muestra = sizeof(float) + sizeof(bool);
    if (numero_muestras > static_cast<int64_t>(
            bytes_disponibles_muestras / bytes_por_muestra
        )) {
        return false;
    }

    const size_t bytes_memoria_compartida =
        calcular_bytes_memoria_compartida_ruta_warp(numero_muestras, k);
    if (bytes_memoria_compartida > limite_bytes_memoria_compartida_ruta_warp) {
        return false;
    }

    int bytes_maximos_memoria_compartida = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &bytes_maximos_memoria_compartida,
        cudaDevAttrMaxSharedMemoryPerBlock,
        indice_dispositivo
    ));
    return bytes_memoria_compartida <=
        static_cast<size_t>(bytes_maximos_memoria_compartida);
}

__device__ void reducir_candidatos_warp(
    float& distancia,
    int64_t& indice
) {
    const unsigned mascara_warp = __activemask();
    const int indice_lane = threadIdx.x % hilos_por_warp;

    for (int desplazamiento = hilos_por_warp / 2;
         desplazamiento > 0;
         desplazamiento /= 2) {
        const float distancia_companero =
            __shfl_down_sync(mascara_warp, distancia, desplazamiento);
        const int64_t indice_companero =
            __shfl_down_sync(mascara_warp, indice, desplazamiento);

        if (
            indice_lane < desplazamiento &&
            candidato_es_mejor(
                distancia_companero,
                indice_companero,
                distancia,
                indice
            )
        ) {
            distancia = distancia_companero;
            indice = indice_companero;
        }
    }
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
    const int indice_warp = threadIdx.x / hilos_por_warp;
    const int indice_lane = threadIdx.x % hilos_por_warp;

    __shared__ float distancias_candidatas[numero_warps_por_bloque];
    __shared__ int64_t indices_candidatos[numero_warps_por_bloque];

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

        reducir_candidatos_warp(distancia_mejor, indice_mejor);

        if (indice_lane == 0) {
            distancias_candidatas[indice_warp] = distancia_mejor;
            indices_candidatos[indice_warp] = indice_mejor;
        }
        __syncthreads();

        if (indice_warp == 0) {
            if (indice_lane < numero_warps_por_bloque) {
                distancia_mejor = distancias_candidatas[indice_lane];
                indice_mejor = indices_candidatos[indice_lane];
            } else {
                distancia_mejor = 0.0F;
                indice_mejor = -1;
            }
            reducir_candidatos_warp(distancia_mejor, indice_mejor);
        }

        if (threadIdx.x == 0) {
            const int64_t indice_ganador = indice_mejor;
            distancias_seleccionadas[inicio_salida + posicion_seleccionada] =
                distancia_mejor;
            indices_seleccionados[inicio_salida + posicion_seleccionada] =
                indice_ganador;
            seleccionados[inicio_fila + indice_ganador] = true;
        }
        __syncthreads();
    }
}

__global__ void seleccionar_top_k_warp_kernel(
    const float* distancias,
    float* distancias_seleccionadas,
    int64_t* indices_seleccionados,
    int64_t numero_muestras,
    int64_t k
) {
    extern __shared__ unsigned char memoria_compartida[];

    const int64_t indice_consulta = static_cast<int64_t>(blockIdx.x);
    const int64_t inicio_fila = indice_consulta * numero_muestras;
    const int64_t inicio_salida = indice_consulta * k;
    const int indice_warp = threadIdx.x / hilos_por_warp;
    const int indice_lane = threadIdx.x % hilos_por_warp;

    auto* distancias_compartidas =
        reinterpret_cast<float*>(memoria_compartida);
    size_t desplazamiento_memoria =
        static_cast<size_t>(numero_muestras) * sizeof(float);
    auto* seleccionados_locales = reinterpret_cast<bool*>(
        memoria_compartida + desplazamiento_memoria
    );
    desplazamiento_memoria += static_cast<size_t>(numero_muestras) * sizeof(bool);
    desplazamiento_memoria = alinear_bytes(
        desplazamiento_memoria,
        alignof(Candidato)
    );
    auto* candidatos_locales = reinterpret_cast<Candidato*>(
        memoria_compartida + desplazamiento_memoria
    );

    for (
        int64_t indice_muestra = threadIdx.x;
        indice_muestra < numero_muestras;
        indice_muestra += blockDim.x
    ) {
        distancias_compartidas[indice_muestra] =
            distancias[inicio_fila + indice_muestra];
        seleccionados_locales[indice_muestra] = false;
    }
    __syncthreads();

    const int64_t inicio_segmento =
        numero_muestras * indice_warp / numero_warps_por_bloque;
    const int64_t fin_segmento =
        numero_muestras * (indice_warp + 1) / numero_warps_por_bloque;

    for (
        int64_t posicion_seleccionada = 0;
        posicion_seleccionada < k;
        ++posicion_seleccionada
    ) {
        float distancia_mejor = 0.0F;
        int64_t indice_mejor = -1;

        for (
            int64_t indice_muestra = inicio_segmento + indice_lane;
            indice_muestra < fin_segmento;
            indice_muestra += hilos_por_warp
        ) {
            if (seleccionados_locales[indice_muestra]) {
                continue;
            }

            const float distancia = distancias_compartidas[indice_muestra];
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

        reducir_candidatos_warp(distancia_mejor, indice_mejor);

        if (indice_lane == 0) {
            candidatos_locales[indice_warp * k + posicion_seleccionada] = {
                distancia_mejor,
                indice_mejor,
            };
            seleccionados_locales[indice_mejor] = true;
        }
        __syncwarp();
    }
    __syncthreads();

    if (indice_warp != 0) {
        return;
    }

    const int64_t numero_candidatos = numero_warps_por_bloque * k;
    for (
        int64_t posicion_seleccionada = 0;
        posicion_seleccionada < k;
        ++posicion_seleccionada
    ) {
        float distancia_mejor = 0.0F;
        int64_t indice_mejor = -1;

        for (
            int64_t posicion_candidato = indice_lane;
            posicion_candidato < numero_candidatos;
            posicion_candidato += hilos_por_warp
        ) {
            const Candidato candidato = candidatos_locales[posicion_candidato];
            if (candidato_es_mejor(
                    candidato.distancia,
                    candidato.indice,
                    distancia_mejor,
                    indice_mejor
                )) {
                distancia_mejor = candidato.distancia;
                indice_mejor = candidato.indice;
            }
        }

        reducir_candidatos_warp(distancia_mejor, indice_mejor);

        if (indice_lane == 0) {
            distancias_seleccionadas[inicio_salida + posicion_seleccionada] =
                distancia_mejor;
            indices_seleccionados[inicio_salida + posicion_seleccionada] =
                indice_mejor;
            for (
                int64_t posicion_candidato = 0;
                posicion_candidato < numero_candidatos;
                ++posicion_candidato
            ) {
                if (candidatos_locales[posicion_candidato].indice == indice_mejor) {
                    candidatos_locales[posicion_candidato].indice = -1;
                    break;
                }
            }
        }
        __syncwarp();
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
    if (usar_ruta_warp(numero_muestras, k, distancias.get_device())) {
        const size_t bytes_memoria_compartida =
            calcular_bytes_memoria_compartida_ruta_warp(numero_muestras, k);
        seleccionar_top_k_warp_kernel<<<
            static_cast<unsigned int>(numero_consultas),
            hilos_por_bloque,
            bytes_memoria_compartida,
            c10::cuda::getCurrentCUDAStream().stream()
        >>>(
            distancias_contiguas.data_ptr<float>(),
            distancias_seleccionadas.data_ptr<float>(),
            indices_seleccionados.data_ptr<int64_t>(),
            numero_muestras,
            k
        );
    } else {
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
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return std::make_tuple(distancias_seleccionadas, indices_seleccionados);
}

}
