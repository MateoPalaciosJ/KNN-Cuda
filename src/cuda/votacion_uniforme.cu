#include "knn_cuda/operadores_cuda.h"

#include <limits>

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int hilos_por_bloque = 256;

void validar_etiquetas_cuda(const torch::Tensor& etiquetas_vecinos) {
    TORCH_CHECK(etiquetas_vecinos.is_cuda(), "etiquetas_vecinos debe estar en CUDA");
    TORCH_CHECK(
        etiquetas_vecinos.scalar_type() != torch::kBool,
        "etiquetas_vecinos no debe tener dtype booleano"
    );
    TORCH_CHECK(
        c10::isIntegralType(etiquetas_vecinos.scalar_type(), false),
        "etiquetas_vecinos debe tener dtype entero"
    );
    TORCH_CHECK(etiquetas_vecinos.dim() == 2, "etiquetas_vecinos debe ser bidimensional");
    TORCH_CHECK(etiquetas_vecinos.numel() > 0, "etiquetas_vecinos no debe estar vacio");
}

__device__ bool candidato_es_mejor(
    int64_t primer_conteo,
    int64_t primera_etiqueta,
    int64_t segundo_conteo,
    int64_t segunda_etiqueta
) {
    if (primer_conteo < 0) {
        return false;
    }
    if (segundo_conteo < 0) {
        return true;
    }
    return primer_conteo > segundo_conteo ||
        (primer_conteo == segundo_conteo && primera_etiqueta < segunda_etiqueta);
}

template <typename tipo_etiqueta>
__global__ void votacion_uniforme_kernel(
    const tipo_etiqueta* etiquetas_vecinos,
    tipo_etiqueta* predicciones,
    int64_t numero_vecinos
) {
    const int64_t indice_consulta = static_cast<int64_t>(blockIdx.x);
    const int64_t inicio_fila = indice_consulta * numero_vecinos;

    __shared__ int64_t conteos_candidatos[hilos_por_bloque];
    __shared__ int64_t etiquetas_candidatas[hilos_por_bloque];

    int64_t mejor_conteo = -1;
    int64_t mejor_etiqueta = 0;

    for (int64_t indice_vecino = threadIdx.x;
         indice_vecino < numero_vecinos;
         indice_vecino += blockDim.x) {
        const int64_t etiqueta = static_cast<int64_t>(
            etiquetas_vecinos[inicio_fila + indice_vecino]
        );
        int64_t conteo = 0;

        for (int64_t indice_comparacion = 0;
             indice_comparacion < numero_vecinos;
             ++indice_comparacion) {
            const int64_t etiqueta_comparacion = static_cast<int64_t>(
                etiquetas_vecinos[inicio_fila + indice_comparacion]
            );
            if (etiqueta == etiqueta_comparacion) {
                ++conteo;
            }
        }

        if (candidato_es_mejor(
                conteo,
                etiqueta,
                mejor_conteo,
                mejor_etiqueta
            )) {
            mejor_conteo = conteo;
            mejor_etiqueta = etiqueta;
        }
    }

    conteos_candidatos[threadIdx.x] = mejor_conteo;
    etiquetas_candidatas[threadIdx.x] = mejor_etiqueta;
    __syncthreads();

    for (int desplazamiento = blockDim.x / 2;
         desplazamiento > 0;
         desplazamiento /= 2) {
        if (threadIdx.x < desplazamiento) {
            const int indice_companero = threadIdx.x + desplazamiento;
            if (candidato_es_mejor(
                    conteos_candidatos[indice_companero],
                    etiquetas_candidatas[indice_companero],
                    conteos_candidatos[threadIdx.x],
                    etiquetas_candidatas[threadIdx.x]
                )) {
                conteos_candidatos[threadIdx.x] =
                    conteos_candidatos[indice_companero];
                etiquetas_candidatas[threadIdx.x] =
                    etiquetas_candidatas[indice_companero];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        predicciones[indice_consulta] = static_cast<tipo_etiqueta>(
            etiquetas_candidatas[0]
        );
    }
}

}

namespace knn_cuda {

torch::Tensor votacion_uniforme_cuda(const torch::Tensor& etiquetas_vecinos) {
    validar_etiquetas_cuda(etiquetas_vecinos);

    const int64_t numero_consultas = etiquetas_vecinos.size(0);
    const int64_t numero_vecinos = etiquetas_vecinos.size(1);
    TORCH_CHECK(
        numero_consultas <= std::numeric_limits<unsigned int>::max(),
        "el numero de consultas excede el rango admitido"
    );

    const c10::cuda::CUDAGuard guard(etiquetas_vecinos.device());
    const auto etiquetas_contiguas = etiquetas_vecinos.contiguous();
    auto predicciones = torch::empty(
        {numero_consultas}, etiquetas_vecinos.options()
    );

    AT_DISPATCH_INTEGRAL_TYPES(
        etiquetas_contiguas.scalar_type(),
        "votacion_uniforme_cuda",
        [&] {
            votacion_uniforme_kernel<scalar_t><<<
                static_cast<unsigned int>(numero_consultas),
                hilos_por_bloque,
                0,
                c10::cuda::getCurrentCUDAStream().stream()
            >>>(
                etiquetas_contiguas.data_ptr<scalar_t>(),
                predicciones.data_ptr<scalar_t>(),
                numero_vecinos
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return predicciones;
}

}
