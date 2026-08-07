#include "knn_cuda/operadores.h"

#include <torch/torch.h>

namespace {

void validar_tensor_cpu(const torch::Tensor& tensor, const char* nombre) {
    TORCH_CHECK(tensor.device().is_cpu(), nombre, " debe estar en CPU");
}

void validar_tensor_cpu_float32(const torch::Tensor& tensor, const char* nombre) {
    validar_tensor_cpu(tensor, nombre);
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

void validar_tensor_etiquetas(
    const torch::Tensor& etiquetas,
    const char* nombre,
    int64_t numero_dimensiones,
    const char* descripcion_dimensiones
) {
    validar_tensor_cpu(etiquetas, nombre);
    TORCH_CHECK(
        c10::isIntegralType(etiquetas.scalar_type(), false),
        nombre,
        " debe tener dtype entero"
    );
    TORCH_CHECK(
        etiquetas.dim() == numero_dimensiones,
        nombre,
        " debe ser ",
        descripcion_dimensiones
    );
    TORCH_CHECK(etiquetas.numel() > 0, nombre, " no debe estar vacio");
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

std::tuple<torch::Tensor, torch::Tensor> seleccionar_top_k(
    const torch::Tensor& distancias,
    int64_t k
) {
    validar_matriz(distancias, "distancias");
    TORCH_CHECK(k >= 1, "k debe ser mayor o igual a 1");
    TORCH_CHECK(
        k <= distancias.size(1),
        "k no debe ser mayor que el numero de columnas de distancias"
    );

    const auto resultado_ordenado = distancias.sort(true, 1, false);
    const auto& distancias_ordenadas = std::get<0>(resultado_ordenado);
    const auto& indices_ordenados = std::get<1>(resultado_ordenado);

    return std::make_tuple(
        distancias_ordenadas.narrow(1, 0, k),
        indices_ordenados.narrow(1, 0, k)
    );
}

torch::Tensor votacion_uniforme(const torch::Tensor& etiquetas_vecinos) {
    validar_tensor_etiquetas(
        etiquetas_vecinos, "etiquetas_vecinos", 2, "bidimensional"
    );

    auto predicciones = torch::empty(
        {etiquetas_vecinos.size(0)}, etiquetas_vecinos.options()
    );

    for (int64_t indice_consulta = 0;
         indice_consulta < etiquetas_vecinos.size(0);
         ++indice_consulta) {
        const auto etiquetas_consulta = etiquetas_vecinos.select(0, indice_consulta);
        const auto etiquetas_ordenadas = std::get<0>(
            etiquetas_consulta.sort(true, 0, false)
        );
        const auto resultado_unico = torch::unique_consecutive(
            etiquetas_ordenadas, false, true
        );
        const auto& etiquetas_distintas = std::get<0>(resultado_unico);
        const auto& conteos = std::get<2>(resultado_unico);
        const auto indice_ganador = conteos.argmax();

        predicciones.select(0, indice_consulta).copy_(
            etiquetas_distintas.index({indice_ganador})
        );
    }

    return predicciones;
}

torch::Tensor predecir_knn(
    const torch::Tensor& datos_entrenamiento,
    const torch::Tensor& etiquetas_entrenamiento,
    const torch::Tensor& datos_consulta,
    int64_t k
) {
    validar_tensor_etiquetas(
        etiquetas_entrenamiento, "etiquetas_entrenamiento", 1, "unidimensional"
    );
    TORCH_CHECK(
        datos_entrenamiento.dim() >= 1,
        "datos_entrenamiento debe tener al menos una dimension"
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

    const auto distancias = distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    );
    const auto resultado_seleccion = seleccionar_top_k(distancias, k);
    const auto& indices_seleccionados = std::get<1>(resultado_seleccion);
    const auto etiquetas_vecinos = etiquetas_entrenamiento.index({indices_seleccionados});

    return votacion_uniforme(etiquetas_vecinos);
}

}
