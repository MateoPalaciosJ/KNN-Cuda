import numpy as np
import pytest
import torch

from knn_cuda import ClasificadorKNNCUDA
from knn_cuda.referencia import predecir_knn


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA no esta disponible en este entorno"
)


def test_clasificador_cuda_ajusta_y_predice_igual_que_numpy_y_cpu() -> None:
    datos_entrenamiento = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32
    )
    etiquetas_entrenamiento = np.array([2, 5, 9, 5], dtype=np.int64)
    datos_consulta = np.array([[1.8, 0.2], [0.1, 1.9]], dtype=np.float32)
    esperado = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 3
    )
    clasificador_cuda = ClasificadorKNNCUDA(numero_vecinos=3, dispositivo="cuda")
    clasificador_cpu = ClasificadorKNNCUDA(numero_vecinos=3, dispositivo="cpu")

    clasificador_cuda.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    clasificador_cpu.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones_cuda = clasificador_cuda.predecir(datos_consulta)
    predicciones_cpu = clasificador_cpu.predecir(datos_consulta)

    assert clasificador_cuda.dispositivo_efectivo_.type == "cuda"
    assert clasificador_cuda.datos_entrenamiento_tensor_.device.type == "cuda"
    assert clasificador_cuda.etiquetas_entrenamiento_tensor_.device.type == "cuda"
    assert isinstance(predicciones_cuda, np.ndarray)
    np.testing.assert_array_equal(predicciones_cuda, esperado)
    np.testing.assert_array_equal(predicciones_cuda, predicciones_cpu)


def test_clasificador_cuda_preserva_int32_etiquetas_originales_y_empates() -> None:
    datos_entrenamiento = np.array([[0.0], [0.0], [2.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([-3, 100, 100, -3], dtype=np.int32)
    datos_consulta = np.array([[0.0], [1.0]], dtype=np.float32)
    clasificador = ClasificadorKNNCUDA(numero_vecinos=4, dispositivo="cuda")

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(datos_consulta)

    assert predicciones.dtype == np.int32
    np.testing.assert_array_equal(predicciones, np.array([-3, -3], dtype=np.int32))


def test_clasificador_cuda_resuelve_empate_de_distancia_por_indice_menor() -> None:
    datos_entrenamiento = np.array([[0.0], [0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2, 9], dtype=np.int64)
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1, dispositivo="cuda")

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(np.array([[0.0]], dtype=np.float32))

    np.testing.assert_array_equal(predicciones, np.array([5], dtype=np.int64))


def test_clasificador_cuda_vecinos_conserva_distancias_euclidianas_e_indices() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1, dispositivo="cuda")
    clasificador.ajustar(
        np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32),
        np.array([1, 2], dtype=np.int64),
    )
    datos_consulta = np.array([[0.0, 0.0]], dtype=np.float32)

    distancias, indices = clasificador.vecinos_mas_cercanos(datos_consulta)
    indices_sin_distancias = clasificador.vecinos_mas_cercanos(
        datos_consulta, devolver_distancias=False
    )

    np.testing.assert_array_equal(distancias, np.array([[0.0]], dtype=np.float32))
    np.testing.assert_array_equal(indices, np.array([[1]], dtype=np.int64))
    np.testing.assert_array_equal(indices_sin_distancias, indices)


def test_clasificador_auto_usa_cuda_cuando_runtime_y_kernels_estan_disponibles() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1, dispositivo="auto")

    clasificador.ajustar(
        np.array([[0.0]], dtype=np.float32), np.array([7], dtype=np.int64)
    )

    assert clasificador.dispositivo_efectivo_.type == "cuda"
    assert clasificador.datos_entrenamiento_tensor_.device.type == "cuda"


def test_clasificador_cuda_reutiliza_entrenamiento_y_conserva_entradas() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2], dtype=np.int64)
    datos_consulta = np.array([[0.1], [1.9]], dtype=np.float32)
    datos_entrenamiento_antes = datos_entrenamiento.copy()
    etiquetas_entrenamiento_antes = etiquetas_entrenamiento.copy()
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1, dispositivo="cuda")

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    datos_entrenamiento_tensor = clasificador.datos_entrenamiento_tensor_
    etiquetas_entrenamiento_tensor = clasificador.etiquetas_entrenamiento_tensor_
    datos_entrenamiento[0, 0] = 100.0
    etiquetas_entrenamiento[0] = 9
    primeras_predicciones = clasificador.predecir(datos_consulta)
    segundas_predicciones = clasificador.predecir(datos_consulta)

    assert clasificador.datos_entrenamiento_tensor_ is datos_entrenamiento_tensor
    assert clasificador.etiquetas_entrenamiento_tensor_ is etiquetas_entrenamiento_tensor
    np.testing.assert_array_equal(datos_entrenamiento_antes, clasificador.datos_entrenamiento_)
    np.testing.assert_array_equal(etiquetas_entrenamiento_antes, clasificador.etiquetas_entrenamiento_)
    np.testing.assert_array_equal(primeras_predicciones, np.array([5, 2], dtype=np.int64))
    np.testing.assert_array_equal(primeras_predicciones, segundas_predicciones)
