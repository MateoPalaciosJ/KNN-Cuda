import importlib

import numpy as np
import pytest
import torch

from knn_cuda.referencia import predecir_knn


importlib.import_module("knn_cuda._backend_cpp")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA no esta disponible en este entorno"
)


def test_predecir_knn_cuda_tiene_kernel_registrado() -> None:
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "knn_cuda::predecir_knn", "CUDA"
    )


def test_predecir_knn_cuda_coincide_con_numpy_y_cpu_en_k_uno() -> None:
    datos_entrenamiento_numpy = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento_numpy = np.array([5, 2, 5], dtype=np.int64)
    datos_consulta_numpy = np.array([[1.0], [3.0]], dtype=np.float32)
    esperado = predecir_knn(
        datos_entrenamiento_numpy,
        etiquetas_entrenamiento_numpy,
        datos_consulta_numpy,
        1,
    )

    predicciones_cpu = torch.ops.knn_cuda.predecir_knn(
        torch.from_numpy(datos_entrenamiento_numpy),
        torch.from_numpy(etiquetas_entrenamiento_numpy),
        torch.from_numpy(datos_consulta_numpy),
        1,
    )
    predicciones_cuda = torch.ops.knn_cuda.predecir_knn(
        torch.from_numpy(datos_entrenamiento_numpy).cuda(),
        torch.from_numpy(etiquetas_entrenamiento_numpy).cuda(),
        torch.from_numpy(datos_consulta_numpy).cuda(),
        1,
    )

    np.testing.assert_array_equal(predicciones_cuda.cpu().numpy(), esperado)
    assert torch.equal(predicciones_cuda.cpu(), predicciones_cpu)


def test_predecir_knn_cuda_coincide_con_numpy_en_multiclase_y_k_n() -> None:
    datos_entrenamiento_numpy = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float32
    )
    etiquetas_entrenamiento_numpy = np.array([2, 5, 9, 5], dtype=np.int64)
    datos_consulta_numpy = np.array([[1.8, 0.2], [0.1, 1.9]], dtype=np.float32)
    esperado = predecir_knn(
        datos_entrenamiento_numpy,
        etiquetas_entrenamiento_numpy,
        datos_consulta_numpy,
        4,
    )

    predicciones_cuda = torch.ops.knn_cuda.predecir_knn(
        torch.from_numpy(datos_entrenamiento_numpy).cuda(),
        torch.from_numpy(etiquetas_entrenamiento_numpy).cuda(),
        torch.from_numpy(datos_consulta_numpy).cuda(),
        4,
    )

    np.testing.assert_array_equal(predicciones_cuda.cpu().numpy(), esperado)


def test_predecir_knn_cuda_resuelve_empate_de_distancia_por_indice_menor() -> None:
    datos_entrenamiento_numpy = np.array([[0.0], [0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento_numpy = np.array([5, 2, 9], dtype=np.int64)
    datos_consulta_numpy = np.array([[0.0]], dtype=np.float32)
    esperado = predecir_knn(
        datos_entrenamiento_numpy,
        etiquetas_entrenamiento_numpy,
        datos_consulta_numpy,
        1,
    )

    predicciones_cuda = torch.ops.knn_cuda.predecir_knn(
        torch.from_numpy(datos_entrenamiento_numpy).cuda(),
        torch.from_numpy(etiquetas_entrenamiento_numpy).cuda(),
        torch.from_numpy(datos_consulta_numpy).cuda(),
        1,
    )

    np.testing.assert_array_equal(predicciones_cuda.cpu().numpy(), esperado)
    assert torch.equal(predicciones_cuda.cpu(), torch.tensor([5], dtype=torch.int64))


def test_predecir_knn_cuda_resuelve_empate_de_votos_por_etiqueta_menor() -> None:
    datos_entrenamiento_numpy = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    etiquetas_entrenamiento_numpy = np.array([7, 2, 7, 2], dtype=np.int64)
    datos_consulta_numpy = np.array([[3.0]], dtype=np.float32)
    esperado = predecir_knn(
        datos_entrenamiento_numpy,
        etiquetas_entrenamiento_numpy,
        datos_consulta_numpy,
        4,
    )

    predicciones_cuda = torch.ops.knn_cuda.predecir_knn(
        torch.from_numpy(datos_entrenamiento_numpy).cuda(),
        torch.from_numpy(etiquetas_entrenamiento_numpy).cuda(),
        torch.from_numpy(datos_consulta_numpy).cuda(),
        4,
    )

    np.testing.assert_array_equal(predicciones_cuda.cpu().numpy(), esperado)
    assert torch.equal(predicciones_cuda.cpu(), torch.tensor([2], dtype=torch.int64))


def test_predecir_knn_cuda_combina_empates_de_distancia_y_votos() -> None:
    datos_entrenamiento_numpy = np.array([[0.0], [0.0], [2.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento_numpy = np.array([7, 2, 7, 2], dtype=np.int64)
    datos_consulta_numpy = np.array([[1.0]], dtype=np.float32)
    esperado = predecir_knn(
        datos_entrenamiento_numpy,
        etiquetas_entrenamiento_numpy,
        datos_consulta_numpy,
        4,
    )

    predicciones_cuda = torch.ops.knn_cuda.predecir_knn(
        torch.from_numpy(datos_entrenamiento_numpy).cuda(),
        torch.from_numpy(etiquetas_entrenamiento_numpy).cuda(),
        torch.from_numpy(datos_consulta_numpy).cuda(),
        4,
    )

    np.testing.assert_array_equal(predicciones_cuda.cpu().numpy(), esperado)
    assert torch.equal(predicciones_cuda.cpu(), torch.tensor([2], dtype=torch.int64))


def test_predecir_knn_cuda_acepta_etiquetas_int32_y_preserva_dtype() -> None:
    datos_entrenamiento = torch.tensor([[0.0], [2.0]], dtype=torch.float32, device="cuda")
    etiquetas_entrenamiento = torch.tensor([1000000, -3], dtype=torch.int32, device="cuda")
    datos_consulta = torch.tensor([[0.0], [2.0]], dtype=torch.float32, device="cuda")

    predicciones = torch.ops.knn_cuda.predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    assert predicciones.dtype == torch.int32
    assert torch.equal(predicciones.cpu(), torch.tensor([1000000, -3], dtype=torch.int32))


def test_predecir_knn_cuda_funciona_con_una_muestra_de_entrenamiento() -> None:
    datos_entrenamiento = torch.tensor([[0.0]], dtype=torch.float32, device="cuda")
    etiquetas_entrenamiento = torch.tensor([-3], dtype=torch.int64, device="cuda")
    datos_consulta = torch.tensor([[4.0]], dtype=torch.float32, device="cuda")

    predicciones = torch.ops.knn_cuda.predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    assert torch.equal(predicciones.cpu(), torch.tensor([-3], dtype=torch.int64))


def test_predecir_knn_cuda_acepta_vistas_no_contiguas() -> None:
    datos_entrenamiento = torch.tensor(
        [[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    etiquetas_entrenamiento = torch.tensor(
        [2, 99, 5, 99, 2, 99], dtype=torch.int64, device="cuda"
    )[::2]
    datos_consulta = torch.tensor(
        [[0.0, 2.0], [0.0, 0.0]], dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    esperado = predecir_knn(
        datos_entrenamiento.cpu().numpy(),
        etiquetas_entrenamiento.cpu().numpy(),
        datos_consulta.cpu().numpy(),
        1,
    )

    predicciones = torch.ops.knn_cuda.predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    assert not datos_entrenamiento.is_contiguous()
    assert not etiquetas_entrenamiento.is_contiguous()
    assert not datos_consulta.is_contiguous()
    np.testing.assert_array_equal(predicciones.cpu().numpy(), esperado)


def test_predecir_knn_cuda_es_determinista_y_no_modifica_las_entradas() -> None:
    datos_entrenamiento = torch.tensor(
        [[0.0], [2.0], [4.0]], dtype=torch.float32, device="cuda"
    )
    etiquetas_entrenamiento = torch.tensor([5, 2, 2], dtype=torch.int64, device="cuda")
    datos_consulta = torch.tensor([[1.0], [3.0]], dtype=torch.float32, device="cuda")
    datos_entrenamiento_antes = datos_entrenamiento.clone()
    etiquetas_entrenamiento_antes = etiquetas_entrenamiento.clone()
    datos_consulta_antes = datos_consulta.clone()

    primeras_predicciones = torch.ops.knn_cuda.predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 2
    )
    segundas_predicciones = torch.ops.knn_cuda.predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 2
    )

    assert torch.equal(primeras_predicciones, segundas_predicciones)
    assert torch.equal(datos_entrenamiento, datos_entrenamiento_antes)
    assert torch.equal(etiquetas_entrenamiento, etiquetas_entrenamiento_antes)
    assert torch.equal(datos_consulta, datos_consulta_antes)


def test_predecir_knn_cuda_rechaza_etiquetas_invalidas() -> None:
    datos_entrenamiento = torch.tensor([[0.0]], dtype=torch.float32, device="cuda")
    datos_consulta = torch.tensor([[0.0]], dtype=torch.float32, device="cuda")
    etiquetas_float = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    etiquetas_booleanas = torch.tensor([True], dtype=torch.bool, device="cuda")
    etiquetas_bidimensionales = torch.tensor([[1]], dtype=torch.int64, device="cuda")
    etiquetas_vacias = torch.empty((0,), dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="etiquetas_entrenamiento"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_float, datos_consulta, 1
        )
    with pytest.raises(RuntimeError, match="etiquetas_entrenamiento"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_booleanas, datos_consulta, 1
        )
    with pytest.raises(RuntimeError, match="etiquetas_entrenamiento"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_bidimensionales, datos_consulta, 1
        )
    with pytest.raises(RuntimeError, match="etiquetas_entrenamiento"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_vacias, datos_consulta, 1
        )


def test_predecir_knn_cuda_rechaza_inconsistencia_y_k_antes_del_calculo() -> None:
    datos_entrenamiento = torch.tensor(
        [[0.0], [1.0]], dtype=torch.float32, device="cuda"
    )
    etiquetas_entrenamiento = torch.tensor([1], dtype=torch.int64, device="cuda")
    datos_consulta = torch.tensor([[float("nan")]], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="etiquetas_entrenamiento"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )

    etiquetas_validas = torch.tensor([1, 2], dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_validas, datos_consulta, 0
        )
    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_validas, datos_consulta, -1
        )
    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento, etiquetas_validas, datos_consulta, 3
        )


def test_predecir_knn_cuda_propaga_errores_de_matrices() -> None:
    etiquetas_entrenamiento = torch.tensor([1], dtype=torch.int64, device="cuda")
    datos_entrenamiento_float64 = torch.tensor([[0.0]], dtype=torch.float64, device="cuda")
    datos_entrenamiento_infinito = torch.tensor(
        [[float("inf")]], dtype=torch.float32, device="cuda"
    )
    datos_entrenamiento = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device="cuda")
    datos_consulta = torch.tensor([[0.0]], dtype=torch.float32, device="cuda")
    datos_consulta_nan = torch.tensor([[float("nan")]], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="datos_entrenamiento"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento_float64,
            etiquetas_entrenamiento,
            datos_consulta,
            1,
        )
    with pytest.raises(RuntimeError, match="datos_entrenamiento"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento_infinito,
            etiquetas_entrenamiento,
            datos_consulta,
            1,
        )
    with pytest.raises(RuntimeError, match="datos_consulta"):
        torch.ops.knn_cuda.predecir_knn(
            torch.tensor([[0.0]], dtype=torch.float32, device="cuda"),
            etiquetas_entrenamiento,
            datos_consulta_nan,
            1,
        )
    with pytest.raises(RuntimeError, match="misma cantidad de caracteristicas"):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento,
            etiquetas_entrenamiento,
            datos_consulta,
            1,
        )
