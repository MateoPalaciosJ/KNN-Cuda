import importlib

import numpy as np
import pytest
import torch

from knn_cuda.referencia import votacion_uniforme


importlib.import_module("knn_cuda._backend_cpp")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA no esta disponible en este entorno"
)


def test_votacion_uniforme_cuda_tiene_kernel_registrado() -> None:
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "knn_cuda::votacion_uniforme", "CUDA"
    )


def test_votacion_uniforme_cuda_vota_una_consulta_binaria() -> None:
    etiquetas_vecinos = torch.tensor([[5, 2, 5]], dtype=torch.int64, device="cuda")

    predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert predicciones.shape == (1,)
    assert predicciones.dtype == torch.int64
    assert torch.equal(predicciones.cpu(), torch.tensor([5], dtype=torch.int64))


def test_votacion_uniforme_cuda_coincide_con_numpy_y_cpu_en_multiclase() -> None:
    etiquetas_numpy = np.array(
        [[5, 2, 5], [9, 5, 2], [7, 7, 2]], dtype=np.int64
    )
    esperado_numpy = votacion_uniforme(etiquetas_numpy)
    etiquetas_cpu = torch.from_numpy(etiquetas_numpy)
    predicciones_cpu = torch.ops.knn_cuda.votacion_uniforme(etiquetas_cpu)
    predicciones_cuda = torch.ops.knn_cuda.votacion_uniforme(etiquetas_cpu.cuda())

    np.testing.assert_array_equal(predicciones_cuda.cpu().numpy(), esperado_numpy)
    assert torch.equal(predicciones_cuda.cpu(), predicciones_cpu)


def test_votacion_uniforme_cuda_acepta_int32_y_preserva_dtype() -> None:
    etiquetas_vecinos = torch.tensor(
        [[1000000, -3, 1000000], [8, 8, 2]], dtype=torch.int32, device="cuda"
    )

    predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert predicciones.dtype == torch.int32
    assert torch.equal(predicciones.cpu(), torch.tensor([1000000, 8], dtype=torch.int32))


def test_votacion_uniforme_cuda_resuelve_empate_entre_dos_etiquetas() -> None:
    etiquetas_vecinos = torch.tensor([[5, 2]], dtype=torch.int64, device="cuda")

    predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert torch.equal(predicciones.cpu(), torch.tensor([2], dtype=torch.int64))


def test_votacion_uniforme_cuda_resuelve_empate_entre_tres_etiquetas() -> None:
    etiquetas_vecinos = torch.tensor([[9, 5, 2]], dtype=torch.int64, device="cuda")

    predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert torch.equal(predicciones.cpu(), torch.tensor([2], dtype=torch.int64))


def test_votacion_uniforme_cuda_acepta_etiquetas_negativas_y_no_consecutivas() -> None:
    etiquetas_vecinos = torch.tensor(
        [[-3, 10, -3], [100, -50, 100, -50]], dtype=torch.int64, device="cuda"
    )

    predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert torch.equal(predicciones.cpu(), torch.tensor([-3, -50], dtype=torch.int64))


def test_votacion_uniforme_cuda_funciona_con_k_uno_y_una_sola_clase() -> None:
    etiquetas_vecinos = torch.tensor([[7], [9]], dtype=torch.int64, device="cuda")

    predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert torch.equal(predicciones.cpu(), torch.tensor([7, 9], dtype=torch.int64))


def test_votacion_uniforme_cuda_es_determinista_y_no_modifica_la_entrada() -> None:
    etiquetas_vecinos = torch.tensor(
        [[7, 2, 7, 2], [5, 5, 2, 9]], dtype=torch.int64, device="cuda"
    )
    etiquetas_antes = etiquetas_vecinos.clone()

    primeras_predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)
    segundas_predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert torch.equal(primeras_predicciones, segundas_predicciones)
    assert torch.equal(primeras_predicciones.cpu(), torch.tensor([2, 5], dtype=torch.int64))
    assert torch.equal(etiquetas_vecinos, etiquetas_antes)


def test_votacion_uniforme_cuda_acepta_vista_no_contigua() -> None:
    etiquetas_vecinos = torch.tensor(
        [[5, 2], [2, 2], [5, 9]], dtype=torch.int64, device="cuda"
    ).transpose(0, 1)
    esperado = votacion_uniforme(etiquetas_vecinos.cpu().numpy())

    predicciones = torch.ops.knn_cuda.votacion_uniforme(etiquetas_vecinos)

    assert not etiquetas_vecinos.is_contiguous()
    np.testing.assert_array_equal(predicciones.cpu().numpy(), esperado)


def test_votacion_uniforme_cuda_funciona_con_k_igual_y_mayor_a_un_bloque() -> None:
    etiquetas_igual_bloque = torch.tensor(
        [[3] * 256], dtype=torch.int64, device="cuda"
    )
    etiquetas_mayor_bloque = torch.tensor(
        [[7, 2] * 256 + [2]], dtype=torch.int64, device="cuda"
    )

    prediccion_igual_bloque = torch.ops.knn_cuda.votacion_uniforme(etiquetas_igual_bloque)
    prediccion_mayor_bloque = torch.ops.knn_cuda.votacion_uniforme(etiquetas_mayor_bloque)

    assert torch.equal(prediccion_igual_bloque.cpu(), torch.tensor([3], dtype=torch.int64))
    assert torch.equal(prediccion_mayor_bloque.cpu(), torch.tensor([2], dtype=torch.int64))


def test_votacion_uniforme_cuda_rechaza_dtype_no_entero() -> None:
    etiquetas_float32 = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    etiquetas_float64 = torch.tensor([[1.0]], dtype=torch.float64, device="cuda")
    etiquetas_booleanas = torch.tensor([[True]], dtype=torch.bool, device="cuda")

    with pytest.raises(RuntimeError, match="etiquetas_vecinos"):
        torch.ops.knn_cuda.votacion_uniforme(etiquetas_float32)
    with pytest.raises(RuntimeError, match="etiquetas_vecinos"):
        torch.ops.knn_cuda.votacion_uniforme(etiquetas_float64)
    with pytest.raises(RuntimeError, match="etiquetas_vecinos"):
        torch.ops.knn_cuda.votacion_uniforme(etiquetas_booleanas)


def test_votacion_uniforme_cuda_rechaza_dimensiones_y_tensor_vacio() -> None:
    etiquetas_unidimensionales = torch.tensor([1], dtype=torch.int64, device="cuda")
    etiquetas_tridimensionales = torch.ones((1, 1, 1), dtype=torch.int64, device="cuda")
    etiquetas_vacias = torch.empty((1, 0), dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="etiquetas_vecinos"):
        torch.ops.knn_cuda.votacion_uniforme(etiquetas_unidimensionales)
    with pytest.raises(RuntimeError, match="etiquetas_vecinos"):
        torch.ops.knn_cuda.votacion_uniforme(etiquetas_tridimensionales)
    with pytest.raises(RuntimeError, match="etiquetas_vecinos"):
        torch.ops.knn_cuda.votacion_uniforme(etiquetas_vacias)
