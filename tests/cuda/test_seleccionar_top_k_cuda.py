import importlib

import numpy as np
import pytest
import torch

from knn_cuda.referencia import seleccionar_top_k


importlib.import_module("knn_cuda._backend_cpp")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA no esta disponible en este entorno"
)


def test_seleccionar_top_k_cuda_tiene_kernel_registrado() -> None:
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "knn_cuda::seleccionar_top_k", "CUDA"
    )


def test_seleccionar_top_k_cuda_selecciona_k_uno() -> None:
    distancias = torch.tensor([[4.0, 1.0, 3.0]], dtype=torch.float32, device="cuda")

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)
    )

    assert distancias_seleccionadas.shape == (1, 1)
    assert indices_seleccionados.shape == (1, 1)
    assert distancias_seleccionadas.dtype == torch.float32
    assert indices_seleccionados.dtype == torch.int64
    assert torch.equal(distancias_seleccionadas.cpu(), torch.tensor([[1.0]]))
    assert torch.equal(indices_seleccionados.cpu(), torch.tensor([[1]]))


def test_seleccionar_top_k_cuda_selecciona_k_n_y_orden_ascendente() -> None:
    distancias = torch.tensor([[4.0, 1.0, 1.0, 3.0]], dtype=torch.float32, device="cuda")

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 4)
    )

    assert torch.equal(
        distancias_seleccionadas.cpu(), torch.tensor([[1.0, 1.0, 3.0, 4.0]])
    )
    assert torch.equal(indices_seleccionados.cpu(), torch.tensor([[1, 2, 3, 0]]))


def test_seleccionar_top_k_cuda_coincide_con_numpy_y_cpu() -> None:
    distancias_numpy = np.array(
        [[4.0, 1.0, 1.0, 3.0], [5.0, 2.0, 2.0, 0.0]], dtype=np.float32
    )
    esperadas_numpy, indices_esperados_numpy = seleccionar_top_k(distancias_numpy, 3)
    distancias_cpu = torch.from_numpy(distancias_numpy)
    esperadas_cpu, indices_esperados_cpu = torch.ops.knn_cuda.seleccionar_top_k(
        distancias_cpu, 3
    )
    distancias_cuda, indices_cuda = torch.ops.knn_cuda.seleccionar_top_k(
        distancias_cpu.cuda(), 3
    )

    np.testing.assert_array_equal(distancias_cuda.cpu().numpy(), esperadas_numpy)
    np.testing.assert_array_equal(indices_cuda.cpu().numpy(), indices_esperados_numpy)
    assert torch.equal(distancias_cuda.cpu(), esperadas_cpu)
    assert torch.equal(indices_cuda.cpu(), indices_esperados_cpu)


def test_seleccionar_top_k_cuda_resuelve_empates_multiples_por_indice_menor() -> None:
    distancias = torch.tensor(
        [[2.0, 1.0, 1.0, 1.0, 2.0]], dtype=torch.float32, device="cuda"
    )

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 3)
    )

    assert torch.equal(distancias_seleccionadas.cpu(), torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.equal(indices_seleccionados.cpu(), torch.tensor([[1, 2, 3]]))


def test_seleccionar_top_k_cuda_acepta_una_muestra_y_varias_consultas() -> None:
    distancias = torch.tensor([[7.0], [0.0]], dtype=torch.float32, device="cuda")

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)
    )

    assert torch.equal(distancias_seleccionadas.cpu(), torch.tensor([[7.0], [0.0]]))
    assert torch.equal(indices_seleccionados.cpu(), torch.tensor([[0], [0]]))


def test_seleccionar_top_k_cuda_es_determinista_y_no_modifica_la_entrada() -> None:
    distancias = torch.tensor(
        [[3.0, 1.0, 1.0], [2.0, 2.0, 0.0]], dtype=torch.float32, device="cuda"
    )
    distancias_antes = distancias.clone()

    primer_resultado = torch.ops.knn_cuda.seleccionar_top_k(distancias, 2)
    segundo_resultado = torch.ops.knn_cuda.seleccionar_top_k(distancias, 2)

    assert torch.equal(primer_resultado[0], segundo_resultado[0])
    assert torch.equal(primer_resultado[1], segundo_resultado[1])
    assert torch.equal(distancias, distancias_antes)


def test_seleccionar_top_k_cuda_acepta_vista_no_contigua() -> None:
    distancias = torch.tensor(
        [[4.0, 1.0], [2.0, 3.0], [1.0, 1.0]], dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    esperadas_numpy, indices_esperados_numpy = seleccionar_top_k(
        distancias.cpu().numpy(), 2
    )

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 2)
    )

    assert not distancias.is_contiguous()
    np.testing.assert_array_equal(distancias_seleccionadas.cpu().numpy(), esperadas_numpy)
    np.testing.assert_array_equal(indices_seleccionados.cpu().numpy(), indices_esperados_numpy)


def test_seleccionar_top_k_cuda_funciona_con_n_mayor_a_un_bloque() -> None:
    distancias = torch.zeros((2, 513), dtype=torch.float32, device="cuda")

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 4)
    )

    assert torch.equal(
        distancias_seleccionadas, torch.zeros((2, 4), dtype=torch.float32, device="cuda")
    )
    assert torch.equal(
        indices_seleccionados,
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int64, device="cuda"),
    )


def test_seleccionar_top_k_cuda_resuelve_empates_entre_warps() -> None:
    distancias = torch.full((1, 513), 10.0, dtype=torch.float32, device="cuda")
    distancias[0, torch.tensor([7, 45, 100, 300], device="cuda")] = 1.0

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 4)
    )

    assert torch.equal(
        distancias_seleccionadas,
        torch.ones((1, 4), dtype=torch.float32, device="cuda"),
    )
    assert torch.equal(
        indices_seleccionados,
        torch.tensor([[7, 45, 100, 300]], dtype=torch.int64, device="cuda"),
    )


def test_seleccionar_top_k_cuda_ruta_warp_coincide_con_numpy_y_cpu() -> None:
    posiciones_empate = np.array(
        [7, 45, 100, 200, 300, 400, 500, 600, 700, 800,
         900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800,
         1900, 2000],
        dtype=np.int64,
    )
    distancias_numpy = np.full((1, 2048), 10.0, dtype=np.float32)
    distancias_numpy[0, posiciones_empate] = 1.0
    esperadas_numpy, indices_esperados_numpy = seleccionar_top_k(distancias_numpy, 20)
    distancias_cuda = torch.from_numpy(distancias_numpy).cuda()
    distancias_antes = distancias_cuda.clone()

    primer_resultado = torch.ops.knn_cuda.seleccionar_top_k(distancias_cuda, 20)
    segundo_resultado = torch.ops.knn_cuda.seleccionar_top_k(distancias_cuda, 20)
    resultado_cpu = torch.ops.knn_cuda.seleccionar_top_k(
        torch.from_numpy(distancias_numpy), 20
    )

    np.testing.assert_array_equal(primer_resultado[0].cpu().numpy(), esperadas_numpy)
    np.testing.assert_array_equal(
        primer_resultado[1].cpu().numpy(), indices_esperados_numpy
    )
    assert torch.equal(primer_resultado[0], segundo_resultado[0])
    assert torch.equal(primer_resultado[1], segundo_resultado[1])
    assert torch.equal(primer_resultado[0].cpu(), resultado_cpu[0])
    assert torch.equal(primer_resultado[1].cpu(), resultado_cpu[1])
    assert torch.equal(distancias_cuda, distancias_antes)


def test_seleccionar_top_k_cuda_ruta_warp_acepta_vista_con_n_irregular() -> None:
    distancias_base = torch.arange(
        2051 * 2, dtype=torch.float32, device="cuda"
    ).reshape(2051, 2)
    distancias = distancias_base.transpose(0, 1)
    esperadas_numpy, indices_esperados_numpy = seleccionar_top_k(
        distancias.cpu().numpy(), 20
    )

    distancias_seleccionadas, indices_seleccionados = (
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 20)
    )

    assert not distancias.is_contiguous()
    np.testing.assert_array_equal(distancias_seleccionadas.cpu().numpy(), esperadas_numpy)
    np.testing.assert_array_equal(indices_seleccionados.cpu().numpy(), indices_esperados_numpy)


def test_seleccionar_top_k_cuda_rechaza_dtype_invalido() -> None:
    distancias_float64 = torch.tensor([[1.0]], dtype=torch.float64, device="cuda")
    distancias_enteras = torch.tensor([[1]], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias_float64, 1)
    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias_enteras, 1)


def test_seleccionar_top_k_cuda_rechaza_dimensiones_y_matriz_vacia() -> None:
    distancias_unidimensionales = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    distancias_tridimensionales = torch.zeros((1, 1, 1), dtype=torch.float32, device="cuda")
    distancias_vacias = torch.empty((1, 0), dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias_unidimensionales, 1)
    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias_tridimensionales, 1)
    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias_vacias, 1)


def test_seleccionar_top_k_cuda_rechaza_valores_no_finitos_y_k_invalido() -> None:
    distancias_nan = torch.tensor([[float("nan")]], dtype=torch.float32, device="cuda")
    distancias_infinitas = torch.tensor([[float("inf")]], dtype=torch.float32, device="cuda")
    distancias = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias_nan, 1)
    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias_infinitas, 1)
    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 0)
    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, -1)
    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 3)
