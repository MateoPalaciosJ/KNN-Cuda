import importlib

import numpy as np
import pytest
import torch

from knn_cuda.referencia import distancias_l2_cuadradas


importlib.import_module("knn_cuda._backend_cpp")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA no esta disponible en este entorno"
)


def test_distancias_l2_cuadradas_cuda_tiene_kernel_registrado() -> None:
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "knn_cuda::distancias_l2_cuadradas", "CUDA"
    )


def test_distancias_l2_cuadradas_cuda_calcula_una_consulta_y_una_muestra() -> None:
    datos_consulta = torch.tensor([[1.0]], dtype=torch.float32, device="cuda")
    datos_entrenamiento = torch.tensor([[4.0]], dtype=torch.float32, device="cuda")

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert distancias.shape == (1, 1)
    assert distancias.dtype == torch.float32
    assert distancias.device.type == "cuda"
    assert torch.equal(distancias.cpu(), torch.tensor([[9.0]], dtype=torch.float32))


def test_distancias_l2_cuadradas_cuda_coincide_con_numpy_y_cpu() -> None:
    datos_consulta_numpy = np.array([[0.0, -1.0], [2.0, 3.0]], dtype=np.float32)
    datos_entrenamiento_numpy = np.array(
        [[0.0, -1.0], [1.0, 1.0], [2.0, 3.0]], dtype=np.float32
    )
    esperado = distancias_l2_cuadradas(
        datos_consulta_numpy, datos_entrenamiento_numpy
    )
    datos_consulta_cpu = torch.from_numpy(datos_consulta_numpy)
    datos_entrenamiento_cpu = torch.from_numpy(datos_entrenamiento_numpy)

    distancias_cpu = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta_cpu, datos_entrenamiento_cpu
    )
    distancias_cuda = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta_cpu.cuda(), datos_entrenamiento_cpu.cuda()
    )

    np.testing.assert_allclose(distancias_cuda.cpu().numpy(), esperado, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        distancias_cuda.cpu().numpy(), distancias_cpu.numpy(), rtol=1e-4, atol=1e-5
    )


def test_distancias_l2_cuadradas_cuda_maneja_datos_duplicados_y_distancia_cero() -> None:
    datos_consulta = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device="cuda")
    datos_entrenamiento = torch.tensor(
        [[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32, device="cuda"
    )

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert torch.equal(distancias.cpu(), torch.zeros((1, 2), dtype=torch.float32))


def test_distancias_l2_cuadradas_cuda_acepta_vistas_no_contiguas() -> None:
    datos_consulta = torch.tensor(
        [[0.0, 2.0], [0.0, 0.0]], dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    datos_entrenamiento = torch.tensor(
        [[0.0, 2.0, 4.0], [0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda"
    ).transpose(0, 1)
    esperado = distancias_l2_cuadradas(
        datos_consulta.cpu().numpy(), datos_entrenamiento.cpu().numpy()
    )

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert not datos_consulta.is_contiguous()
    assert not datos_entrenamiento.is_contiguous()
    np.testing.assert_allclose(distancias.cpu().numpy(), esperado, rtol=1e-4, atol=1e-5)


def test_distancias_l2_cuadradas_cuda_es_determinista_y_no_modifica_las_entradas() -> None:
    datos_consulta = torch.tensor([[1.0], [-2.0]], dtype=torch.float32, device="cuda")
    datos_entrenamiento = torch.tensor(
        [[0.0], [3.0]], dtype=torch.float32, device="cuda"
    )
    datos_consulta_antes = datos_consulta.clone()
    datos_entrenamiento_antes = datos_entrenamiento.clone()

    primeras_distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )
    segundas_distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert torch.equal(primeras_distancias, segundas_distancias)
    assert torch.equal(datos_consulta, datos_consulta_antes)
    assert torch.equal(datos_entrenamiento, datos_entrenamiento_antes)


def test_distancias_l2_cuadradas_cuda_funciona_con_menos_de_un_bloque() -> None:
    datos_consulta = torch.zeros((1, 1), dtype=torch.float32, device="cuda")
    datos_entrenamiento = torch.zeros((3, 1), dtype=torch.float32, device="cuda")

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert distancias.shape == (1, 3)


def test_distancias_l2_cuadradas_cuda_funciona_con_mas_de_un_bloque() -> None:
    datos_consulta = torch.zeros((17, 1), dtype=torch.float32, device="cuda")
    datos_entrenamiento = torch.zeros((17, 1), dtype=torch.float32, device="cuda")

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert distancias.shape == (17, 17)
    assert torch.equal(distancias, torch.zeros((17, 17), dtype=torch.float32, device="cuda"))


def test_distancias_l2_cuadradas_cuda_maneja_bordes_de_baldosas_y_caracteristicas() -> None:
    datos_consulta_numpy = np.arange(17 * 33, dtype=np.float32).reshape(17, 33)
    datos_entrenamiento_numpy = np.arange(
        19 * 33, dtype=np.float32
    ).reshape(19, 33)
    esperado = distancias_l2_cuadradas(
        datos_consulta_numpy, datos_entrenamiento_numpy
    )

    distancias_cuda = torch.ops.knn_cuda.distancias_l2_cuadradas(
        torch.from_numpy(datos_consulta_numpy).cuda(),
        torch.from_numpy(datos_entrenamiento_numpy).cuda(),
    )

    assert distancias_cuda.shape == (17, 19)
    np.testing.assert_allclose(
        distancias_cuda.cpu().numpy(), esperado, rtol=1e-4, atol=1e-5
    )
