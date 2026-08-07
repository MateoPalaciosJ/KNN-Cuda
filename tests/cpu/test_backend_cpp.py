import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from knn_cuda.referencia import distancias_l2_cuadradas, seleccionar_top_k


def test_backend_cpp_se_importa_como_extension_compilada() -> None:
    backend_cpp = importlib.import_module("knn_cuda._backend_cpp")

    assert Path(backend_cpp.__file__).suffix in {".pyd", ".so"}


def test_verificar_backend_cpu_devuelve_verdadero() -> None:
    importlib.import_module("knn_cuda._backend_cpp")

    assert torch.ops.knn_cuda.verificar_backend_cpu() is True


def test_sumar_vectores_devuelve_la_suma_esperada() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    primer_vector = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float32)
    segundo_vector = torch.tensor([3.0, 5.0, 0.0], dtype=torch.float32)
    esperado = torch.tensor([4.0, 3.0, 0.0], dtype=torch.float32)

    resultado = torch.ops.knn_cuda.sumar_vectores(primer_vector, segundo_vector)

    assert resultado.dtype == torch.float32
    assert torch.equal(resultado, esperado)


def test_sumar_vectores_no_modifica_las_entradas() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    primer_vector = torch.tensor([1.0, 2.0], dtype=torch.float32)
    segundo_vector = torch.tensor([3.0, 4.0], dtype=torch.float32)
    primer_vector_antes = primer_vector.clone()
    segundo_vector_antes = segundo_vector.clone()

    torch.ops.knn_cuda.sumar_vectores(primer_vector, segundo_vector)

    assert torch.equal(primer_vector, primer_vector_antes)
    assert torch.equal(segundo_vector, segundo_vector_antes)


def test_sumar_vectores_rechaza_float64() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    primer_vector = torch.tensor([1.0], dtype=torch.float64)
    segundo_vector = torch.tensor([2.0], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="primer_vector"):
        torch.ops.knn_cuda.sumar_vectores(primer_vector, segundo_vector)


def test_sumar_vectores_rechaza_enteros() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    primer_vector = torch.tensor([1], dtype=torch.int64)
    segundo_vector = torch.tensor([2], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="primer_vector"):
        torch.ops.knn_cuda.sumar_vectores(primer_vector, segundo_vector)


def test_sumar_vectores_rechaza_vectores_bidimensionales() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    primer_vector = torch.tensor([[1.0]], dtype=torch.float32)
    segundo_vector = torch.tensor([[2.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="primer_vector"):
        torch.ops.knn_cuda.sumar_vectores(primer_vector, segundo_vector)


def test_sumar_vectores_rechaza_vectores_vacios() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    primer_vector = torch.empty(0, dtype=torch.float32)
    segundo_vector = torch.empty(0, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="primer_vector"):
        torch.ops.knn_cuda.sumar_vectores(primer_vector, segundo_vector)


def test_sumar_vectores_rechaza_longitudes_distintas() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    primer_vector = torch.tensor([1.0], dtype=torch.float32)
    segundo_vector = torch.tensor([2.0, 3.0], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="misma longitud"):
        torch.ops.knn_cuda.sumar_vectores(primer_vector, segundo_vector)


def test_distancias_l2_cuadradas_calcula_una_consulta_y_una_muestra() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    esperado = torch.tensor([[25.0]], dtype=torch.float32)

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert distancias.device.type == "cpu"
    assert distancias.dtype == torch.float32
    assert distancias.shape == (1, 1)
    assert torch.equal(distancias, esperado)


def test_distancias_l2_cuadradas_calcula_varias_consultas_y_muestras() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[0.0, -1.0], [2.0, 3.0]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor(
        [[0.0, -1.0], [0.0, -1.0], [-1.0, 2.0]], dtype=torch.float32
    )
    esperado = torch.tensor(
        [[0.0, 0.0, 10.0], [20.0, 20.0, 10.0]], dtype=torch.float32
    )

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert distancias.shape == (2, 3)
    assert torch.equal(distancias, esperado)


def test_distancias_l2_cuadradas_calcula_con_una_caracteristica() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[1.0], [-2.0]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[1.0], [3.0]], dtype=torch.float32)
    esperado = torch.tensor([[0.0, 4.0], [9.0, 25.0]], dtype=torch.float32)

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert torch.equal(distancias, esperado)


def test_distancias_l2_cuadradas_coincide_con_la_referencia_numpy() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta_numpy = np.array(
        [[0.5, -1.0, 2.0], [3.0, 0.0, -2.0]], dtype=np.float32
    )
    datos_entrenamiento_numpy = np.array(
        [[0.0, -1.0, 1.0], [2.0, 3.0, -2.0], [-1.0, 4.0, 0.0]],
        dtype=np.float32,
    )
    esperado = distancias_l2_cuadradas(
        datos_consulta_numpy, datos_entrenamiento_numpy
    )
    datos_consulta = torch.from_numpy(datos_consulta_numpy)
    datos_entrenamiento = torch.from_numpy(datos_entrenamiento_numpy)

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    np.testing.assert_allclose(
        distancias.numpy(), esperado, rtol=1e-6, atol=1e-6
    )


def test_distancias_l2_cuadradas_acepta_vistas_no_contiguas() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor(
        [[0.0, 2.0], [1.0, 3.0]], dtype=torch.float32
    ).transpose(0, 1)
    datos_entrenamiento = torch.tensor(
        [[0.0, 4.0], [2.0, 6.0]], dtype=torch.float32
    ).transpose(0, 1)
    esperado = distancias_l2_cuadradas(
        datos_consulta.numpy(), datos_entrenamiento.numpy()
    )

    distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    assert not datos_consulta.is_contiguous()
    assert not datos_entrenamiento.is_contiguous()
    np.testing.assert_allclose(
        distancias.numpy(), esperado, rtol=1e-6, atol=1e-6
    )


def test_distancias_l2_cuadradas_es_determinista_y_no_modifica_entradas() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[1.0, 0.0], [4.0, 5.0]], dtype=torch.float32)
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


def test_distancias_l2_cuadradas_rechaza_datos_consulta_float64() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[1.0]], dtype=torch.float64)
    datos_entrenamiento = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="datos_consulta"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_datos_entrenamiento_float64() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[1.0]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[1.0]], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="datos_entrenamiento"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_datos_enteros() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[1]], dtype=torch.int64)
    datos_entrenamiento = torch.tensor([[1]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="datos_consulta"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_datos_consulta_unidimensionales() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([1.0], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="datos_consulta"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_datos_entrenamiento_tridimensionales() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[1.0]], dtype=torch.float32)
    datos_entrenamiento = torch.ones((1, 1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="datos_entrenamiento"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_datos_vacios() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.empty((0, 1), dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="datos_consulta"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_caracteristicas_distintas() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="misma cantidad de caracteristicas"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_nan() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[float("nan")]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="datos_consulta"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_distancias_l2_cuadradas_rechaza_infinito() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    datos_consulta = torch.tensor([[1.0]], dtype=torch.float32)
    datos_entrenamiento = torch.tensor([[float("inf")]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="datos_entrenamiento"):
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta, datos_entrenamiento
        )


def test_seleccionar_top_k_funciona_con_k_igual_a_uno() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]], dtype=torch.float32)
    distancias_esperadas = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
    indices_esperados = torch.tensor([[1], [1]], dtype=torch.int64)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 1
    )

    assert distancias_seleccionadas.shape == (2, 1)
    assert indices_seleccionados.shape == (2, 1)
    assert distancias_seleccionadas.dtype == torch.float32
    assert indices_seleccionados.dtype == torch.int64
    assert torch.equal(distancias_seleccionadas, distancias_esperadas)
    assert torch.equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_funciona_con_k_igual_a_n() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[2.0, 1.0, 3.0]], dtype=torch.float32)
    distancias_esperadas = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    indices_esperados = torch.tensor([[1, 0, 2]], dtype=torch.int64)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 3
    )

    assert torch.equal(distancias_seleccionadas, distancias_esperadas)
    assert torch.equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_funciona_con_una_muestra() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[4.0], [1.0]], dtype=torch.float32)
    distancias_esperadas = torch.tensor([[4.0], [1.0]], dtype=torch.float32)
    indices_esperados = torch.tensor([[0], [0]], dtype=torch.int64)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 1
    )

    assert torch.equal(distancias_seleccionadas, distancias_esperadas)
    assert torch.equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_ordena_distancias_de_forma_ascendente() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[9.0, 1.0, 5.0, 3.0]], dtype=torch.float32)
    distancias_esperadas = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float32)
    indices_esperados = torch.tensor([[1, 3, 2]], dtype=torch.int64)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 3
    )

    assert torch.equal(distancias_seleccionadas, distancias_esperadas)
    assert torch.equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_resuelve_empates_por_menor_indice() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[4.0, 1.0, 1.0, 3.0]], dtype=torch.float32)
    distancias_esperadas = torch.tensor([[1.0, 1.0, 3.0]], dtype=torch.float32)
    indices_esperados = torch.tensor([[1, 2, 3]], dtype=torch.int64)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 3
    )

    assert torch.equal(distancias_seleccionadas, distancias_esperadas)
    assert torch.equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_resuelve_multiples_empates_por_menor_indice() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor(
        [[2.0, 1.0, 1.0, 2.0], [3.0, 3.0, 0.0, 0.0]], dtype=torch.float32
    )
    distancias_esperadas = torch.tensor(
        [[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 3.0, 3.0]], dtype=torch.float32
    )
    indices_esperados = torch.tensor([[1, 2, 0, 3], [2, 3, 0, 1]], dtype=torch.int64)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 4
    )

    assert torch.equal(distancias_seleccionadas, distancias_esperadas)
    assert torch.equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_coincide_con_la_referencia_numpy() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias_numpy = np.array(
        [[5.0, 1.0, 1.0, 3.0], [2.0, 4.0, 0.0, 2.0]], dtype=np.float32
    )
    distancias_esperadas, indices_esperados = seleccionar_top_k(distancias_numpy, 3)
    distancias = torch.from_numpy(distancias_numpy)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 3
    )

    np.testing.assert_array_equal(distancias_seleccionadas.numpy(), distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados.numpy(), indices_esperados)


def test_seleccionar_top_k_acepta_vistas_no_contiguas() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor(
        [[4.0, 2.0], [1.0, 5.0], [1.0, 0.0]], dtype=torch.float32
    ).transpose(0, 1)
    distancias_esperadas, indices_esperados = seleccionar_top_k(distancias.numpy(), 2)

    distancias_seleccionadas, indices_seleccionados = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, 2
    )

    assert not distancias.is_contiguous()
    np.testing.assert_array_equal(distancias_seleccionadas.numpy(), distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados.numpy(), indices_esperados)


def test_seleccionar_top_k_es_determinista_y_no_modifica_las_distancias() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[2.0, 1.0, 1.0], [3.0, 0.0, 2.0]], dtype=torch.float32)
    distancias_antes = distancias.clone()

    primeras_salidas = torch.ops.knn_cuda.seleccionar_top_k(distancias, 2)
    segundas_salidas = torch.ops.knn_cuda.seleccionar_top_k(distancias, 2)

    assert torch.equal(primeras_salidas[0], segundas_salidas[0])
    assert torch.equal(primeras_salidas[1], segundas_salidas[1])
    assert torch.equal(distancias, distancias_antes)


def test_seleccionar_top_k_rechaza_distancias_float64() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[1.0]], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_enteras() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[1]], dtype=torch.int64)

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_unidimensionales() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([1.0], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_tridimensionales() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.ones((1, 1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_vacias() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.empty((0, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_nan() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[float("nan")]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_infinito() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[float("inf")]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="distancias"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_k_igual_a_cero() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 0)


def test_seleccionar_top_k_rechaza_k_negativo() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, -1)


def test_seleccionar_top_k_rechaza_k_mayor_que_n() -> None:
    importlib.import_module("knn_cuda._backend_cpp")
    distancias = torch.tensor([[1.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="k"):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, 2)
