import importlib
from pathlib import Path

import pytest
import torch


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
