import importlib

import torch


OPERADORES_REQUERIDOS = (
    "distancias_l2_cuadradas",
    "seleccionar_top_k",
    "predecir_knn",
)


def _obtener_operador_sin_kernel(clave_despacho: str) -> str | None:
    importlib.import_module("knn_cuda._backend_cpp")

    for operador in OPERADORES_REQUERIDOS:
        nombre_operador = f"knn_cuda::{operador}"
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            nombre_operador, clave_despacho
        ):
            return nombre_operador
    return None


def _tiene_kernels_registrados(clave_despacho: str) -> bool:
    return _obtener_operador_sin_kernel(clave_despacho) is None


def _verificar_kernels_registrados(clave_despacho: str) -> None:
    nombre_operador = _obtener_operador_sin_kernel(clave_despacho)
    if nombre_operador is None:
        return

    raise RuntimeError(
        f"el backend {clave_despacho} no tiene un kernel registrado para {nombre_operador}"
    )


def verificar_backend_cpu() -> None:
    _verificar_kernels_registrados("CPU")


def tiene_backend_cuda() -> bool:
    return _tiene_kernels_registrados("CUDA")


def verificar_backend_cuda() -> None:
    _verificar_kernels_registrados("CUDA")
