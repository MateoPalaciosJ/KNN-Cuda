import importlib

import torch


OPERADORES_CPU_REQUERIDOS = (
    "distancias_l2_cuadradas",
    "seleccionar_top_k",
    "predecir_knn",
)


def verificar_backend_cpu() -> None:
    importlib.import_module("knn_cuda._backend_cpp")

    for operador in OPERADORES_CPU_REQUERIDOS:
        nombre_operador = f"knn_cuda::{operador}"
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            nombre_operador, "CPU"
        ):
            raise RuntimeError(
                f"el backend CPU no tiene un kernel registrado para {nombre_operador}"
            )
