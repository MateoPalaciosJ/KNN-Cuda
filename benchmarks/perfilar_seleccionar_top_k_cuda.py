from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.medicion import ESCENARIOS, generar_datos, obtener_escenarios
from benchmarks.perfilado import ejecutar_perfilado_cuda, validar_opciones_perfilado
from knn_cuda._backend_nativo import tiene_backend_cuda
from knn_cuda.referencia import distancias_l2_cuadradas, seleccionar_top_k


def perfilar_seleccionar_top_k(
    nombre_escenario: str,
    calentamiento: int,
    filas: int,
    ruta_traza: Path | None,
) -> None:
    validar_opciones_perfilado(calentamiento, filas)
    if not torch.cuda.is_available():
        print("CUDA no esta disponible, el profiling de seleccionar_top_k se omitio")
        return
    if not tiene_backend_cuda():
        print("el backend CUDA de KNN-Cuda no esta registrado, el profiling se omitio")
        return

    escenario = obtener_escenarios(nombre_escenario)[0]
    datos = generar_datos(escenario)
    distancias_numpy = distancias_l2_cuadradas(
        datos.datos_consulta, datos.datos_entrenamiento
    )
    distancias_esperadas, indices_esperados = seleccionar_top_k(
        distancias_numpy, escenario.numero_vecinos
    )
    dispositivo = torch.device("cuda", torch.cuda.current_device())
    distancias = torch.from_numpy(distancias_numpy).to(dispositivo)

    resultado = torch.ops.knn_cuda.seleccionar_top_k(
        distancias, escenario.numero_vecinos
    )
    np.testing.assert_array_equal(resultado[0].cpu().numpy(), distancias_esperadas)
    np.testing.assert_array_equal(resultado[1].cpu().numpy(), indices_esperados)
    del resultado

    for _ in range(calentamiento):
        torch.ops.knn_cuda.seleccionar_top_k(distancias, escenario.numero_vecinos)
    torch.cuda.synchronize(dispositivo)

    print(
        f"escenario={escenario.nombre} N={escenario.numero_muestras} "
        f"Q={escenario.numero_consultas} D={escenario.numero_caracteristicas} "
        f"k={escenario.numero_vecinos}"
    )
    print("llamadas perfiladas=1")
    print(f"distancias_contiguas={distancias.is_contiguous()}")
    resultado, memoria = ejecutar_perfilado_cuda(
        lambda: torch.ops.knn_cuda.seleccionar_top_k(
            distancias, escenario.numero_vecinos
        ),
        "knn_cuda::seleccionar_top_k",
        dispositivo,
        filas,
        ruta_traza,
    )
    print(
        "memoria_cuda_mib "
        f"asignada={memoria.asignada_mib:.2f} "
        f"pico_adicional={memoria.pico_adicional_mib:.2f}"
    )
    del resultado


def main() -> None:
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument(
        "--escenario", choices=tuple(ESCENARIOS), default="grande"
    )
    argumentos.add_argument("--calentamiento", type=int, default=10)
    argumentos.add_argument("--filas", type=int, default=30)
    argumentos.add_argument("--traza", type=Path)
    opciones = argumentos.parse_args()
    perfilar_seleccionar_top_k(
        opciones.escenario,
        opciones.calentamiento,
        opciones.filas,
        opciones.traza,
    )


if __name__ == "__main__":
    main()
