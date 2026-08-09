"""Perfila una ejecucion nativa completa de predecir_knn en CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarks.medicion import ESCENARIOS, generar_datos, obtener_escenarios
from benchmarks.perfilado import ejecutar_perfilado_cuda, validar_opciones_perfilado
from knn_cuda._backend_nativo import tiene_backend_cuda
from knn_cuda.referencia import predecir_knn


def perfilar_pipeline_cuda(
    nombre_escenario: str = "grande",
    calentamiento: int = 10,
    filas: int = 30,
    ruta_traza: Path | None = None,
) -> None:
    """Perfila el pipeline nativo CUDA y valida su resultado antes de medir."""
    validar_opciones_perfilado(calentamiento, filas)

    if not torch.cuda.is_available():
        print("CUDA no esta disponible, el profiling del pipeline se omitio")
        return

    if not tiene_backend_cuda():
        print("El backend CUDA de KNN-Cuda no esta disponible, el profiling se omitio")
        return

    escenario = obtener_escenarios(nombre_escenario)[0]
    datos = generar_datos(escenario)
    dispositivo = torch.device("cuda", torch.cuda.current_device())

    predicciones_esperadas = predecir_knn(
        datos.datos_entrenamiento,
        datos.etiquetas_entrenamiento,
        datos.datos_consulta,
        escenario.numero_vecinos,
    )
    datos_entrenamiento = torch.from_numpy(datos.datos_entrenamiento).to(dispositivo)
    etiquetas_entrenamiento = torch.from_numpy(datos.etiquetas_entrenamiento).to(dispositivo)
    datos_consulta = torch.from_numpy(datos.datos_consulta).to(dispositivo)

    predicciones = torch.ops.knn_cuda.predecir_knn(
        datos_entrenamiento,
        etiquetas_entrenamiento,
        datos_consulta,
        escenario.numero_vecinos,
    )
    np.testing.assert_array_equal(predicciones.cpu().numpy(), predicciones_esperadas)
    del predicciones

    for _ in range(calentamiento):
        torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento,
            etiquetas_entrenamiento,
            datos_consulta,
            escenario.numero_vecinos,
        )
    torch.cuda.synchronize(dispositivo)

    print(
        f"Profiling pipeline CUDA | escenario={escenario.nombre} | "
        f"N={escenario.numero_muestras} | Q={escenario.numero_consultas} | "
        f"D={escenario.numero_caracteristicas} | k={escenario.numero_vecinos} | "
        f"semilla={escenario.semilla} | calentamiento={calentamiento}"
    )
    print(
        "Temporales esperados: distancias [Q, N], seleccionados [Q, N], "
        "distancias seleccionadas [Q, k], indices [Q, k], etiquetas vecinas [Q, k]"
    )

    predicciones, memoria = ejecutar_perfilado_cuda(
        lambda: torch.ops.knn_cuda.predecir_knn(
            datos_entrenamiento,
            etiquetas_entrenamiento,
            datos_consulta,
            escenario.numero_vecinos,
        ),
        "knn_cuda::predecir_knn",
        dispositivo,
        filas,
        ruta_traza,
    )
    del predicciones
    print(
        "Memoria CUDA | asignada={:.3f} MiB | pico adicional={:.3f} MiB".format(
            memoria.asignada_mib,
            memoria.pico_adicional_mib,
        )
    )


def crear_argumentos() -> argparse.ArgumentParser:
    """Crea los argumentos de linea de comandos del perfilador."""
    analizador = argparse.ArgumentParser(
        description="Perfila el pipeline CUDA nativo de KNN-Cuda"
    )
    analizador.add_argument(
        "--escenario",
        choices=tuple(ESCENARIOS),
        default="grande",
        help="Escenario reproducible que se perfilara",
    )
    analizador.add_argument(
        "--calentamiento",
        type=int,
        default=10,
        help="Llamadas previas que no se incluyen en la captura",
    )
    analizador.add_argument(
        "--filas",
        type=int,
        default=30,
        help="Cantidad maxima de filas por tabla del profiler",
    )
    analizador.add_argument(
        "--traza",
        type=Path,
        help="Ruta opcional para exportar una traza Chrome",
    )
    return analizador


def main() -> None:
    """Ejecuta el perfilador desde la linea de comandos."""
    argumentos = crear_argumentos().parse_args()
    perfilar_pipeline_cuda(
        argumentos.escenario,
        argumentos.calentamiento,
        argumentos.filas,
        argumentos.traza,
    )


if __name__ == "__main__":
    main()
