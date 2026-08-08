from __future__ import annotations

import argparse

import numpy as np
import torch

from benchmarks.medicion import (
    ESCENARIOS,
    ResultadoMedicion,
    generar_datos,
    imprimir_aceleracion,
    imprimir_resultados,
    medir_cpu,
    medir_cuda_pared,
    obtener_escenarios,
)
from knn_cuda._backend_nativo import tiene_backend_cuda, verificar_backend_cpu
from knn_cuda.clasificador import ClasificadorKNNCUDA
from knn_cuda.referencia import predecir_knn


def _crear_clasificador_ajustado(datos, numero_vecinos: int, dispositivo: str):
    clasificador = ClasificadorKNNCUDA(numero_vecinos, dispositivo=dispositivo)
    clasificador.ajustar(datos.datos_entrenamiento, datos.etiquetas_entrenamiento)
    return clasificador


def medir_clasificador(
    nombre_escenario: str,
    calentamiento: int,
    repeticiones: int,
) -> list[ResultadoMedicion]:
    escenario = obtener_escenarios(nombre_escenario)[0]
    datos = generar_datos(escenario)
    verificar_backend_cpu()
    esperado = predecir_knn(
        datos.datos_entrenamiento,
        datos.etiquetas_entrenamiento,
        datos.datos_consulta,
        escenario.numero_vecinos,
    )

    clasificador_cpu = _crear_clasificador_ajustado(
        datos, escenario.numero_vecinos, "cpu"
    )
    np.testing.assert_array_equal(clasificador_cpu.predecir(datos.datos_consulta), esperado)
    tiempo_ajustar_cpu = medir_cpu(
        lambda: _crear_clasificador_ajustado(datos, escenario.numero_vecinos, "cpu"),
        calentamiento,
        repeticiones,
    )
    tiempo_predecir_cpu = medir_cpu(
        lambda: clasificador_cpu.predecir(datos.datos_consulta),
        calentamiento,
        repeticiones,
    )
    resultados = [
        ResultadoMedicion(
            escenario.nombre, "clasificador_cpu", "ajustar", tiempo_ajustar_cpu
        ),
        ResultadoMedicion(
            escenario.nombre, "clasificador_cpu", "predecir", tiempo_predecir_cpu
        ),
    ]

    if not torch.cuda.is_available() or not tiene_backend_cuda():
        return resultados

    clasificador_cuda = _crear_clasificador_ajustado(
        datos, escenario.numero_vecinos, "cuda"
    )
    np.testing.assert_array_equal(clasificador_cuda.predecir(datos.datos_consulta), esperado)
    tiempo_ajustar_cuda = medir_cuda_pared(
        lambda: _crear_clasificador_ajustado(datos, escenario.numero_vecinos, "cuda"),
        calentamiento,
        repeticiones,
    )
    tiempo_predecir_cuda = medir_cuda_pared(
        lambda: clasificador_cuda.predecir(datos.datos_consulta),
        calentamiento,
        repeticiones,
    )
    resultados.extend(
        [
            ResultadoMedicion(
                escenario.nombre,
                "clasificador_cuda",
                "ajustar_extremo_a_extremo",
                tiempo_ajustar_cuda,
            ),
            ResultadoMedicion(
                escenario.nombre,
                "clasificador_cuda",
                "predecir_extremo_a_extremo",
                tiempo_predecir_cuda,
            ),
        ]
    )
    imprimir_aceleracion(
        escenario.nombre, "clasificador_predecir", tiempo_predecir_cpu, tiempo_predecir_cuda
    )
    return resultados


def main() -> None:
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument(
        "--escenario", choices=(*ESCENARIOS, "todos"), default="pequeno"
    )
    argumentos.add_argument("--calentamiento", type=int, default=3)
    argumentos.add_argument("--repeticiones", type=int, default=15)
    opciones = argumentos.parse_args()

    resultados = []
    for escenario in obtener_escenarios(opciones.escenario):
        resultados.extend(
            medir_clasificador(
                escenario.nombre, opciones.calentamiento, opciones.repeticiones
            )
        )
    imprimir_resultados(resultados)


if __name__ == "__main__":
    main()
