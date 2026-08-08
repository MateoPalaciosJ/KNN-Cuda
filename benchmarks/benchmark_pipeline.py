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
    medir_cuda_eventos,
    medir_cuda_pared,
    obtener_escenarios,
)
from knn_cuda._backend_nativo import tiene_backend_cuda, verificar_backend_cpu
from knn_cuda.referencia import predecir_knn


def medir_pipeline(
    escenario_nombre: str,
    calentamiento: int,
    repeticiones: int,
) -> list[ResultadoMedicion]:
    escenario = obtener_escenarios(escenario_nombre)[0]
    datos = generar_datos(escenario)
    verificar_backend_cpu()
    entrenamiento_cpu = torch.from_numpy(datos.datos_entrenamiento)
    etiquetas_cpu = torch.from_numpy(datos.etiquetas_entrenamiento)
    consulta_cpu = torch.from_numpy(datos.datos_consulta)
    esperado = predecir_knn(
        datos.datos_entrenamiento,
        datos.etiquetas_entrenamiento,
        datos.datos_consulta,
        escenario.numero_vecinos,
    )
    funcion_numpy = lambda: predecir_knn(
        datos.datos_entrenamiento,
        datos.etiquetas_entrenamiento,
        datos.datos_consulta,
        escenario.numero_vecinos,
    )
    funcion_cpu = lambda: torch.ops.knn_cuda.predecir_knn(
        entrenamiento_cpu, etiquetas_cpu, consulta_cpu, escenario.numero_vecinos
    )
    np.testing.assert_array_equal(funcion_cpu().numpy(), esperado)
    tiempo_numpy = medir_cpu(funcion_numpy, calentamiento, repeticiones)
    tiempo_cpu = medir_cpu(funcion_cpu, calentamiento, repeticiones)
    resultados = [
        ResultadoMedicion(escenario.nombre, "numpy", "pipeline", tiempo_numpy),
        ResultadoMedicion(escenario.nombre, "cpp_cpu", "pipeline", tiempo_cpu),
    ]

    if not torch.cuda.is_available() or not tiene_backend_cuda():
        return resultados

    dispositivo = torch.device("cuda", torch.cuda.current_device())
    entrenamiento_cuda = entrenamiento_cpu.to(dispositivo)
    etiquetas_cuda = etiquetas_cpu.to(dispositivo)
    consulta_cuda = consulta_cpu.to(dispositivo)
    funcion_cuda = lambda: torch.ops.knn_cuda.predecir_knn(
        entrenamiento_cuda, etiquetas_cuda, consulta_cuda, escenario.numero_vecinos
    )
    np.testing.assert_array_equal(funcion_cuda().cpu().numpy(), esperado)
    tiempo_cuda = medir_cuda_eventos(funcion_cuda, calentamiento, repeticiones)
    resultados.append(
        ResultadoMedicion(escenario.nombre, "cuda", "calculo", tiempo_cuda)
    )
    imprimir_aceleracion(escenario.nombre, "pipeline", tiempo_cpu, tiempo_cuda)

    tiempo_preparacion = medir_cpu(
        lambda: (
            torch.from_numpy(datos.datos_entrenamiento),
            torch.from_numpy(datos.etiquetas_entrenamiento),
            torch.from_numpy(datos.datos_consulta),
        ),
        calentamiento,
        repeticiones,
    )
    tiempo_cpu_gpu = medir_cuda_pared(
        lambda: (
            entrenamiento_cpu.to(dispositivo),
            etiquetas_cpu.to(dispositivo),
            consulta_cpu.to(dispositivo),
        ),
        calentamiento,
        repeticiones,
    )
    predicciones_cuda = funcion_cuda()
    tiempo_gpu_cpu = medir_cuda_pared(
        lambda: predicciones_cuda.detach().cpu(), calentamiento, repeticiones
    )
    tiempo_extremo_a_extremo = medir_cuda_pared(
        lambda: funcion_cuda().detach().cpu(), calentamiento, repeticiones
    )
    resultados.extend(
        [
            ResultadoMedicion(
                escenario.nombre, "python", "preparacion_tensores", tiempo_preparacion
            ),
            ResultadoMedicion(
                escenario.nombre, "cuda", "transferencia_cpu_gpu", tiempo_cpu_gpu
            ),
            ResultadoMedicion(
                escenario.nombre, "cuda", "transferencia_gpu_cpu", tiempo_gpu_cpu
            ),
            ResultadoMedicion(
                escenario.nombre,
                "cuda",
                "calculo_y_salida",
                tiempo_extremo_a_extremo,
            ),
        ]
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
            medir_pipeline(
                escenario.nombre, opciones.calentamiento, opciones.repeticiones
            )
        )
    imprimir_resultados(resultados)


if __name__ == "__main__":
    main()
