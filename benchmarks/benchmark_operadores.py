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
    medir_memoria_cuda,
    obtener_escenarios,
)
from knn_cuda._backend_nativo import tiene_backend_cuda, verificar_backend_cpu
from knn_cuda.referencia import (
    distancias_l2_cuadradas,
    predecir_knn,
    seleccionar_top_k,
    votacion_uniforme,
)


def medir_operadores(
    nombre_escenario: str,
    calentamiento: int,
    repeticiones: int,
    incluir_cuda: bool,
) -> list[ResultadoMedicion]:
    escenario = obtener_escenarios(nombre_escenario)[0]
    datos = generar_datos(escenario)
    verificar_backend_cpu()
    consulta_cpu = torch.from_numpy(datos.datos_consulta)
    entrenamiento_cpu = torch.from_numpy(datos.datos_entrenamiento)
    etiquetas_cpu = torch.from_numpy(datos.etiquetas_entrenamiento)
    vecinos_cpu = torch.from_numpy(datos.etiquetas_vecinos)
    distancias_numpy = distancias_l2_cuadradas(
        datos.datos_consulta, datos.datos_entrenamiento
    )
    distancias_seleccionadas_numpy, indices_seleccionados_numpy = seleccionar_top_k(
        distancias_numpy, escenario.numero_vecinos
    )
    predicciones_numpy = votacion_uniforme(datos.etiquetas_vecinos)
    distancias_cpu = torch.from_numpy(distancias_numpy)

    np.testing.assert_allclose(
        torch.ops.knn_cuda.distancias_l2_cuadradas(
            consulta_cpu, entrenamiento_cpu
        ).numpy(),
        distancias_numpy,
        rtol=1e-6,
        atol=1e-6,
    )
    distancias_seleccionadas_cpu, indices_seleccionados_cpu = (
        torch.ops.knn_cuda.seleccionar_top_k(
            distancias_cpu, escenario.numero_vecinos
        )
    )
    np.testing.assert_array_equal(
        distancias_seleccionadas_cpu.numpy(), distancias_seleccionadas_numpy
    )
    np.testing.assert_array_equal(
        indices_seleccionados_cpu.numpy(), indices_seleccionados_numpy
    )
    np.testing.assert_array_equal(
        torch.ops.knn_cuda.votacion_uniforme(vecinos_cpu).numpy(), predicciones_numpy
    )
    np.testing.assert_array_equal(
        torch.ops.knn_cuda.predecir_knn(
            entrenamiento_cpu, etiquetas_cpu, consulta_cpu, escenario.numero_vecinos
        ).numpy(),
        predecir_knn(
            datos.datos_entrenamiento,
            datos.etiquetas_entrenamiento,
            datos.datos_consulta,
            escenario.numero_vecinos,
        ),
    )

    funciones_numpy = {
        "distancias": lambda: distancias_l2_cuadradas(
            datos.datos_consulta, datos.datos_entrenamiento
        ),
        "top_k": lambda: seleccionar_top_k(distancias_numpy, escenario.numero_vecinos),
        "votacion": lambda: votacion_uniforme(datos.etiquetas_vecinos),
        "pipeline": lambda: predecir_knn(
            datos.datos_entrenamiento,
            datos.etiquetas_entrenamiento,
            datos.datos_consulta,
            escenario.numero_vecinos,
        ),
    }
    funciones_cpu = {
        "distancias": lambda: torch.ops.knn_cuda.distancias_l2_cuadradas(
            consulta_cpu, entrenamiento_cpu
        ),
        "top_k": lambda: torch.ops.knn_cuda.seleccionar_top_k(
            distancias_cpu, escenario.numero_vecinos
        ),
        "votacion": lambda: torch.ops.knn_cuda.votacion_uniforme(vecinos_cpu),
        "pipeline": lambda: torch.ops.knn_cuda.predecir_knn(
            entrenamiento_cpu, etiquetas_cpu, consulta_cpu, escenario.numero_vecinos
        ),
    }
    resultados = []
    tiempos_cpu = {}
    for etapa, funcion in funciones_numpy.items():
        estadisticas = medir_cpu(funcion, calentamiento, repeticiones)
        resultados.append(ResultadoMedicion(escenario.nombre, "numpy", etapa, estadisticas))
    for etapa, funcion in funciones_cpu.items():
        estadisticas = medir_cpu(funcion, calentamiento, repeticiones)
        tiempos_cpu[etapa] = estadisticas
        resultados.append(ResultadoMedicion(escenario.nombre, "cpp_cpu", etapa, estadisticas))

    if not incluir_cuda or not torch.cuda.is_available() or not tiene_backend_cuda():
        return resultados

    dispositivo = torch.device("cuda", torch.cuda.current_device())
    consulta_cuda = consulta_cpu.to(dispositivo)
    entrenamiento_cuda = entrenamiento_cpu.to(dispositivo)
    etiquetas_cuda = etiquetas_cpu.to(dispositivo)
    vecinos_cuda = vecinos_cpu.to(dispositivo)
    distancias_cuda = distancias_cpu.to(dispositivo)
    funciones_cuda = {
        "distancias": lambda: torch.ops.knn_cuda.distancias_l2_cuadradas(
            consulta_cuda, entrenamiento_cuda
        ),
        "top_k": lambda: torch.ops.knn_cuda.seleccionar_top_k(
            distancias_cuda, escenario.numero_vecinos
        ),
        "votacion": lambda: torch.ops.knn_cuda.votacion_uniforme(vecinos_cuda),
        "pipeline": lambda: torch.ops.knn_cuda.predecir_knn(
            entrenamiento_cuda, etiquetas_cuda, consulta_cuda, escenario.numero_vecinos
        ),
    }
    np.testing.assert_allclose(
        funciones_cuda["distancias"]().cpu().numpy(),
        distancias_numpy,
        rtol=1e-6,
        atol=1e-6,
    )
    distancias_seleccionadas_cuda, indices_seleccionados_cuda = funciones_cuda[
        "top_k"
    ]()
    np.testing.assert_array_equal(
        distancias_seleccionadas_cuda.cpu().numpy(), distancias_seleccionadas_numpy
    )
    np.testing.assert_array_equal(
        indices_seleccionados_cuda.cpu().numpy(), indices_seleccionados_numpy
    )
    np.testing.assert_array_equal(
        funciones_cuda["votacion"]().cpu().numpy(), predicciones_numpy
    )
    for etapa, funcion in funciones_cuda.items():
        resultado_cuda = funcion()
        if etapa == "pipeline":
            np.testing.assert_array_equal(
                resultado_cuda.cpu().numpy(),
                predecir_knn(
                    datos.datos_entrenamiento,
                    datos.etiquetas_entrenamiento,
                    datos.datos_consulta,
                    escenario.numero_vecinos,
                ),
            )
        estadisticas = medir_cuda_eventos(funcion, calentamiento, repeticiones)
        memoria = medir_memoria_cuda(funcion)
        resultados.append(
            ResultadoMedicion(escenario.nombre, "cuda", etapa, estadisticas, memoria)
        )
        imprimir_aceleracion(escenario.nombre, etapa, tiempos_cpu[etapa], estadisticas)
    return resultados


def main() -> None:
    argumentos = argparse.ArgumentParser()
    argumentos.add_argument(
        "--escenario", choices=(*ESCENARIOS, "todos"), default="pequeno"
    )
    argumentos.add_argument("--calentamiento", type=int, default=3)
    argumentos.add_argument("--repeticiones", type=int, default=15)
    argumentos.add_argument("--solo-cpu", action="store_true")
    opciones = argumentos.parse_args()

    resultados = []
    for escenario in obtener_escenarios(opciones.escenario):
        resultados.extend(
            medir_operadores(
                escenario.nombre,
                opciones.calentamiento,
                opciones.repeticiones,
                not opciones.solo_cpu,
            )
        )
    imprimir_resultados(resultados)


if __name__ == "__main__":
    main()
