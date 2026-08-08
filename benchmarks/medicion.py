from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median, pstdev
from time import perf_counter_ns
from typing import Callable, TypeVar

import numpy as np
import torch


Resultado = TypeVar("Resultado")


@dataclass(frozen=True)
class Escenario:
    nombre: str
    numero_muestras: int
    numero_consultas: int
    numero_caracteristicas: int
    numero_vecinos: int


@dataclass(frozen=True)
class DatosEscenario:
    datos_entrenamiento: np.ndarray
    etiquetas_entrenamiento: np.ndarray
    datos_consulta: np.ndarray
    etiquetas_vecinos: np.ndarray


@dataclass(frozen=True)
class EstadisticasTiempo:
    mediana_milisegundos: float
    promedio_milisegundos: float
    desviacion_milisegundos: float


@dataclass(frozen=True)
class MemoriaCuda:
    asignada_mib: float
    pico_adicional_mib: float


@dataclass(frozen=True)
class ResultadoMedicion:
    escenario: str
    backend: str
    etapa: str
    estadisticas: EstadisticasTiempo
    memoria_cuda: MemoriaCuda | None = None


ESCENARIOS = {
    "pequeno": Escenario("pequeno", 128, 16, 8, 5),
    "mediano": Escenario("mediano", 512, 64, 32, 10),
    "grande": Escenario("grande", 2048, 128, 64, 20),
}


def obtener_escenarios(nombre: str) -> tuple[Escenario, ...]:
    if nombre == "todos":
        return tuple(ESCENARIOS.values())
    return (ESCENARIOS[nombre],)


def generar_datos(escenario: Escenario, semilla: int = 2026) -> DatosEscenario:
    generador = np.random.default_rng(semilla)
    datos_entrenamiento = generador.normal(
        size=(escenario.numero_muestras, escenario.numero_caracteristicas)
    ).astype(np.float32)
    datos_consulta = generador.normal(
        size=(escenario.numero_consultas, escenario.numero_caracteristicas)
    ).astype(np.float32)
    etiquetas_entrenamiento = generador.integers(
        -1000, 1001, size=escenario.numero_muestras, dtype=np.int64
    )
    posiciones_vecinos = generador.integers(
        0,
        escenario.numero_muestras,
        size=(escenario.numero_consultas, escenario.numero_vecinos),
    )
    etiquetas_vecinos = etiquetas_entrenamiento[posiciones_vecinos]
    return DatosEscenario(
        datos_entrenamiento,
        etiquetas_entrenamiento,
        datos_consulta,
        etiquetas_vecinos,
    )


def _calcular_estadisticas(duraciones_milisegundos: list[float]) -> EstadisticasTiempo:
    return EstadisticasTiempo(
        median(duraciones_milisegundos),
        mean(duraciones_milisegundos),
        pstdev(duraciones_milisegundos),
    )


def _validar_parametros_medicion(calentamiento: int, repeticiones: int) -> None:
    if calentamiento < 0:
        raise ValueError("calentamiento debe ser mayor o igual que cero")
    if repeticiones < 1:
        raise ValueError("repeticiones debe ser mayor o igual que uno")


def medir_cpu(
    funcion: Callable[[], Resultado], calentamiento: int, repeticiones: int
) -> EstadisticasTiempo:
    _validar_parametros_medicion(calentamiento, repeticiones)
    for _ in range(calentamiento):
        funcion()

    duraciones_milisegundos = []
    for _ in range(repeticiones):
        inicio = perf_counter_ns()
        funcion()
        fin = perf_counter_ns()
        duraciones_milisegundos.append((fin - inicio) / 1_000_000)
    return _calcular_estadisticas(duraciones_milisegundos)


def medir_cuda_eventos(
    funcion: Callable[[], Resultado], calentamiento: int, repeticiones: int
) -> EstadisticasTiempo:
    _validar_parametros_medicion(calentamiento, repeticiones)
    for _ in range(calentamiento):
        funcion()
    torch.cuda.synchronize()

    duraciones_milisegundos = []
    for _ in range(repeticiones):
        inicio = torch.cuda.Event(enable_timing=True)
        fin = torch.cuda.Event(enable_timing=True)
        inicio.record()
        funcion()
        fin.record()
        fin.synchronize()
        duraciones_milisegundos.append(inicio.elapsed_time(fin))
    return _calcular_estadisticas(duraciones_milisegundos)


def medir_cuda_pared(
    funcion: Callable[[], Resultado], calentamiento: int, repeticiones: int
) -> EstadisticasTiempo:
    _validar_parametros_medicion(calentamiento, repeticiones)
    for _ in range(calentamiento):
        funcion()
    torch.cuda.synchronize()

    duraciones_milisegundos = []
    for _ in range(repeticiones):
        torch.cuda.synchronize()
        inicio = perf_counter_ns()
        funcion()
        torch.cuda.synchronize()
        fin = perf_counter_ns()
        duraciones_milisegundos.append((fin - inicio) / 1_000_000)
    return _calcular_estadisticas(duraciones_milisegundos)


def medir_memoria_cuda(funcion: Callable[[], Resultado]) -> MemoriaCuda:
    torch.cuda.synchronize()
    memoria_antes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    resultado = funcion()
    torch.cuda.synchronize()
    memoria_despues = torch.cuda.memory_allocated()
    memoria_pico = torch.cuda.max_memory_allocated()
    del resultado
    return MemoriaCuda(
        (memoria_despues - memoria_antes) / (1024 * 1024),
        max(memoria_pico - memoria_antes, 0) / (1024 * 1024),
    )


def imprimir_resultados(resultados: list[ResultadoMedicion]) -> None:
    encabezado = (
        "escenario      backend              etapa                    "
        "mediana_ms  promedio_ms  desv_ms  memoria_mib  pico_mib"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for resultado in resultados:
        memoria = resultado.memoria_cuda
        memoria_asignada = "-" if memoria is None else f"{memoria.asignada_mib:.2f}"
        memoria_pico = "-" if memoria is None else f"{memoria.pico_adicional_mib:.2f}"
        estadisticas = resultado.estadisticas
        print(
            f"{resultado.escenario:<14}{resultado.backend:<21}"
            f"{resultado.etapa:<25}{estadisticas.mediana_milisegundos:>10.3f}"
            f"{estadisticas.promedio_milisegundos:>12.3f}"
            f"{estadisticas.desviacion_milisegundos:>9.3f}"
            f"{memoria_asignada:>13}{memoria_pico:>10}"
        )


def imprimir_aceleracion(
    escenario: str, etapa: str, tiempo_cpu: EstadisticasTiempo, tiempo_cuda: EstadisticasTiempo
) -> None:
    if tiempo_cuda.mediana_milisegundos == 0:
        return
    aceleracion = tiempo_cpu.mediana_milisegundos / tiempo_cuda.mediana_milisegundos
    print(f"aceleracion {escenario} {etapa}: {aceleracion:.2f}x CPU/CUDA")
