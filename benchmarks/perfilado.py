from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import torch


Resultado = TypeVar("Resultado")


@dataclass(frozen=True)
class MemoriaPerfiladoCuda:
    asignada_mib: float
    pico_adicional_mib: float


def validar_opciones_perfilado(calentamiento: int, filas: int) -> None:
    if calentamiento < 0:
        raise ValueError("calentamiento debe ser mayor o igual que cero")
    if filas < 1:
        raise ValueError("filas debe ser mayor o igual que uno")


def ejecutar_perfilado_cuda(
    funcion: Callable[[], Resultado],
    etiqueta: str,
    dispositivo: torch.device,
    filas: int,
    ruta_traza: Path | None,
) -> tuple[Resultado, MemoriaPerfiladoCuda]:
    torch.cuda.reset_peak_memory_stats(dispositivo)
    memoria_antes = torch.cuda.memory_allocated(dispositivo)
    actividades = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(
        activities=actividades,
        record_shapes=True,
        profile_memory=True,
    ) as perfilador:
        with torch.profiler.record_function(etiqueta):
            resultado = funcion()
        torch.cuda.synchronize(dispositivo)

    memoria_despues = torch.cuda.memory_allocated(dispositivo)
    memoria_pico = torch.cuda.max_memory_allocated(dispositivo)
    memoria = MemoriaPerfiladoCuda(
        (memoria_despues - memoria_antes) / (1024 * 1024),
        max(memoria_pico - memoria_antes, 0) / (1024 * 1024),
    )
    promedios = perfilador.key_averages()
    print("\nActividades CPU ordenadas por tiempo propio")
    print(promedios.table(sort_by="self_cpu_time_total", row_limit=filas))
    print("\nActividades CUDA ordenadas por tiempo propio")
    print(promedios.table(sort_by="self_cuda_time_total", row_limit=filas))

    if ruta_traza is not None:
        perfilador.export_chrome_trace(str(ruta_traza))
        print(f"traza_exportada={ruta_traza}")

    return resultado, memoria
