from pathlib import Path
import re
import subprocess
import sys
import warnings

from setuptools import setup

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
except ModuleNotFoundError as error_importacion:
    if error_importacion.name == "torch":
        raise RuntimeError(
            "PyTorch debe estar instalado antes de compilar KNN-Cuda desde fuente"
        ) from error_importacion
    raise


raiz_proyecto = Path(__file__).resolve().parent
fuentes_cpp = [
    "src/cpp/operadores.cpp",
    "src/cpp/registro.cpp",
]
fuentes_cuda = [
    "src/cuda/distancias_l2_cuadradas.cu",
    "src/cuda/seleccionar_top_k.cu",
    "src/cuda/votacion_uniforme.cu",
]
directorios_include = [
    str(raiz_proyecto / "src" / "cpp" / "include"),
    str(raiz_proyecto / "src" / "cuda" / "include"),
]


def obtener_version_toolkit_cuda() -> str | None:
    if CUDA_HOME is None:
        return None

    nombre_nvcc = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    ruta_nvcc = Path(CUDA_HOME) / "bin" / nombre_nvcc

    try:
        salida = subprocess.check_output(
            [str(ruta_nvcc), "--version"], text=True, stderr=subprocess.STDOUT
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    coincidencia = re.search(r"release (\d+\.\d+)", salida)
    return coincidencia.group(1) if coincidencia else None


def toolkit_cuda_disponible(version_toolkit_cuda: str | None) -> bool:
    return version_toolkit_cuda is not None and version_toolkit_cuda == torch.version.cuda


def crear_extension_nativa():
    argumentos = {
        "name": "knn_cuda._backend_cpp",
        "sources": fuentes_cpp,
        "include_dirs": directorios_include,
    }

    if torch.version.cuda is None:
        return CppExtension(**argumentos)

    version_toolkit_cuda = obtener_version_toolkit_cuda()

    if version_toolkit_cuda is None:
        warnings.warn(
            "PyTorch tiene soporte CUDA pero no se encontro un CUDA Toolkit utilizable, se compilara solo el backend CPU",
            stacklevel=2,
        )
        return CppExtension(**argumentos)

    if not toolkit_cuda_disponible(version_toolkit_cuda):
        raise RuntimeError(
            "La version del CUDA Toolkit no coincide con la version CUDA de PyTorch"
        )

    argumentos["sources"] = fuentes_cpp + fuentes_cuda
    argumentos["define_macros"] = [("KNN_CUDA_CON_CUDA", "1")]
    return CUDAExtension(**argumentos)


setup(
    ext_modules=[
        crear_extension_nativa(),
    ],
    cmdclass={"build_ext": BuildExtension},
)
