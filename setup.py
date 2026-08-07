from pathlib import Path

from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


raiz_proyecto = Path(__file__).resolve().parent
fuentes_cpp = [
    str(raiz_proyecto / "src" / "cpp" / "operadores.cpp"),
    str(raiz_proyecto / "src" / "cpp" / "registro.cpp"),
]
directorios_include = [
    str(raiz_proyecto / "src" / "cpp" / "include"),
    str(raiz_proyecto / "src" / "cuda" / "include"),
]


def toolkit_cuda_disponible() -> bool:
    return CUDA_HOME is not None and torch.version.cuda is not None


def crear_extension_nativa():
    argumentos = {
        "name": "knn_cuda._backend_cpp",
        "sources": fuentes_cpp,
        "include_dirs": directorios_include,
    }

    if not toolkit_cuda_disponible():
        return CppExtension(**argumentos)

    argumentos["sources"] = fuentes_cpp + [
        str(raiz_proyecto / "src" / "cuda" / "distancias_l2_cuadradas.cu"),
    ]
    argumentos["define_macros"] = [("KNN_CUDA_CON_CUDA", "1")]
    return CUDAExtension(**argumentos)


setup(
    ext_modules=[
        crear_extension_nativa(),
    ],
    cmdclass={"build_ext": BuildExtension},
)
