from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


raiz_proyecto = Path(__file__).resolve().parent


setup(
    ext_modules=[
        CppExtension(
            name="knn_cuda._backend_cpp",
            sources=[
                str(raiz_proyecto / "src" / "cpp" / "operadores.cpp"),
                str(raiz_proyecto / "src" / "cpp" / "registro.cpp"),
            ],
            include_dirs=[str(raiz_proyecto / "src" / "cpp" / "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
