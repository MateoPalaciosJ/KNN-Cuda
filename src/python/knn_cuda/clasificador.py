import numpy as np
import torch

from ._backend_nativo import verificar_backend_cpu


class ClasificadorKNNCUDA:
    def __init__(self, numero_vecinos: int = 5, dispositivo: str = "cpu") -> None:
        self._validar_numero_vecinos(numero_vecinos)
        self._validar_dispositivo(dispositivo)
        self.numero_vecinos = numero_vecinos
        self.dispositivo = dispositivo
        self.ajustado_ = False

    @staticmethod
    def _validar_numero_vecinos(numero_vecinos: int) -> None:
        if not isinstance(numero_vecinos, (int, np.integer)) or isinstance(
            numero_vecinos, (bool, np.bool_)
        ):
            raise TypeError("numero_vecinos debe ser un entero")
        if numero_vecinos < 1:
            raise ValueError("numero_vecinos debe ser mayor o igual a 1")

    @staticmethod
    def _validar_dispositivo(dispositivo: str) -> None:
        if not isinstance(dispositivo, str):
            raise TypeError("dispositivo debe ser una cadena")
        if dispositivo not in {"cpu", "cuda", "auto"}:
            raise ValueError("dispositivo debe ser cpu, cuda o auto")

    def _validar_dispositivo_ajuste(self) -> None:
        if self.dispositivo != "cpu":
            raise RuntimeError(
                f"dispositivo {self.dispositivo} todavia no esta integrado en ClasificadorKNNCUDA"
            )

    @staticmethod
    def _validar_datos_consulta(datos_consulta: np.ndarray) -> None:
        if not isinstance(datos_consulta, np.ndarray):
            raise TypeError("datos_consulta debe ser un numpy.ndarray")
        if datos_consulta.ndim != 2:
            raise ValueError("datos_consulta debe ser bidimensional")
        if datos_consulta.size == 0:
            raise ValueError("datos_consulta no debe estar vacio")
        if datos_consulta.dtype != np.float32:
            raise TypeError("datos_consulta debe tener dtype float32")

    def _validar_consulta_compatible(self, datos_consulta: np.ndarray) -> None:
        self._validar_datos_consulta(datos_consulta)
        if datos_consulta.shape[1] != self.numero_caracteristicas_:
            raise ValueError(
                "datos_consulta y datos_entrenamiento deben tener el mismo numero de caracteristicas"
            )
        if not np.isfinite(datos_consulta).all():
            raise ValueError("datos_consulta no debe contener NaN ni valores infinitos")

    def ajustar(
        self,
        datos_entrenamiento: np.ndarray,
        etiquetas_entrenamiento: np.ndarray,
    ) -> "ClasificadorKNNCUDA":
        if not isinstance(datos_entrenamiento, np.ndarray):
            raise TypeError("datos_entrenamiento debe ser un numpy.ndarray")
        if not isinstance(etiquetas_entrenamiento, np.ndarray):
            raise TypeError("etiquetas_entrenamiento debe ser un numpy.ndarray")
        if datos_entrenamiento.ndim != 2:
            raise ValueError("datos_entrenamiento debe ser bidimensional")
        if etiquetas_entrenamiento.ndim != 1:
            raise ValueError("etiquetas_entrenamiento debe ser unidimensional")
        if datos_entrenamiento.size == 0:
            raise ValueError("datos_entrenamiento no debe estar vacio")
        if etiquetas_entrenamiento.size == 0:
            raise ValueError("etiquetas_entrenamiento no debe estar vacio")
        if datos_entrenamiento.dtype != np.float32:
            raise TypeError("datos_entrenamiento debe tener dtype float32")
        if not np.issubdtype(etiquetas_entrenamiento.dtype, np.integer) or np.issubdtype(
            etiquetas_entrenamiento.dtype, np.bool_
        ):
            raise TypeError("etiquetas_entrenamiento debe tener dtype entero")
        if not np.isfinite(datos_entrenamiento).all():
            raise ValueError(
                "datos_entrenamiento no debe contener NaN ni valores infinitos"
            )
        if datos_entrenamiento.shape[0] != etiquetas_entrenamiento.shape[0]:
            raise ValueError(
                "etiquetas_entrenamiento debe coincidir con las muestras de datos_entrenamiento"
            )

        self._validar_dispositivo_ajuste()
        verificar_backend_cpu()

        datos_entrenamiento_copiados = datos_entrenamiento.copy()
        etiquetas_entrenamiento_copiadas = etiquetas_entrenamiento.copy()
        datos_entrenamiento_tensor = torch.from_numpy(datos_entrenamiento_copiados)
        etiquetas_entrenamiento_tensor = torch.from_numpy(
            etiquetas_entrenamiento_copiadas
        )

        self.datos_entrenamiento_ = datos_entrenamiento_copiados
        self.etiquetas_entrenamiento_ = etiquetas_entrenamiento_copiadas
        self.datos_entrenamiento_tensor_ = datos_entrenamiento_tensor
        self.etiquetas_entrenamiento_tensor_ = etiquetas_entrenamiento_tensor
        self.numero_muestras_entrenamiento_ = datos_entrenamiento_copiados.shape[0]
        self.numero_caracteristicas_ = datos_entrenamiento_copiados.shape[1]
        self.dispositivo_efectivo_ = torch.device("cpu")
        self.ajustado_ = True
        return self

    def predecir(self, datos_consulta: np.ndarray) -> np.ndarray:
        if not self.ajustado_:
            raise RuntimeError("el clasificador debe ajustarse antes de predecir")
        self._validar_consulta_compatible(datos_consulta)
        if self.numero_vecinos > self.numero_muestras_entrenamiento_:
            raise ValueError("k no debe ser mayor que el numero de columnas de distancias")

        datos_consulta_tensor = torch.from_numpy(datos_consulta)
        predicciones = torch.ops.knn_cuda.predecir_knn(
            self.datos_entrenamiento_tensor_,
            self.etiquetas_entrenamiento_tensor_,
            datos_consulta_tensor,
            self.numero_vecinos,
        )
        return predicciones.numpy()

    def vecinos_mas_cercanos(
        self,
        datos_consulta: np.ndarray,
        numero_vecinos: int | None = None,
        devolver_distancias: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if not self.ajustado_:
            raise RuntimeError(
                "el clasificador debe ajustarse antes de buscar vecinos"
            )

        vecinos = self.numero_vecinos if numero_vecinos is None else numero_vecinos
        self._validar_numero_vecinos(vecinos)
        if vecinos > self.numero_muestras_entrenamiento_:
            raise ValueError(
                "numero_vecinos no debe ser mayor que el numero de muestras de entrenamiento"
            )

        self._validar_consulta_compatible(datos_consulta)
        datos_consulta_tensor = torch.from_numpy(datos_consulta)
        distancias = torch.ops.knn_cuda.distancias_l2_cuadradas(
            datos_consulta_tensor, self.datos_entrenamiento_tensor_
        )
        distancias_seleccionadas, indices_seleccionados = (
            torch.ops.knn_cuda.seleccionar_top_k(distancias, vecinos)
        )
        if not devolver_distancias:
            return indices_seleccionados.numpy()

        distancias_euclidianas = torch.sqrt(distancias_seleccionadas)
        return distancias_euclidianas.numpy(), indices_seleccionados.numpy()
