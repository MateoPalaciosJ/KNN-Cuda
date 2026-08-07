import numpy as np

from .referencia import (
    distancias_l2_cuadradas,
    predecir_knn,
    seleccionar_top_k,
)


class ClasificadorKNNCUDA:
    def __init__(self, numero_vecinos: int = 5) -> None:
        self._validar_numero_vecinos(numero_vecinos)
        self.numero_vecinos = numero_vecinos
        self.ajustado_ = False

    @staticmethod
    def _validar_numero_vecinos(numero_vecinos: int) -> None:
        if not isinstance(numero_vecinos, (int, np.integer)) or isinstance(
            numero_vecinos, (bool, np.bool_)
        ):
            raise TypeError("numero_vecinos debe ser un entero")
        if numero_vecinos < 1:
            raise ValueError("numero_vecinos debe ser mayor o igual a 1")

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

        self.datos_entrenamiento_ = datos_entrenamiento.copy()
        self.etiquetas_entrenamiento_ = etiquetas_entrenamiento.copy()
        self.numero_muestras_entrenamiento_ = datos_entrenamiento.shape[0]
        self.numero_caracteristicas_ = datos_entrenamiento.shape[1]
        self.ajustado_ = True
        return self

    def predecir(self, datos_consulta: np.ndarray) -> np.ndarray:
        if not self.ajustado_:
            raise RuntimeError("el clasificador debe ajustarse antes de predecir")
        return predecir_knn(
            self.datos_entrenamiento_,
            self.etiquetas_entrenamiento_,
            datos_consulta,
            self.numero_vecinos,
        )

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

        distancias = distancias_l2_cuadradas(
            datos_consulta, self.datos_entrenamiento_
        )
        distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(
            distancias, vecinos
        )
        if not devolver_distancias:
            return indices_seleccionados

        distancias_euclidianas = np.sqrt(distancias_seleccionadas)
        return distancias_euclidianas, indices_seleccionados
