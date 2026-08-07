import numpy as np


def distancias_l2_cuadradas(
    datos_consulta: np.ndarray, datos_entrenamiento: np.ndarray
) -> np.ndarray:
    if not isinstance(datos_consulta, np.ndarray):
        raise TypeError("datos_consulta debe ser un numpy.ndarray")
    if not isinstance(datos_entrenamiento, np.ndarray):
        raise TypeError("datos_entrenamiento debe ser un numpy.ndarray")

    if datos_consulta.ndim != 2:
        raise ValueError("datos_consulta debe ser bidimensional")
    if datos_entrenamiento.ndim != 2:
        raise ValueError("datos_entrenamiento debe ser bidimensional")
    if datos_consulta.size == 0:
        raise ValueError("datos_consulta no debe estar vacio")
    if datos_entrenamiento.size == 0:
        raise ValueError("datos_entrenamiento no debe estar vacio")

    if datos_consulta.dtype != np.float32:
        raise TypeError("datos_consulta debe tener dtype float32")
    if datos_entrenamiento.dtype != np.float32:
        raise TypeError("datos_entrenamiento debe tener dtype float32")

    if datos_consulta.shape[1] != datos_entrenamiento.shape[1]:
        raise ValueError(
            "datos_consulta y datos_entrenamiento deben tener el mismo numero de caracteristicas"
        )
    if not np.isfinite(datos_consulta).all():
        raise ValueError("datos_consulta no debe contener NaN ni valores infinitos")
    if not np.isfinite(datos_entrenamiento).all():
        raise ValueError("datos_entrenamiento no debe contener NaN ni valores infinitos")

    diferencias = (
        datos_consulta[:, np.newaxis, :] - datos_entrenamiento[np.newaxis, :, :]
    )
    distancias_cuadradas = np.sum(
        diferencias * diferencias, axis=2, dtype=np.float32
    )
    return distancias_cuadradas


def seleccionar_top_k(
    distancias: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(distancias, np.ndarray):
        raise TypeError("distancias debe ser un numpy.ndarray")
    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise TypeError("k debe ser un entero")

    if distancias.ndim != 2:
        raise ValueError("distancias debe ser bidimensional")
    if distancias.size == 0:
        raise ValueError("distancias no debe estar vacia")
    if distancias.dtype != np.float32:
        raise TypeError("distancias debe tener dtype float32")
    if not np.isfinite(distancias).all():
        raise ValueError("distancias no debe contener NaN ni valores infinitos")
    if k < 1:
        raise ValueError("k debe ser mayor o igual a 1")
    if k > distancias.shape[1]:
        raise ValueError("k no debe ser mayor que el numero de columnas de distancias")

    indices_ordenados = np.argsort(distancias, axis=1, kind="stable")
    indices_seleccionados = indices_ordenados[:, :k].astype(np.int64, copy=False)
    distancias_seleccionadas = np.take_along_axis(
        distancias, indices_seleccionados, axis=1
    )
    return distancias_seleccionadas, indices_seleccionados


def votacion_uniforme(etiquetas_vecinos: np.ndarray) -> np.ndarray:
    if not isinstance(etiquetas_vecinos, np.ndarray):
        raise TypeError("etiquetas_vecinos debe ser un numpy.ndarray")
    if etiquetas_vecinos.ndim != 2:
        raise ValueError("etiquetas_vecinos debe ser bidimensional")
    if etiquetas_vecinos.size == 0:
        raise ValueError("etiquetas_vecinos no debe estar vacio")
    if not np.issubdtype(etiquetas_vecinos.dtype, np.integer) or np.issubdtype(
        etiquetas_vecinos.dtype, np.bool_
    ):
        raise TypeError("etiquetas_vecinos debe tener dtype entero")

    predicciones = np.empty(
        etiquetas_vecinos.shape[0], dtype=etiquetas_vecinos.dtype
    )
    for indice_consulta, etiquetas_consulta in enumerate(etiquetas_vecinos):
        etiquetas_distintas, conteos = np.unique(
            etiquetas_consulta, return_counts=True
        )
        conteo_maximo = conteos.max()
        etiquetas_ganadoras = etiquetas_distintas[conteos == conteo_maximo]
        predicciones[indice_consulta] = etiquetas_ganadoras.min()
    return predicciones


def predecir_knn(
    datos_entrenamiento: np.ndarray,
    etiquetas_entrenamiento: np.ndarray,
    datos_consulta: np.ndarray,
    k: int,
) -> np.ndarray:
    if not isinstance(etiquetas_entrenamiento, np.ndarray):
        raise TypeError("etiquetas_entrenamiento debe ser un numpy.ndarray")
    if etiquetas_entrenamiento.ndim != 1:
        raise ValueError("etiquetas_entrenamiento debe ser unidimensional")
    if etiquetas_entrenamiento.size == 0:
        raise ValueError("etiquetas_entrenamiento no debe estar vacio")
    if not np.issubdtype(etiquetas_entrenamiento.dtype, np.integer) or np.issubdtype(
        etiquetas_entrenamiento.dtype, np.bool_
    ):
        raise TypeError("etiquetas_entrenamiento debe tener dtype entero")

    if etiquetas_entrenamiento.shape[0] != datos_entrenamiento.shape[0]:
        raise ValueError(
            "etiquetas_entrenamiento debe coincidir con las muestras de datos_entrenamiento"
        )

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)
    _, indices_seleccionados = seleccionar_top_k(distancias, k)
    etiquetas_vecinos = etiquetas_entrenamiento[indices_seleccionados]
    return votacion_uniforme(etiquetas_vecinos)
