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
