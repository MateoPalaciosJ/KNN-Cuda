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
