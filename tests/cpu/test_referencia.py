import numpy as np
import pytest

from knn_cuda.referencia import (
    distancias_l2_cuadradas,
    predecir_knn,
    seleccionar_top_k,
    votacion_uniforme,
)


def test_distancias_l2_cuadradas_devuelve_forma_y_tipo_esperados() -> None:
    datos_consulta = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    datos_entrenamiento = np.array(
        [[1.0, 0.0], [4.0, 5.0], [2.0, 2.0]], dtype=np.float32
    )

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    assert distancias.shape == (2, 3)
    assert distancias.dtype == np.float32


def test_distancias_l2_cuadradas_calcula_distancias_unidimensionales() -> None:
    datos_consulta = np.array([[0.0], [2.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[1.0], [4.0]], dtype=np.float32)
    esperado = np.array([[1.0, 16.0], [1.0, 4.0]], dtype=np.float32)

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    np.testing.assert_array_equal(distancias, esperado)


def test_distancias_l2_cuadradas_calcula_distancias_bidimensionales() -> None:
    datos_consulta = np.array([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[3.0, 4.0], [-1.0, 2.0]], dtype=np.float32)
    esperado = np.array([[25.0, 5.0], [8.0, 4.0]], dtype=np.float32)

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    np.testing.assert_array_equal(distancias, esperado)


def test_distancias_l2_cuadradas_calcula_varias_consultas_y_muestras() -> None:
    datos_consulta = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32
    )
    datos_entrenamiento = np.array(
        [[0.0, 1.0], [2.0, 2.0], [3.0, 0.0]], dtype=np.float32
    )
    esperado = np.array(
        [[1.0, 8.0, 9.0], [1.0, 2.0, 5.0], [5.0, 4.0, 1.0]],
        dtype=np.float32,
    )

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    np.testing.assert_array_equal(distancias, esperado)


def test_distancias_l2_cuadradas_devuelve_cero_para_muestras_iguales() -> None:
    datos_consulta = np.array([[1.5, -2.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[1.5, -2.0], [0.0, 0.0]], dtype=np.float32)

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    assert distancias[0, 0] == np.float32(0.0)


def test_distancias_l2_cuadradas_devuelve_distancias_iguales_para_duplicados() -> None:
    datos_consulta = np.array([[2.0, -1.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[0.0, 3.0], [0.0, 3.0]], dtype=np.float32)

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    assert distancias[0, 0] == distancias[0, 1]


def test_distancias_l2_cuadradas_maneja_valores_negativos() -> None:
    datos_consulta = np.array([[-2.0, 3.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[1.0, -1.0]], dtype=np.float32)
    esperado = np.array([[25.0]], dtype=np.float32)

    distancias = distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    np.testing.assert_array_equal(distancias, esperado)


def test_distancias_l2_cuadradas_no_modifica_las_entradas() -> None:
    datos_consulta = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
    datos_consulta_antes = datos_consulta.copy()
    datos_entrenamiento_antes = datos_entrenamiento.copy()

    distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)

    np.testing.assert_array_equal(datos_consulta, datos_consulta_antes)
    np.testing.assert_array_equal(datos_entrenamiento, datos_entrenamiento_antes)


def test_distancias_l2_cuadradas_es_determinista() -> None:
    datos_consulta = np.array([[0.0, 2.0], [3.0, -1.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[1.0, 1.0], [4.0, 0.0]], dtype=np.float32)

    primeras_distancias = distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )
    segundas_distancias = distancias_l2_cuadradas(
        datos_consulta, datos_entrenamiento
    )

    np.testing.assert_array_equal(primeras_distancias, segundas_distancias)


def test_distancias_l2_cuadradas_rechaza_consulta_que_no_es_arreglo() -> None:
    datos_entrenamiento = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="datos_consulta"):
        distancias_l2_cuadradas([[0.0]], datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_entrenamiento_que_no_es_arreglo() -> None:
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, [[1.0]])


def test_distancias_l2_cuadradas_rechaza_consulta_unidimensional() -> None:
    datos_consulta = np.array([0.0, 1.0], dtype=np.float32)
    datos_entrenamiento = np.array([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_consulta"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_entrenamiento_unidimensional() -> None:
    datos_consulta = np.array([[0.0, 1.0]], dtype=np.float32)
    datos_entrenamiento = np.array([0.0, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_consulta_tridimensional() -> None:
    datos_consulta = np.zeros((1, 1, 1), dtype=np.float32)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_consulta"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_entrenamiento_tridimensional() -> None:
    datos_consulta = np.array([[0.0]], dtype=np.float32)
    datos_entrenamiento = np.zeros((1, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_consulta_vacia() -> None:
    datos_consulta = np.empty((0, 2), dtype=np.float32)
    datos_entrenamiento = np.array([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_consulta"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_entrenamiento_vacio() -> None:
    datos_consulta = np.array([[0.0, 1.0]], dtype=np.float32)
    datos_entrenamiento = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_consulta_float64() -> None:
    datos_consulta = np.array([[0.0]], dtype=np.float64)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="datos_consulta"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_entrenamiento_float64() -> None:
    datos_consulta = np.array([[0.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float64)

    with pytest.raises(TypeError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_consulta_entera() -> None:
    datos_consulta = np.array([[0]], dtype=np.int32)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="datos_consulta"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_entrenamiento_entero() -> None:
    datos_consulta = np.array([[0.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[0]], dtype=np.int32)

    with pytest.raises(TypeError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_cantidades_distintas_de_caracteristicas() -> None:
    datos_consulta = np.array([[0.0, 1.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="mismo numero de caracteristicas"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_nan_en_consulta() -> None:
    datos_consulta = np.array([[np.nan]], dtype=np.float32)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_consulta"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_nan_en_entrenamiento() -> None:
    datos_consulta = np.array([[0.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_infinito_en_consulta() -> None:
    datos_consulta = np.array([[np.inf]], dtype=np.float32)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_consulta"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_distancias_l2_cuadradas_rechaza_infinito_en_entrenamiento() -> None:
    datos_consulta = np.array([[0.0]], dtype=np.float32)
    datos_entrenamiento = np.array([[np.inf]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        distancias_l2_cuadradas(datos_consulta, datos_entrenamiento)


def test_seleccionar_top_k_devuelve_la_forma_esperada() -> None:
    distancias = np.array([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 2)

    assert distancias_seleccionadas.shape == (2, 2)
    assert indices_seleccionados.shape == (2, 2)


def test_seleccionar_top_k_conserva_el_tipo_float32_de_las_distancias() -> None:
    distancias = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)

    distancias_seleccionadas, _ = seleccionar_top_k(distancias, 2)

    assert distancias_seleccionadas.dtype == np.float32


def test_seleccionar_top_k_devuelve_indices_int64() -> None:
    distancias = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)

    _, indices_seleccionados = seleccionar_top_k(distancias, 2)

    assert indices_seleccionados.dtype == np.int64


def test_seleccionar_top_k_funciona_con_k_igual_a_uno() -> None:
    distancias = np.array([[3.0, 1.0, 2.0], [4.0, 0.0, 5.0]], dtype=np.float32)
    distancias_esperadas = np.array([[1.0], [0.0]], dtype=np.float32)
    indices_esperados = np.array([[1], [1]], dtype=np.int64)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 1)

    np.testing.assert_array_equal(distancias_seleccionadas, distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_funciona_con_k_igual_a_n() -> None:
    distancias = np.array([[2.0, 1.0, 3.0]], dtype=np.float32)
    distancias_esperadas = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    indices_esperados = np.array([[1, 0, 2]], dtype=np.int64)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 3)

    np.testing.assert_array_equal(distancias_seleccionadas, distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_funciona_con_una_consulta() -> None:
    distancias = np.array([[5.0, 2.0, 4.0]], dtype=np.float32)
    distancias_esperadas = np.array([[2.0, 4.0]], dtype=np.float32)
    indices_esperados = np.array([[1, 2]], dtype=np.int64)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 2)

    np.testing.assert_array_equal(distancias_seleccionadas, distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_funciona_con_varias_consultas() -> None:
    distancias = np.array(
        [[4.0, 1.0, 3.0], [2.0, 5.0, 0.0]], dtype=np.float32
    )
    distancias_esperadas = np.array([[1.0, 3.0], [0.0, 2.0]], dtype=np.float32)
    indices_esperados = np.array([[1, 2], [2, 0]], dtype=np.int64)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 2)

    np.testing.assert_array_equal(distancias_seleccionadas, distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_funciona_con_una_muestra_de_entrenamiento() -> None:
    distancias = np.array([[4.0], [1.0]], dtype=np.float32)
    distancias_esperadas = np.array([[4.0], [1.0]], dtype=np.float32)
    indices_esperados = np.array([[0], [0]], dtype=np.int64)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 1)

    np.testing.assert_array_equal(distancias_seleccionadas, distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_ordena_distancias_de_forma_ascendente() -> None:
    distancias = np.array([[9.0, 1.0, 5.0, 3.0]], dtype=np.float32)
    distancias_esperadas = np.array([[1.0, 3.0, 5.0, 9.0]], dtype=np.float32)

    distancias_seleccionadas, _ = seleccionar_top_k(distancias, 4)

    np.testing.assert_array_equal(distancias_seleccionadas, distancias_esperadas)


def test_seleccionar_top_k_devuelve_los_indices_correctos() -> None:
    distancias = np.array([[8.0, 2.0, 5.0, 1.0]], dtype=np.float32)
    indices_esperados = np.array([[3, 1, 2]], dtype=np.int64)

    _, indices_seleccionados = seleccionar_top_k(distancias, 3)

    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_mantiene_la_correspondencia_entre_distancia_e_indice() -> None:
    distancias = np.array([[6.0, 2.0, 4.0]], dtype=np.float32)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 3)

    assert distancias_seleccionadas[0, 0] == distancias[0, indices_seleccionados[0, 0]]
    assert distancias_seleccionadas[0, 1] == distancias[0, indices_seleccionados[0, 1]]
    assert distancias_seleccionadas[0, 2] == distancias[0, indices_seleccionados[0, 2]]


def test_seleccionar_top_k_resuelve_empates_por_menor_indice() -> None:
    distancias = np.array([[2.0, 1.0, 1.0, 2.0]], dtype=np.float32)
    distancias_esperadas = np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
    indices_esperados = np.array([[1, 2, 0, 3]], dtype=np.int64)

    distancias_seleccionadas, indices_seleccionados = seleccionar_top_k(distancias, 4)

    np.testing.assert_array_equal(distancias_seleccionadas, distancias_esperadas)
    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_mantiene_el_orden_en_distancias_repetidas() -> None:
    distancias = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    indices_esperados = np.array([[0, 1]], dtype=np.int64)

    _, indices_seleccionados = seleccionar_top_k(distancias, 2)

    np.testing.assert_array_equal(indices_seleccionados, indices_esperados)


def test_seleccionar_top_k_es_determinista() -> None:
    distancias = np.array([[2.0, 1.0, 1.0], [3.0, 0.0, 2.0]], dtype=np.float32)

    primeras_salidas = seleccionar_top_k(distancias, 2)
    segundas_salidas = seleccionar_top_k(distancias, 2)

    np.testing.assert_array_equal(primeras_salidas[0], segundas_salidas[0])
    np.testing.assert_array_equal(primeras_salidas[1], segundas_salidas[1])


def test_seleccionar_top_k_no_modifica_las_distancias() -> None:
    distancias = np.array([[3.0, 1.0, 2.0]], dtype=np.float32)
    distancias_antes = distancias.copy()

    seleccionar_top_k(distancias, 2)

    np.testing.assert_array_equal(distancias, distancias_antes)


def test_seleccionar_top_k_rechaza_distancias_que_no_son_arreglo() -> None:
    with pytest.raises(TypeError, match="distancias"):
        seleccionar_top_k([[1.0]], 1)


def test_seleccionar_top_k_rechaza_distancias_unidimensionales() -> None:
    distancias = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="distancias"):
        seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_tridimensionales() -> None:
    distancias = np.zeros((1, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="distancias"):
        seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_vacias() -> None:
    distancias = np.empty((0, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="distancias"):
        seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_float64() -> None:
    distancias = np.array([[1.0]], dtype=np.float64)

    with pytest.raises(TypeError, match="distancias"):
        seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_distancias_enteras() -> None:
    distancias = np.array([[1]], dtype=np.int32)

    with pytest.raises(TypeError, match="distancias"):
        seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_nan() -> None:
    distancias = np.array([[np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="distancias"):
        seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_infinito() -> None:
    distancias = np.array([[np.inf]], dtype=np.float32)

    with pytest.raises(ValueError, match="distancias"):
        seleccionar_top_k(distancias, 1)


def test_seleccionar_top_k_rechaza_k_no_entero() -> None:
    distancias = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="k"):
        seleccionar_top_k(distancias, 1.5)


def test_seleccionar_top_k_rechaza_k_booleano() -> None:
    distancias = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="k"):
        seleccionar_top_k(distancias, True)


def test_seleccionar_top_k_rechaza_k_igual_a_cero() -> None:
    distancias = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="k"):
        seleccionar_top_k(distancias, 0)


def test_seleccionar_top_k_rechaza_k_negativo() -> None:
    distancias = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="k"):
        seleccionar_top_k(distancias, -1)


def test_seleccionar_top_k_rechaza_k_mayor_que_el_numero_de_columnas() -> None:
    distancias = np.array([[1.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="k"):
        seleccionar_top_k(distancias, 3)


def test_votacion_uniforme_devuelve_la_forma_esperada() -> None:
    etiquetas_vecinos = np.array([[1, 2], [3, 3]], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    assert predicciones.shape == (2,)


def test_votacion_uniforme_conserva_el_dtype_de_entrada() -> None:
    etiquetas_vecinos = np.array([[1, 2]], dtype=np.int32)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    assert predicciones.dtype == np.int32


def test_votacion_uniforme_clasifica_etiquetas_binarias() -> None:
    etiquetas_vecinos = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64)
    esperado = np.array([1, 0], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_clasifica_etiquetas_multiclase() -> None:
    etiquetas_vecinos = np.array([[2, 5, 5, 9], [2, 9, 9, 9]], dtype=np.int64)
    esperado = np.array([5, 9], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_funciona_con_una_sola_clase() -> None:
    etiquetas_vecinos = np.array([[7, 7], [7, 7]], dtype=np.int64)
    esperado = np.array([7, 7], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_funciona_con_k_igual_a_uno() -> None:
    etiquetas_vecinos = np.array([[4], [-2]], dtype=np.int64)
    esperado = np.array([4, -2], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_funciona_con_varias_consultas() -> None:
    etiquetas_vecinos = np.array([[1, 1, 2], [3, 4, 4], [8, 8, 8]], dtype=np.int64)
    esperado = np.array([1, 4, 8], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_maneja_etiquetas_negativas() -> None:
    etiquetas_vecinos = np.array([[-3, 10, -3]], dtype=np.int64)
    esperado = np.array([-3], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_maneja_etiquetas_no_consecutivas() -> None:
    etiquetas_vecinos = np.array([[100, 200, 100]], dtype=np.int64)
    esperado = np.array([100], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_maneja_etiquetas_grandes() -> None:
    etiquetas_vecinos = np.array([[1_000_000, 2_000_000, 2_000_000]], dtype=np.int64)
    esperado = np.array([2_000_000], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_resuelve_empate_entre_dos_clases() -> None:
    etiquetas_vecinos = np.array([[5, 2]], dtype=np.int64)
    esperado = np.array([2], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_resuelve_empate_entre_mas_de_dos_clases() -> None:
    etiquetas_vecinos = np.array([[9, 5, 2]], dtype=np.int64)
    esperado = np.array([2], dtype=np.int64)

    predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(predicciones, esperado)


def test_votacion_uniforme_es_determinista() -> None:
    etiquetas_vecinos = np.array([[5, 2, 2], [3, 3, 1]], dtype=np.int64)

    primeras_predicciones = votacion_uniforme(etiquetas_vecinos)
    segundas_predicciones = votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(primeras_predicciones, segundas_predicciones)


def test_votacion_uniforme_no_modifica_las_etiquetas() -> None:
    etiquetas_vecinos = np.array([[1, 2, 2]], dtype=np.int64)
    etiquetas_vecinos_antes = etiquetas_vecinos.copy()

    votacion_uniforme(etiquetas_vecinos)

    np.testing.assert_array_equal(etiquetas_vecinos, etiquetas_vecinos_antes)


def test_votacion_uniforme_rechaza_una_entrada_que_no_es_arreglo() -> None:
    with pytest.raises(TypeError, match="etiquetas_vecinos"):
        votacion_uniforme([[1, 2]])


def test_votacion_uniforme_rechaza_una_entrada_unidimensional() -> None:
    etiquetas_vecinos = np.array([1, 2], dtype=np.int64)

    with pytest.raises(ValueError, match="etiquetas_vecinos"):
        votacion_uniforme(etiquetas_vecinos)


def test_votacion_uniforme_rechaza_una_entrada_tridimensional() -> None:
    etiquetas_vecinos = np.zeros((1, 1, 1), dtype=np.int64)

    with pytest.raises(ValueError, match="etiquetas_vecinos"):
        votacion_uniforme(etiquetas_vecinos)


def test_votacion_uniforme_rechaza_una_entrada_vacia() -> None:
    etiquetas_vecinos = np.empty((0, 1), dtype=np.int64)

    with pytest.raises(ValueError, match="etiquetas_vecinos"):
        votacion_uniforme(etiquetas_vecinos)


def test_votacion_uniforme_rechaza_etiquetas_float32() -> None:
    etiquetas_vecinos = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="etiquetas_vecinos"):
        votacion_uniforme(etiquetas_vecinos)


def test_votacion_uniforme_rechaza_etiquetas_float64() -> None:
    etiquetas_vecinos = np.array([[1.0]], dtype=np.float64)

    with pytest.raises(TypeError, match="etiquetas_vecinos"):
        votacion_uniforme(etiquetas_vecinos)


def test_votacion_uniforme_rechaza_etiquetas_booleanas() -> None:
    etiquetas_vecinos = np.array([[True]], dtype=np.bool_)

    with pytest.raises(TypeError, match="etiquetas_vecinos"):
        votacion_uniforme(etiquetas_vecinos)


def test_votacion_uniforme_rechaza_etiquetas_object() -> None:
    etiquetas_vecinos = np.array([["a"]], dtype=object)

    with pytest.raises(TypeError, match="etiquetas_vecinos"):
        votacion_uniforme(etiquetas_vecinos)


def test_predecir_knn_funciona_con_k_igual_a_uno() -> None:
    datos_entrenamiento = np.array([[0.0], [3.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2], dtype=np.int64)
    datos_consulta = np.array([[0.2], [2.7]], dtype=np.float32)
    esperado = np.array([1, 2], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_funciona_con_k_igual_a_n() -> None:
    datos_entrenamiento = np.array([[0.0], [1.0], [10.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2, 2], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)
    esperado = np.array([2], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 3
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_clasifica_datos_binarios() -> None:
    datos_entrenamiento = np.array([[0.0], [4.0], [5.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([0, 1, 1], dtype=np.int64)
    datos_consulta = np.array([[0.5], [4.5]], dtype=np.float32)
    esperado = np.array([0, 1], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_clasifica_datos_multiclase() -> None:
    datos_entrenamiento = np.array([[0.0], [3.0], [6.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([10, 20, 30], dtype=np.int64)
    datos_consulta = np.array([[2.8]], dtype=np.float32)
    esperado = np.array([20], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_funciona_con_una_sola_clase() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([7, 7], dtype=np.int64)
    datos_consulta = np.array([[1.0]], dtype=np.float32)
    esperado = np.array([7], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 2
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_funciona_con_una_consulta() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([4, 9], dtype=np.int64)
    datos_consulta = np.array([[1.8]], dtype=np.float32)
    esperado = np.array([9], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_funciona_con_varias_consultas() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2, 3], dtype=np.int64)
    datos_consulta = np.array([[0.1], [1.9], [3.8]], dtype=np.float32)
    esperado = np.array([1, 2, 3], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_maneja_etiquetas_no_consecutivas() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([100, 200], dtype=np.int64)
    datos_consulta = np.array([[0.1]], dtype=np.float32)
    esperado = np.array([100], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_maneja_etiquetas_negativas() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([-5, 10], dtype=np.int64)
    datos_consulta = np.array([[0.2]], dtype=np.float32)
    esperado = np.array([-5], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_resuelve_empate_de_distancia_por_menor_indice() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([9, 1], dtype=np.int64)
    datos_consulta = np.array([[1.0]], dtype=np.float32)
    esperado = np.array([9], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_resuelve_empate_de_votos_por_menor_etiqueta() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2], dtype=np.int64)
    datos_consulta = np.array([[1.0]], dtype=np.float32)
    esperado = np.array([2], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 2
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_funciona_con_datos_duplicados() -> None:
    datos_entrenamiento = np.array([[0.0], [0.0], [3.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([4, 4, 9], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)
    esperado = np.array([4], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 2
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_resuelve_un_caso_manual_con_varias_caracteristicas() -> None:
    datos_entrenamiento = np.array(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]], dtype=np.float32
    )
    etiquetas_entrenamiento = np.array([1, 2, 2], dtype=np.int64)
    datos_consulta = np.array([[0.1, 0.0], [1.0, 3.8]], dtype=np.float32)
    esperado = np.array([1, 2], dtype=np.int64)

    predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
    )

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_knn_es_determinista() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2, 5], dtype=np.int64)
    datos_consulta = np.array([[1.0], [3.0]], dtype=np.float32)

    primeras_predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 2
    )
    segundas_predicciones = predecir_knn(
        datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 2
    )

    np.testing.assert_array_equal(primeras_predicciones, segundas_predicciones)


def test_predecir_knn_no_modifica_las_entradas() -> None:
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2], dtype=np.int64)
    datos_consulta = np.array([[0.5]], dtype=np.float32)
    datos_entrenamiento_antes = datos_entrenamiento.copy()
    etiquetas_entrenamiento_antes = etiquetas_entrenamiento.copy()
    datos_consulta_antes = datos_consulta.copy()

    predecir_knn(datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1)

    np.testing.assert_array_equal(datos_entrenamiento, datos_entrenamiento_antes)
    np.testing.assert_array_equal(etiquetas_entrenamiento, etiquetas_entrenamiento_antes)
    np.testing.assert_array_equal(datos_consulta, datos_consulta_antes)


def test_predecir_knn_rechaza_etiquetas_que_no_son_arreglo() -> None:
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="etiquetas_entrenamiento"):
        predecir_knn(datos_entrenamiento, [1], datos_consulta, 1)


def test_predecir_knn_rechaza_etiquetas_bidimensionales() -> None:
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([[1]], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="etiquetas_entrenamiento"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )


def test_predecir_knn_rechaza_etiquetas_vacias() -> None:
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="etiquetas_entrenamiento"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )


def test_predecir_knn_rechaza_etiquetas_float() -> None:
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1.0], dtype=np.float32)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="etiquetas_entrenamiento"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )


def test_predecir_knn_rechaza_etiquetas_booleanas() -> None:
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([True], dtype=np.bool_)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(TypeError, match="etiquetas_entrenamiento"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )


def test_predecir_knn_rechaza_cantidad_incorrecta_de_etiquetas() -> None:
    datos_entrenamiento = np.array([[0.0], [1.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="etiquetas_entrenamiento"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )


def test_predecir_knn_propaga_k_invalido() -> None:
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="k"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 0
        )


def test_predecir_knn_propaga_dimensiones_incompatibles() -> None:
    datos_entrenamiento = np.array([[0.0, 1.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_consulta"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )


def test_predecir_knn_propaga_nan_en_consulta() -> None:
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[np.nan]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_consulta"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )


def test_predecir_knn_propaga_infinito_en_entrenamiento() -> None:
    datos_entrenamiento = np.array([[np.inf]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        predecir_knn(
            datos_entrenamiento, etiquetas_entrenamiento, datos_consulta, 1
        )
