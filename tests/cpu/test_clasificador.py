import numpy as np
import pytest
import torch

from knn_cuda import ClasificadorKNNCUDA
from knn_cuda import referencia


def test_clasificador_knn_cuda_usa_numero_vecinos_por_defecto() -> None:
    clasificador = ClasificadorKNNCUDA()

    assert clasificador.numero_vecinos == 5


def test_clasificador_knn_cuda_acepta_numero_vecinos_personalizado() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=3)

    assert clasificador.numero_vecinos == 3


def test_clasificador_knn_cuda_rechaza_numero_vecinos_no_entero() -> None:
    with pytest.raises(TypeError, match="numero_vecinos"):
        ClasificadorKNNCUDA(numero_vecinos=1.5)


def test_clasificador_knn_cuda_rechaza_numero_vecinos_booleano() -> None:
    with pytest.raises(TypeError, match="numero_vecinos"):
        ClasificadorKNNCUDA(numero_vecinos=True)


def test_clasificador_knn_cuda_rechaza_numero_vecinos_cero() -> None:
    with pytest.raises(ValueError, match="numero_vecinos"):
        ClasificadorKNNCUDA(numero_vecinos=0)


def test_clasificador_knn_cuda_rechaza_numero_vecinos_negativo() -> None:
    with pytest.raises(ValueError, match="numero_vecinos"):
        ClasificadorKNNCUDA(numero_vecinos=-1)


def test_clasificador_knn_cuda_inicia_sin_ajustar() -> None:
    clasificador = ClasificadorKNNCUDA()

    assert clasificador.ajustado_ is False
    assert not hasattr(clasificador, "datos_entrenamiento_")
    assert not hasattr(clasificador, "etiquetas_entrenamiento_")


def test_ajustar_devuelve_la_misma_instancia() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    resultado = clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert resultado is clasificador


def test_ajustar_marca_el_clasificador_como_ajustado() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert clasificador.ajustado_ is True


def test_ajustar_almacena_los_metadatos_de_entrenamiento() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array(
        [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32
    )
    etiquetas_entrenamiento = np.array([1, 2, 3], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert clasificador.numero_muestras_entrenamiento_ == 3
    assert clasificador.numero_caracteristicas_ == 2


def test_ajustar_conserva_datos_y_etiquetas_de_entrenamiento() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([4, 9], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    np.testing.assert_array_equal(
        clasificador.datos_entrenamiento_, datos_entrenamiento
    )
    np.testing.assert_array_equal(
        clasificador.etiquetas_entrenamiento_, etiquetas_entrenamiento
    )


def test_ajustar_no_modifica_las_entradas() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([4, 9], dtype=np.int64)
    datos_entrenamiento_antes = datos_entrenamiento.copy()
    etiquetas_entrenamiento_antes = etiquetas_entrenamiento.copy()

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    np.testing.assert_array_equal(datos_entrenamiento, datos_entrenamiento_antes)
    np.testing.assert_array_equal(
        etiquetas_entrenamiento, etiquetas_entrenamiento_antes
    )


def test_ajustar_rechaza_cantidad_incorrecta_de_etiquetas() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="etiquetas_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_datos_de_entrenamiento_vacios() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.empty((0, 1), dtype=np.float32)
    etiquetas_entrenamiento = np.array([], dtype=np.int64)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_datos_entrenamiento_float64() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float64)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(TypeError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_datos_entrenamiento_enteros() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0]], dtype=np.int32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(TypeError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_nan_en_datos_entrenamiento() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[np.nan]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_infinito_en_datos_entrenamiento() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[np.inf]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_datos_entrenamiento_unidimensionales() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([0.0], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_datos_entrenamiento_tridimensionales() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.zeros((1, 1, 1), dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_etiquetas_entrenamiento_float32() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1.0], dtype=np.float32)

    with pytest.raises(TypeError, match="etiquetas_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_etiquetas_entrenamiento_float64() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1.0], dtype=np.float64)

    with pytest.raises(TypeError, match="etiquetas_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_etiquetas_entrenamiento_booleanas() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([True], dtype=np.bool_)

    with pytest.raises(TypeError, match="etiquetas_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_rechaza_etiquetas_entrenamiento_object() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array(["a"], dtype=object)

    with pytest.raises(TypeError, match="etiquetas_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)


def test_ajustar_acepta_etiquetas_entrenamiento_int32() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert clasificador.etiquetas_entrenamiento_.dtype == np.int32


def test_ajustar_acepta_etiquetas_entrenamiento_int64() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert clasificador.etiquetas_entrenamiento_.dtype == np.int64


def test_ajustar_almacena_copias_independientes() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2], dtype=np.int64)
    datos_consulta = np.array([[0.1]], dtype=np.float32)
    datos_entrenamiento_esperados = datos_entrenamiento.copy()
    etiquetas_entrenamiento_esperadas = etiquetas_entrenamiento.copy()

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    datos_entrenamiento[0, 0] = 100.0
    etiquetas_entrenamiento[0] = 9

    np.testing.assert_array_equal(
        clasificador.datos_entrenamiento_, datos_entrenamiento_esperados
    )
    np.testing.assert_array_equal(
        clasificador.etiquetas_entrenamiento_, etiquetas_entrenamiento_esperadas
    )
    np.testing.assert_array_equal(
        clasificador.predecir(datos_consulta), np.array([1], dtype=np.int64)
    )


def test_ajustar_no_deja_estado_parcial_despues_de_un_error() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[np.nan]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(ValueError, match="datos_entrenamiento"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert clasificador.ajustado_ is False
    assert not hasattr(clasificador, "datos_entrenamiento_")
    assert not hasattr(clasificador, "etiquetas_entrenamiento_")


def test_predecir_antes_de_ajustar_genera_error() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="ajustarse"):
        clasificador.predecir(datos_consulta)


def test_vecinos_mas_cercanos_antes_de_ajustar_genera_error() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="ajustarse"):
        clasificador.vecinos_mas_cercanos(datos_consulta)


def test_predecir_funciona_despues_de_ajustar() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [3.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2], dtype=np.int64)
    datos_consulta = np.array([[0.2]], dtype=np.float32)
    esperado = np.array([1], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(datos_consulta)

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_clasifica_datos_binarios() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [4.0], [5.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([0, 1, 1], dtype=np.int64)
    datos_consulta = np.array([[0.5], [4.5]], dtype=np.float32)
    esperado = np.array([0, 1], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(datos_consulta)

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_clasifica_datos_multiclase() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [3.0], [6.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([10, 20, 30], dtype=np.int64)
    datos_consulta = np.array([[2.8]], dtype=np.float32)
    esperado = np.array([20], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(datos_consulta)

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_funciona_con_varias_consultas() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2, 3], dtype=np.int64)
    datos_consulta = np.array([[0.1], [1.9], [3.8]], dtype=np.float32)
    esperado = np.array([1, 2, 3], dtype=np.int64)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(datos_consulta)

    np.testing.assert_array_equal(predicciones, esperado)


def test_predecir_es_determinista() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=2)
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2, 5], dtype=np.int64)
    datos_consulta = np.array([[1.0], [3.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    primeras_predicciones = clasificador.predecir(datos_consulta)
    segundas_predicciones = clasificador.predecir(datos_consulta)

    np.testing.assert_array_equal(primeras_predicciones, segundas_predicciones)


def test_vecinos_mas_cercanos_devuelve_formas_y_tipos_correctos() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=2)
    datos_entrenamiento = np.array([[0.0], [2.0], [5.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2, 3], dtype=np.int64)
    datos_consulta = np.array([[1.0], [4.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    distancias, indices = clasificador.vecinos_mas_cercanos(datos_consulta)

    assert distancias.shape == (2, 2)
    assert indices.shape == (2, 2)
    assert distancias.dtype == np.float32
    assert indices.dtype == np.int64


def test_vecinos_mas_cercanos_devuelve_distancias_euclidianas() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[3.0, 4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[0.0, 0.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    distancias, indices = clasificador.vecinos_mas_cercanos(datos_consulta)

    np.testing.assert_array_equal(distancias, np.array([[5.0]], dtype=np.float32))
    np.testing.assert_array_equal(indices, np.array([[0]], dtype=np.int64))


def test_vecinos_mas_cercanos_puede_omitir_distancias() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2], dtype=np.int64)
    datos_consulta = np.array([[1.8]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    indices = clasificador.vecinos_mas_cercanos(
        datos_consulta, devolver_distancias=False
    )

    np.testing.assert_array_equal(indices, np.array([[1]], dtype=np.int64))


def test_vecinos_mas_cercanos_acepta_numero_vecinos_temporal() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2, 3], dtype=np.int64)
    datos_consulta = np.array([[1.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    _, indices = clasificador.vecinos_mas_cercanos(
        datos_consulta, numero_vecinos=2
    )

    np.testing.assert_array_equal(indices, np.array([[0, 1]], dtype=np.int64))


def test_numero_vecinos_temporal_no_modifica_el_estado() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2, 3], dtype=np.int64)
    datos_consulta = np.array([[1.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    clasificador.vecinos_mas_cercanos(datos_consulta, numero_vecinos=2)

    assert clasificador.numero_vecinos == 1


def test_vecinos_mas_cercanos_conserva_desempate_determinista() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([9, 1], dtype=np.int64)
    datos_consulta = np.array([[1.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    _, indices = clasificador.vecinos_mas_cercanos(datos_consulta)

    np.testing.assert_array_equal(indices, np.array([[0]], dtype=np.int64))


def test_vecinos_mas_cercanos_rechaza_numero_vecinos_mayor_que_entrenamiento() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    with pytest.raises(ValueError, match="numero_vecinos"):
        clasificador.vecinos_mas_cercanos(datos_consulta, numero_vecinos=2)


def test_vecinos_mas_cercanos_propaga_caracteristicas_incompatibles() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0, 1.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[0.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    with pytest.raises(ValueError, match="datos_consulta"):
        clasificador.vecinos_mas_cercanos(datos_consulta)


def test_vecinos_mas_cercanos_propaga_nan() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[np.nan]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    with pytest.raises(ValueError, match="datos_consulta"):
        clasificador.vecinos_mas_cercanos(datos_consulta)


def test_vecinos_mas_cercanos_propaga_infinito() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)
    datos_consulta = np.array([[np.inf]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    with pytest.raises(ValueError, match="datos_consulta"):
        clasificador.vecinos_mas_cercanos(datos_consulta)


def test_vecinos_mas_cercanos_no_modifica_las_consultas() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1, 2], dtype=np.int64)
    datos_consulta = np.array([[0.5]], dtype=np.float32)
    datos_consulta_antes = datos_consulta.copy()

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    clasificador.vecinos_mas_cercanos(datos_consulta)

    np.testing.assert_array_equal(datos_consulta, datos_consulta_antes)


def test_clasificador_knn_cuda_usa_cpu_como_dispositivo_predeterminado() -> None:
    clasificador = ClasificadorKNNCUDA()

    assert clasificador.dispositivo == "cpu"
    assert not hasattr(clasificador, "dispositivo_efectivo_")


@pytest.mark.parametrize("dispositivo", ["cuda", "auto"])
def test_clasificador_reconoce_dispositivos_pendientes_y_falla_al_ajustar(
    dispositivo: str,
) -> None:
    clasificador = ClasificadorKNNCUDA(dispositivo=dispositivo)
    datos_entrenamiento = np.array([[0.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([1], dtype=np.int64)

    with pytest.raises(RuntimeError, match="todavia no esta integrado"):
        clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert clasificador.ajustado_ is False


def test_clasificador_rechaza_dispositivo_invalido() -> None:
    with pytest.raises(ValueError, match="dispositivo"):
        ClasificadorKNNCUDA(dispositivo="otro")


def test_ajustar_prepara_tensores_cpu_y_dispositivo_efectivo() -> None:
    clasificador = ClasificadorKNNCUDA()
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2], dtype=np.int32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)

    assert clasificador.dispositivo_efectivo_ == torch.device("cpu")
    assert clasificador.datos_entrenamiento_tensor_.device.type == "cpu"
    assert clasificador.etiquetas_entrenamiento_tensor_.device.type == "cpu"
    assert clasificador.datos_entrenamiento_tensor_.dtype == torch.float32
    assert clasificador.etiquetas_entrenamiento_tensor_.dtype == torch.int32
    np.testing.assert_array_equal(
        clasificador.datos_entrenamiento_tensor_.numpy(),
        clasificador.datos_entrenamiento_,
    )
    np.testing.assert_array_equal(
        clasificador.etiquetas_entrenamiento_tensor_.numpy(),
        clasificador.etiquetas_entrenamiento_,
    )


def test_predecir_usa_el_operador_cpu_y_no_la_referencia_numpy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    datos_entrenamiento = np.array([[0.0], [2.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([5, 2], dtype=np.int64)
    datos_consulta = np.array([[0.1], [1.9]], dtype=np.float32)

    def fallar_referencia(*argumentos: object, **argumentos_nombrados: object) -> None:
        raise AssertionError("la referencia NumPy no debe ser el backend operativo")

    monkeypatch.setattr(referencia, "predecir_knn", fallar_referencia)
    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(datos_consulta)

    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "knn_cuda::predecir_knn", "CPU"
    )
    np.testing.assert_array_equal(predicciones, np.array([5, 2], dtype=np.int64))


def test_predecir_cpu_preserva_etiquetas_negativas_no_consecutivas_y_dtype() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=2)
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([-3, 100, 100, -3], dtype=np.int32)
    datos_consulta = np.array([[1.0], [5.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    predicciones = clasificador.predecir(datos_consulta)

    assert predicciones.dtype == np.int32
    np.testing.assert_array_equal(predicciones, np.array([-3, -3], dtype=np.int32))


def test_predecir_cpu_conserva_desempates_y_es_determinista() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=2)
    datos_entrenamiento = np.array([[0.0], [2.0], [4.0], [6.0]], dtype=np.float32)
    etiquetas_entrenamiento = np.array([7, 2, 7, 2], dtype=np.int64)
    datos_consulta = np.array([[1.0], [3.0]], dtype=np.float32)

    clasificador.ajustar(datos_entrenamiento, etiquetas_entrenamiento)
    primeras_predicciones = clasificador.predecir(datos_consulta)
    segundas_predicciones = clasificador.predecir(datos_consulta)

    np.testing.assert_array_equal(primeras_predicciones, np.array([2, 2]))
    np.testing.assert_array_equal(primeras_predicciones, segundas_predicciones)


def test_predecir_cpu_rechaza_consulta_float64_con_error_publico() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    clasificador.ajustar(
        np.array([[0.0]], dtype=np.float32), np.array([1], dtype=np.int64)
    )

    with pytest.raises(TypeError, match="datos_consulta"):
        clasificador.predecir(np.array([[0.0]], dtype=np.float64))


def test_ajustar_por_segunda_vez_reemplaza_el_estado_nativo() -> None:
    clasificador = ClasificadorKNNCUDA(numero_vecinos=1)
    clasificador.ajustar(
        np.array([[0.0], [2.0]], dtype=np.float32),
        np.array([1, 2], dtype=np.int64),
    )
    tensor_anterior = clasificador.datos_entrenamiento_tensor_

    clasificador.ajustar(
        np.array([[10.0], [12.0]], dtype=np.float32),
        np.array([7, 8], dtype=np.int64),
    )
    predicciones = clasificador.predecir(np.array([[10.1]], dtype=np.float32))

    assert clasificador.datos_entrenamiento_tensor_ is not tensor_anterior
    assert clasificador.dispositivo_efectivo_ == torch.device("cpu")
    np.testing.assert_array_equal(predicciones, np.array([7], dtype=np.int64))
