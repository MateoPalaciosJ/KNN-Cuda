# Requisitos iniciales de KNN-Cuda


## 1. Principios de diseño
- La corrección tiene prioridad sobre el rendimiento.
- Toda optimización debe estar respaldada por benchmarks.
- Ningún cambio se acepta sin pruebas.
- La API pública debe mantenerse simple.
- Los módulos deben tener una única responsabilidad.

## 2. Convenciones del proyecto

- El código propio del proyecto se escribe en español latinoamericano
- Los identificadores propios se escriben sin tildes ni ñ
- Los nombres externos y las APIs de terceros conservan su nombre original
- Idioma de la documentación: español.
- Commits siguiendo Conventional Commits.
- snake_case para Python.
- camelCase solo cuando sea necesario en C++.
- Toda función pública deberá documentarse.




## 3. Objetivo funcional

KNN-Cuda debe implementar un clasificador KNN exacto cuyo cálculo principal se ejecute en una GPU mediante CUDA. El motor debe exponer una interfaz sencilla desde Python para preparar los datos, consultar vecinos y obtener predicciones sin que el usuario tenga que gestionar directamente los detalles de C++ o CUDA.

## 4. Requisitos funcionales

El sistema debe:

- Soportar clasificación binaria y multiclase.
- Recibir datos de entrada en formato `float32`.
- Utilizar la distancia euclidiana cuadrada.
- Permitir configurar el valor de `k`.
- Implementar las operaciones `ajustar()`, `vecinos_mas_cercanos()` y `predecir()`
- Devolver índices de vecinos, distancias y predicciones según la operación solicitada.
- Soportar inicialmente votación uniforme entre los vecinos.
- Procesar varias consultas por lote.

## 5. Requisitos no funcionales

El proyecto debe cumplir los siguientes requisitos de calidad y operación:

- Producir resultados reproducibles bajo las mismas condiciones de entrada y configuración.
- Mantener una organización de código modular, con límites claros entre Python, C++ y CUDA.
- Contar con pruebas automáticas para validar comportamiento y prevenir regresiones.
- Mantener documentación técnica actualizada sobre arquitectura, requisitos y decisiones.
- Permitir la ejecución de CUDA en Google Colab.
- Permitir el desarrollo local sin requerir una GPU.
- Utilizar Git para el control de versiones y GitHub como repositorio remoto.
- Mantener compatibilidad inicial con Linux en Google Colab.
- Mostrar mensajes de error claros para entradas inválidas, fallos de ejecución y problemas de configuración.

## 6. Criterios de corrección

Una implementación se considerará correcta cuando cumpla lo siguiente:

- Los vecinos encontrados coinciden con una referencia basada en NumPy o con scikit-learn.
- Las predicciones coinciden con scikit-learn en casos controlados.
- Las distancias coinciden dentro de una tolerancia numérica adecuada para operaciones en `float32`.
- Existen pruebas para conjuntos pequeños, empates de distancia y clasificación multiclase.

Las pruebas deben usar datos reproducibles y documentar cualquier diferencia esperada relacionada con el desempate o con la precisión numérica.

## 7. Criterios de rendimiento

La evaluación del rendimiento debe:

- Medir por separado el tiempo de CPU y el tiempo de GPU.
- Separar el tiempo de transferencia de datos del tiempo de ejecución del kernel.
- Medir el número de consultas procesadas por segundo.
- Calcular el *speedup* respecto a la referencia de CPU seleccionada.
- Probar distintos tamaños de dataset, dimensionalidad y valores de `k`.
- Evitar afirmar mejoras de rendimiento sin benchmarks reproducibles y con resultados revisados.

Los benchmarks deben registrar las condiciones de ejecución relevantes, incluyendo hardware, versiones de software, tamaño de los datos y configuración usada.

## 8. Entorno de trabajo

- **Desarrollo local:** Windows y VS Code.
- **Referencia, pruebas e interfaz:** Python.
- **Compilación y ejecución CUDA:** Google Colab con una GPU NVIDIA.
- **Repositorio remoto:** GitHub.

El entorno local debe permitir preparar documentación, desarrollar la interfaz, ejecutar pruebas que no dependan de CUDA y revisar cambios sin disponer de una GPU. La validación completa del motor CUDA se realizará en el entorno Linux de Google Colab.

## 9. Definición de terminado para cada etapa

- El código pasa revisión.

Cada etapa se considerará terminada cuando se hayan cumplido todos estos puntos:

- El código de la etapa está implementado.
- Las pruebas correspondientes han sido superadas.
- La documentación relevante está actualizada.
- Los resultados han sido revisados y son coherentes con los criterios de corrección y rendimiento aplicables.
- Existe un commit pequeño y descriptivo que representa los cambios de la etapa.

La definición de terminado describe el proceso esperado de desarrollo; la creación de commits queda sujeta a la operación explícita correspondiente en cada cambio.

## 10. Fuera del alcance inicial

Quedan fuera del alcance de la primera versión:

- Búsqueda aproximada.
- Ejecución multi-GPU.
- Regresión.
- Métricas arbitrarias distintas de la distancia euclidiana cuadrada definida inicialmente.
- Soporte para CPU dentro del motor CUDA.
- Despliegue como servicio web.
