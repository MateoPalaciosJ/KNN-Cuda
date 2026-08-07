# Arquitectura inicial del motor KNN exacto con CUDA

## 1. Objetivo del proyecto

El proyecto tiene como objetivo construir un motor de **K vecinos más cercanos (KNN) exacto**, acelerado con CUDA. La primera versión estará orientada a clasificación y buscará los vecinos mediante fuerza bruta, calculando la distancia entre cada consulta y cada elemento del conjunto de referencia.

La prioridad inicial es obtener resultados correctos, reproducibles y comparables con la implementación CPU propia basada en NumPy y las reglas deterministas de KNN-Cuda. scikit-learn se utilizará únicamente como referencia secundaria en casos compatibles y sin ambigüedad

## 2. Flujo general

El flujo de ejecución previsto es:

```text
Python → extensión C++ → kernels CUDA → GPU
```

1. Python recibe o prepara los datos, configura los parámetros de KNN y expone una API de alto nivel.
2. La extensión C++ recibe los arreglos y parámetros, valida sus propiedades y coordina la ejecución.
3. La extensión lanza kernels CUDA con los datos y la configuración adecuados.
4. La GPU calcula las distancias, identifica los vecinos más cercanos y produce las predicciones.
5. El resultado vuelve desde la extensión C++ a Python en un formato utilizable para el usuario.

## 3. Responsabilidades de la capa Python

- Proporcionar una API sencilla para entrenar o preparar el conjunto de referencia y realizar predicciones.
- Recibir los datos de entrada, las etiquetas y el valor de `k`.
- Comprobar las condiciones de alto nivel: dimensiones compatibles, tipos esperados, valores válidos de `k` y consistencia entre muestras y etiquetas.
- Gestionar la conversión a `float32` cuando sea necesario y documentar claramente esa conversión.
- Preparar los buffers que se entregarán a la extensión C++ y devolver las predicciones en una estructura familiar para el ecosistema Python.
- Exponer errores de forma comprensible, sin ocultar fallos de la capa nativa o de CUDA.
- Facilitar scripts, notebooks y pruebas de comparación contra scikit-learn.

La capa Python no debe contener la lógica intensiva de cálculo de distancias. Su función principal es servir como interfaz, validar entradas y coordinar el uso del motor.

## 4. Responsabilidades de la capa C++

- Ser el límite entre la API de Python y el runtime de CUDA.
- Validar nuevamente los metadatos necesarios para evitar errores de memoria o lanzamientos inválidos.
- Traducir los buffers y parámetros recibidos desde Python a las representaciones nativas que utilizan los kernels.
- Gestionar la memoria de dispositivo y, cuando corresponda, las transferencias entre CPU y GPU.
- Configurar dimensiones de bloques y de la cuadrícula, lanzar los kernels y comprobar los errores de CUDA.
- Coordinar las etapas de cálculo de distancias, selección de vecinos y clasificación.
- Sincronizar la ejecución cuando sea necesario antes de devolver resultados.
- Mantener aislados los detalles de CUDA de la interfaz Python.

La extensión C++ debe ser una capa de coordinación y seguridad. No debe duplicar innecesariamente la lógica numérica que corresponde a los kernels CUDA.

## 5. Responsabilidades de los kernels CUDA

Los kernels CUDA serán responsables del trabajo masivamente paralelo sobre la GPU:

- Calcular la distancia euclidiana cuadrada entre las consultas y los puntos del conjunto de referencia.
- Mantener o producir los candidatos necesarios para encontrar los `k` vecinos de menor distancia.
- Resolver los empates de manera determinista, según una regla documentada.
- Obtener la clase predicha a partir de las etiquetas de los vecinos seleccionados, inicialmente mediante votación mayoritaria.
- Evitar operaciones de precisión o conversiones no documentadas que alteren la equivalencia con la referencia.
- Dejar los resultados en buffers que la capa C++ pueda transferir de vuelta a Python.

La primera implementación puede separar las fases de cálculo, selección y votación para facilitar la verificación. Las fusiones de kernels, el uso de memoria compartida y otras optimizaciones quedan para una etapa posterior.

## 6. Estrategia de validación

La implementación CPU propia basada en NumPy será la referencia primaria de comportamiento de la primera versión. La validación se realizará con conjuntos pequeños, casos controlados y datos aleatorios reproducibles

scikit-learn será una referencia secundaria para comparar resultados en casos compatibles y sin ambigüedad

- Comparar primero las distancias, los índices y las predicciones del motor CUDA con la referencia CPU propia usando las mismas reglas de orden y desempate
- Comparar de forma secundaria las predicciones con `KNeighborsClassifier` usando la misma métrica, el mismo valor de `k` y el mismo conjunto de referencia cuando no existan ambigüedades de desempate
- Confirmar que ambos lados usan datos `float32` y que la distancia comparada es la euclidiana cuadrada, teniendo en cuenta que scikit-learn puede presentar la distancia euclidiana sin elevar al cuadrado en sus salidas.
- Probar distintos tamaños de conjunto, dimensiones, cantidades de consultas y valores válidos de `k`.
- Incluir casos límite: `k = 1`, `k` igual al número de muestras de referencia, clases repetidas y consultas coincidentes con puntos existentes.
- Diseñar casos con empates de distancia y verificar que la regla de desempate esté definida y sea estable. Si la referencia y el motor no resuelven un empate de la misma forma, la diferencia deberá identificarse y documentarse explícitamente.
- Usar tolerancias numéricas para valores de distancia, pero exigir coincidencia exacta de las etiquetas predichas cuando el caso no contiene ambigüedad.
- Automatizar las comparaciones en pruebas que puedan ejecutarse tanto en un entorno CUDA como en un entorno de referencia.

La validación debe cubrir primero la corrección de la salida; el tiempo de ejecución no sustituye a la equivalencia funcional.

## 7. Estrategia de benchmarks

Los benchmarks medirán por separado el comportamiento del motor y el coste total de uso desde Python.

- Medir el tiempo de extremo a extremo: preparación, transferencias, ejecución y retorno del resultado.
- Medir también el tiempo del cálculo en GPU después de una fase de calentamiento, para distinguir el coste fijo de inicialización del coste de cómputo.
- Variar el número de muestras de referencia, consultas, dimensiones y valor de `k`.
- Comparar contra scikit-learn en CPU usando el mismo conjunto de datos y los mismos parámetros.
- Registrar hardware, versión de CUDA, versión de Python, configuración de compilación y características de los datos.
- Repetir cada medición varias veces e informar medidas representativas, como mediana y dispersión.
- Verificar la corrección de los resultados durante los benchmarks para evitar medir una implementación rápida pero incorrecta.

Los resultados deberán almacenarse de forma reproducible en `benchmarks/` y presentarse con suficiente contexto para interpretar cuándo la aceleración compensa el coste de las transferencias.

## 8. Principio de desarrollo

**Primero corrección, luego optimización.**

La arquitectura inicial favorecerá implementaciones simples, comprobables y deterministas. Cada optimización deberá conservar una referencia funcional clara, incluir pruebas de regresión y demostrar una mejora mediante benchmarks. No se considerará terminada una mejora de rendimiento si modifica resultados válidos o introduce comportamientos no reproducibles.

## 9. Arquitectura objetivo de carpetas

La organización prevista del repositorio es:

```text
KNN-Cuda/
├── src/
│   ├── python/          # API y adaptadores de alto nivel
│   ├── cpp/             # Extensión nativa y coordinación CUDA
│   └── cuda/            # Kernels y utilidades de GPU
├── tests/               # Pruebas unitarias, de integración y de equivalencia
├── benchmarks/          # Medición, configuración y resultados de rendimiento
├── data/                # Datos pequeños o referencias reproducibles
├── examples/            # Ejemplos de uso
├── notebooks/           # Exploración y validación en Google Colab
├── docs/                # Documentación de arquitectura y decisiones
└── README.md            # Introducción y guía de inicio
```

La estructura es un objetivo de organización: los nombres concretos de módulos y archivos se definirán al implementar cada componente. La interfaz Python, la extensión C++ y los kernels CUDA deben conservar límites claros para que cada capa pueda probarse de manera independiente.

## 10. Decisiones iniciales

- **Algoritmo:** KNN exacto por fuerza bruta; no se utilizarán aproximaciones ni índices espaciales en la primera versión.
- **Métrica:** distancia euclidiana cuadrada, evitando la raíz cuadrada porque no cambia el orden de las distancias.
- **Tipo de datos:** `float32`, para alinear el formato con el procesamiento habitual en GPU y mantener controlado el consumo de memoria.
- **Tarea inicial:** clasificación mediante las etiquetas de los vecinos y votación mayoritaria.
- **Ejecución CUDA:** Google Colab será el entorno inicial para ejecutar con acceso a una GPU CUDA y facilitar experimentos reproducibles.
- **Desarrollo local:** VS Code será el entorno de trabajo; Git gestionará el historial local y GitHub alojará el repositorio y la colaboración.

Estas decisiones describen el alcance inicial y no impiden futuras extensiones, como regresión, otras métricas, procesamiento por lotes o estrategias de búsqueda más eficientes.
