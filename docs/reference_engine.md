# Motor CPU de referencia

## 1  Objetivo

El motor CPU implementado con NumPy será la fuente primaria de verdad para validar la implementación CUDA

Su objetivo no es alcanzar rendimiento competitivo, sino ofrecer una base correcta, determinista y fácil de probar. La implementación debe ser sencilla de leer y suficientemente explícita para que cualquier diferencia con CUDA pueda localizarse en una etapa concreta

scikit-learn será una referencia secundaria para validar predicciones en casos controlados y sin ambigüedad. La referencia primaria seguirá siendo NumPy, porque permite comprobar por separado distancias, selección de vecinos y votación bajo las reglas exactas del proyecto

## 2  Alcance inicial

El motor CPU incluirá únicamente:

- Distancia euclidiana cuadrada
- Búsqueda exacta
- Clasificación
- Votación uniforme
- Etiquetas enteras
- Datos `float32`
- Índices `int64`
- Procesamiento en CPU con NumPy

Quedan fuera de esta fase:

- Regresión
- Búsqueda aproximada
- Otras métricas
- Ponderación por distancia
- CUDA
- PyTorch como motor de cálculo
- Procesamiento por bloques
- Optimización

El motor recorrerá el conjunto completo de entrenamiento para conservar una definición directa de la búsqueda exacta. No se añadirá una ruta alternativa de ejecución ni una estrategia de memoria que complique la referencia

## 3  Funciones del módulo

Las funciones públicas del archivo `reference.py` serán únicamente:

- `distancias_l2_cuadradas`, responsable del cálculo de distancias
- `seleccionar_top_k`, responsable de seleccionar y ordenar vecinos
- `votacion_uniforme`, responsable de contar votos y resolver empates
- `predecir_knn`, responsable de coordinar el flujo completo de clasificación

Cada función debe tener una responsabilidad única, un contrato explícito y una salida determinista. `predecir_knn` debe reutilizar las otras funciones en lugar de repetir su lógica

No se agregarán funciones públicas adicionales. Solo podrá proponerse una función auxiliar privada si evita una duplicación real, tiene una responsabilidad clara, recibe y devuelve datos bien definidos y puede probarse de forma independiente. Una función privada no debe ocultar la lógica principal ni convertirse en una capa de abstracción innecesaria

Antes de implementar cualquier función deben quedar definidos sus datos de entrada, su salida, su responsabilidad, aquello que no debe hacer y las pruebas que demostrarán su comportamiento

## 4  Contrato de `distancias_l2_cuadradas`

### Entradas

- `datos_consulta` con forma `[Q, D]`
- `datos_entrenamiento` con forma `[N, D]`
- Ambos valores como `numpy.ndarray`
- Ambos valores en `float32`
- Ambos arreglos bidimensionales
- Misma cantidad de características en la segunda dimensión

### Salida

- Matriz de distancias cuadradas con forma `[Q, N]`
- Tipo `float32`

### Responsabilidad

Calcular exclusivamente la distancia euclidiana cuadrada entre cada consulta y cada muestra de entrenamiento

### No debe

- Ordenar vecinos
- Aplicar raíz cuadrada
- Conocer etiquetas
- Realizar votación
- Modificar las entradas

La función debe producir la misma matriz para las mismas entradas y no debe depender de un estado externo

## 5  Contrato de `seleccionar_top_k`

### Entradas

- Matriz de distancias con forma `[Q, N]`
- `k` como entero válido

### Salidas

- Distancias cuadradas seleccionadas con forma `[Q, K]`
- Índices con forma `[Q, K]` y tipo `int64`

### Responsabilidad

- Seleccionar los `k` vecinos más cercanos por consulta
- Ordenar los vecinos por distancia ascendente
- Resolver distancias iguales mediante el menor índice de entrenamiento

### No debe

- Calcular distancias
- Conocer etiquetas
- Realizar votación
- Aplicar raíz cuadrada
- Modificar la matriz de entrada

La función debe devolver las distancias en su forma cuadrada, porque la raíz cuadrada no participa en la selección del orden

## 6  Contrato de `votacion_uniforme`

### Entradas

- `neighbor_labels` con forma `[Q, K]`
- Etiquetas enteras
- Clases originales ordenadas o una representación equivalente claramente definida

### Salida

- Predicciones con forma `[Q]`
- Etiquetas en la representación original

### Responsabilidad

- Contar los votos de cada consulta
- Elegir la clase con más votos
- En caso de empate, elegir la etiqueta original menor

### No debe

- Calcular distancias
- Seleccionar vecinos
- Acceder a `datos_entrenamiento` o `datos_consulta`
- Modificar las entradas

La función solo recibirá las etiquetas de los vecinos ya seleccionados. No debe inferir índices ni recuperar etiquetas desde el conjunto de entrenamiento

## 7  Contrato de `predecir_knn`

### Entradas

- `datos_entrenamiento`
- `etiquetas_entrenamiento`
- `datos_consulta`
- `k`

Las características deben cumplir el contrato de datos `float32` de la fase y las etiquetas deben ser enteras

### Salida

- Predicciones con forma `[Q]`
- Etiquetas originales

### Responsabilidad

- Coordinar las funciones anteriores sin duplicar su lógica
- Calcular las distancias mediante `distancias_l2_cuadradas`
- Seleccionar vecinos mediante `seleccionar_top_k`
- Recuperar las etiquetas de los índices seleccionados
- Aplicar votación uniforme mediante `votacion_uniforme`

### No debe

- Reimplementar internamente la lógica de distancia
- Reimplementar internamente la lógica de Top-K
- Reimplementar internamente la lógica de votación
- Modificar las entradas
- Introducir optimizaciones prematuras

`predecir_knn` será el punto de coordinación del pipeline CPU y no debe convertirse en un contenedor de lógica que corresponda a las funciones especializadas

## 8  Validaciones

Cada función pública validará únicamente las precondiciones directas de su propio contrato. De esta forma, una función puede fallar con claridad cuando se utiliza de manera independiente sin trasladar todas las validaciones a una única función central

### `distancias_l2_cuadradas`

Validará que `datos_consulta` y `datos_entrenamiento` sean `numpy.ndarray`, bidimensionales, no vacíos, de tipo `float32`, con el mismo número de características y sin valores `NaN` ni infinito

### `seleccionar_top_k`

Validará que la matriz de distancias sea un `numpy.ndarray` bidimensional, no vacía, con valores finitos, y que `k` sea un entero mayor o igual a `1` y menor o igual que `N`

### `votacion_uniforme`

Validará que `neighbor_labels` sea un arreglo bidimensional no vacío, que contenga etiquetas enteras, que el número de clases sea compatible con las etiquetas y que la representación de clases originales esté ordenada y sea inequívoca

### `predecir_knn`

Validará que `datos_entrenamiento`, `etiquetas_entrenamiento` y `datos_consulta` sean entradas NumPy válidas, que las características sean bidimensionales y `float32`, que no estén vacías ni contengan `NaN` o infinito, que `etiquetas_entrenamiento` use etiquetas enteras, que el número de etiquetas coincida con el número de muestras de entrenamiento, que entrenamiento y consultas tengan la misma cantidad de características y que `k` sea un entero mayor o igual a `1` y menor o igual que `N`

Las funciones públicas deben fallar con errores claros, indicar la condición incumplida y evitar mensajes genéricos que oculten el origen del problema. Las validaciones compartidas solo podrán extraerse a auxiliares privadas si existe duplicación real y la separación conserva una responsabilidad única por función

## 9  Determinismo

El comportamiento determinista se define formalmente mediante estas reglas:

- Los vecinos se ordenan por distancia ascendente
- En empate de distancia gana el menor índice de entrenamiento
- En empate de votos gana la etiqueta original menor
- Las mismas entradas deben producir exactamente los mismos índices y predicciones

La selección debe conservar el índice original de cada muestra durante todo el proceso. No se aceptará un orden dependiente de la implementación interna de NumPy cuando existan distancias iguales

## 10  Estrategia de pruebas

Las pruebas deben estar separadas por función para localizar con precisión cualquier error

### `distancias_l2_cuadradas`

- Una consulta
- Varias consultas
- Una dimensión
- Varias dimensiones
- Distancia cero
- Puntos duplicados
- Comparación con un cálculo manual

### `seleccionar_top_k`

- `k` igual a `1`
- `k` igual a `N`
- Distancias empatadas
- Orden determinista por índice
- Varias consultas

### `votacion_uniforme`

- Clasificación binaria
- Clasificación multiclase
- Una sola clase
- Empate de votos
- Etiquetas no consecutivas

### `predecir_knn`

- Pipeline completo
- Comparación con resultados manuales
- Comparación con scikit-learn en casos sin ambigüedad
- Reproducibilidad

### Entradas inválidas

- `NaN`
- Infinito
- `k` igual a `0`
- `k` mayor que `N`
- Formas incompatibles
- Número incorrecto de etiquetas
- Tipos inválidos
- Entradas vacías

Las pruebas deben comprobar tanto valores como formas y tipos de salida. En los casos con distancias iguales deben comprobar los índices exactos y en los casos con votos iguales deben comprobar la etiqueta original seleccionada

## 11  Criterios de aceptación

La fase se considera correcta cuando:

- Todas las funciones respetan su contrato
- No existe lógica duplicada
- Cada función tiene una sola responsabilidad
- Todas las pruebas pasan
- Los resultados coinciden con los casos manuales
- Los resultados coinciden con scikit-learn cuando sus reglas son compatibles
- Los empates siguen las reglas del proyecto
- No existen dependencias distintas de NumPy y pytest para esta fase
- El código puede entenderse sin conocer CUDA
- La implementación sirve como referencia estable para las fases posteriores

La aceptación requiere revisar también que los errores sean claros, que las entradas no se modifiquen y que los índices y predicciones sean reproducibles

## 12  Archivo objetivo

La implementación futura vivirá en:

`src/python/knn_cuda/reference.py`

Las pruebas vivirán en:

`tests/cpu/test_reference.py`

El módulo de referencia debe permanecer separado de la extensión CUDA y no debe depender de PyTorch ni de scikit-learn para realizar sus cálculos

## 13  Qué no debe hacerse en esta fase

- No crear `ClasificadorKNNCUDA`
- No crear kernels CUDA
- No usar PyTorch para calcular
- No usar scikit-learn dentro de la implementación
- No crear clases innecesarias
- No agregar caché
- No agregar procesamiento por bloques
- No optimizar memoria
- No agregar funciones no especificadas
- No mezclar validación, distancia, selección y votación en una sola función

La fase debe terminar con una referencia CPU directa, estable y fácil de depurar, sin decisiones pendientes sobre el comportamiento descrito en este documento
