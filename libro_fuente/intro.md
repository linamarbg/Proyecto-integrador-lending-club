# Predicción de incumplimiento en Lending Club: scikit-learn frente a PySpark

**Autores:** Lina Margarita Buelvas Gutiérrez y Camilo Molina Guerrero

**Curso:** Aprendizaje Automático (doctorado) · **Proyecto Integrador de Aprendizaje Automático** (sección 9.10)

## Problema
Predecir si un préstamo de Lending Club termina en incumplimiento (`default = 1` si el estado final es *Charged Off*, `0` si es *Fully Paid*), comparando **seis modelos** en **dos entornos** (scikit-learn y PySpark) con el **conjunto de datos completo**, sin muestreo.

## Qué contiene este libro
| Capítulo | Contenido |
|---|---|
| 1. Datos y EDA | Población de estudio, calidad de datos, análisis univariado y bivariado |
| 2. Metodología | Partición común, preprocesamiento, modelos, búsqueda de hiperparámetros y diferencias entre entornos |
| 3. Resultados | Métricas, tiempos y curvas ROC |
| 4. Pruebas estadísticas | DeLong, McNemar y bootstrap pareado |
| 5. LIME | Interpretación local del mejor modelo de cada entorno |
| 6. Reflexión crítica | Velocidad, precisión, significancia frente a relevancia, límites |
| Apéndice | Los siete cuadernos con el código completo |

## Hardware declarado
Todos los experimentos (scikit-learn **y** PySpark) se ejecutaron en la misma máquina para que los tiempos sean comparables: **Google Colab Pro con RAM alta, 8 núcleos y 54.8 GB de RAM**. PySpark se ejecutó en modo local (`local[*]`): el *driver* y el *executor* comparten una sola JVM.

## Condiciones obligatorias del enunciado y dónde se cumplen
| Condición | Cómo se cumplió |
|---|---|
| Dataset completo, sin muestreo | Se usan los 1,345,310 préstamos cerrados (ver capítulo 1) |
| Sesión de Spark configurada | `shuffle.partitions=400`, `default.parallelism=400`, `memory.fraction=0.8`, `storageFraction=0.3`, memoria 8g (driver y executor; en modo local la JVM única recibe 16g) |
| Sin `.toPandas()` / `.collect()` del dataset completo antes de entrenar | Solo se traen puntuaciones de prueba **después** de entrenar, y una muestra pequeña para LIME |
| `cache`/`persist` tras el `VectorAssembler` | `persist(MEMORY_AND_DISK)` de `train_p` y `test_p` |
| Sin `randomSplit` | La misma partición 80/20 estratificada (semilla 42) se generó una vez y se une por `id` |
| Sin búsquedas manuales con `for` en Spark | `ParamGridBuilder` + `CrossValidator(numFolds=3)` |
| Transformadores ajustados solo con train | Imputación, escalado y codificación se ajustan con `fit` en train |
| Hardware declarado y tiempos medidos | Entrenamiento (incluye la búsqueda), predicción y transferencia |

```{note}
La población de estudio se limita a préstamos con desenlace final conocido (*Fully Paid* y *Charged Off*). No es un muestreo, es la definición del problema, pero conviene confirmarlo con el profesor.
```
