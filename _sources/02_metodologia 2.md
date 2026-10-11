# 2. Metodología

## Preprocesamiento (idéntico en ambos entornos)
1. `term` y `emp_length` (texto) pasan a número de meses y de años.
2. **Numéricas:** imputación con la mediana de entrenamiento y estandarización.
3. **Categóricas:** nulos como `Desconocido` y codificación *one-hot* completa.
4. Todo lo que aprende parámetros (medianas, medias, categorías) se ajusta **solo con entrenamiento** y se aplica a prueba.
5. En PySpark, el resultado se guarda con `persist` justo después del `VectorAssembler`.

## Modelos y búsqueda de hiperparámetros
Validación cruzada de 3 particiones (`GridSearchCV(cv=3)` frente a `CrossValidator(numFolds=3)`), con el **AUC ROC** como criterio.

| Modelo | Rejilla scikit-learn | Rejilla PySpark |
|---|---|---|
| Regresión logística | `C ∈ {0.929, 0.0929, 0.00929}` | `regParam ∈ {1e-6, 1e-5, 1e-4}` |
| Árbol de decisión | `max_depth ∈ {5, 10, 15}` | `maxDepth ∈ {5, 10, 15}` |
| Random Forest | `n_estimators ∈ {10, 50, 100}`, `max_depth ∈ {5, 10, 15}` | `numTrees ∈ {10, 50, 100}`, `maxDepth ∈ {5, 10, 15}` |
| Gradient Boosting | `max_iter ∈ {50, 100}`, `max_depth ∈ {3, 5}` | `maxIter ∈ {50, 100}`, `maxDepth ∈ {3, 5}` |
| SVM lineal | `C` igual que la logística | `regParam ∈ {1e-6, 1e-5, 1e-4}` |
| Naive Bayes | Gaussiano, sin búsqueda | Gaussiano, sin búsqueda |

La equivalencia entre `C` y `regParam` es `C = 1 / (regParam · n)`, con `n = 1,076,248` filas de entrenamiento. Es una equivalencia aproximada.

## Decisiones que se declaran explícitamente
```{admonition} Gradient Boosting en scikit-learn
:class: warning
Se usó `HistGradientBoostingClassifier` en lugar de `GradientBoostingClassifier`, sustitución permitida por el enunciado cuando el costo es prohibitivo con un millón de filas. Es un algoritmo distinto (discretiza las variables en histogramas), por lo que sus tiempos no son comparables 1 a 1 con el `GBTClassifier` de Spark. Se desactivó `early_stopping` para que el número de iteraciones sea el de la rejilla.
```

```{admonition} SVM lineal: la pérdida
:class: warning
La primera versión de scikit-learn usaba `loss='hinge'` (la de Spark). Con ella se obtuvo un modelo degenerado: AUC 0.515 y ningún positivo predicho, a pesar de que el optimizador convergió. Un diagnóstico sobre una submuestra de 200 000 filas mostró que `hinge` dual converge en 94 iteraciones con AUC 0.578, mientras que `squared_hinge` en forma primal da AUC 0.709 en segundos. **El SVM final de scikit-learn usa `squared_hinge`.** Spark solo implementa `hinge`, así que esta es una diferencia real entre entornos y no un error de ejecución.
```

## Diferencias de implementación entre entornos
| Aspecto | scikit-learn | PySpark | Efecto esperado |
|---|---|---|---|
| Mediana de imputación | Exacta | Aproximada | Despreciable con más de un millón de filas |
| Desviación estándar | Poblacional (n) | Muestral (n−1) | Despreciable |
| Cortes de los árboles | Todos los umbrales | `maxBins = 32` | Árboles ligeramente distintos |
| Regresión logística | `lbfgs`, `max_iter = 300` | `maxIter = 100` | Pequeño |
| SVM lineal | `squared_hinge`, primal | `hinge` | **Grande** (ver arriba) |
| Gradient Boosting | Histogramas | Árboles del `GBTClassifier` | Algoritmo distinto |
| Ejecución | Un proceso, hilos en C | JVM local con tareas y *shuffles* | Costo fijo de coordinación |

## Cómo se calcula el AUC en Spark
El `BinaryClassificationEvaluator` por defecto usa `rawPrediction`. Eso da un AUC **incorrecto** en el árbol de decisión (conteos de hojas) y en Naive Bayes (log-verosimilitud sin normalizar). Se evaluaron con la columna `probability` y con `numBins=0` (AUC exacto); el SVM usa `rawPrediction`. Las puntuaciones de prueba de Spark se recalcularon con scikit-learn como control (`auc_chequeo_sklearn`) y coinciden en los seis modelos.

## Umbrales de clasificación
0.5 sobre probabilidades y 0 sobre el margen del SVM. Con 20 % de incumplimiento, el umbral 0.5 produce pocos positivos, por lo que el F1, la precisión y el recall son bajos aunque el AUC sea razonable.
