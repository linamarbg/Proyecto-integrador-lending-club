# 6. Reflexión crítica

## ¿Qué entorno fue más rápido y por qué?
**scikit-learn fue más rápido en los seis modelos** con este volumen de datos y esta máquina.

```{code-cell} ipython3
import pandas as pd, numpy as np, os
from IPython.display import display
p = 'resultados/pruebas/tiempos.csv'
if os.path.exists(p):
    t = pd.read_csv(p)
    display(t.round(2))
else:
    print('Falta tiempos.csv')
```

Razones:
1. **Todo cabe en una sola máquina.** El conjunto preprocesado ocupa unos 0.5 GB (1.3 millones de filas por 91 variables). Spark se diseñó para repartir datos que no caben en un nodo; aquí paga la coordinación sin obtener beneficio de la distribución.
2. **Costo fijo de Spark:** serialización entre Python y la JVM, planificación de tareas, *shuffles* y lectura del CSV como DataFrame. En modo local no hay red, pero los costos de coordinación se mantienen.
3. **Los algoritmos de scikit-learn están muy optimizados en una máquina.** El `HistGradientBoostingClassifier` usa histogramas y hilos en C; el Random Forest de Spark recorre los datos nivel a nivel con `maxBins = 32`.
4. **Los tiempos incluyen la validación cruzada** (cada modelo se ajusta varias veces).
5. **Comparabilidad.** Los modelos de Gradient Boosting y el SVM son algoritmos diferentes en cada entorno, por lo que sus razones de tiempo no miden solo "Spark frente a scikit-learn".

## ¿Qué entorno fue más preciso?
**Prácticamente empatan.** Para cinco de los seis modelos, el AUC coincide hasta el tercer decimal (diferencias de 0.0000 a 0.0012). El mejor modelo es Gradient Boosting en ambos (0.7179 y 0.7174). La única diferencia grande es el SVM (0.709 frente a 0.607), por la pérdida utilizada.

## Diferencias significativas frente a prácticamente relevantes
Con 269,062 observaciones de prueba, DeLong tiene tanto poder que detecta diferencias de AUC de 0.0005 (por ejemplo, en Naive Bayes y Random Forest entre entornos). Por eso se define un margen de relevancia práctica de 0.005 y se informa por separado la **significancia** (p ajustado por Holm) y la **relevancia** (tamaño de la diferencia). El único caso significativo **y** relevante entre entornos es el SVM.

## Diferencias de implementación que afectan los resultados
Mediana aproximada frente a exacta, desviación muestral frente a poblacional, `maxBins = 32` en los árboles, solucionadores distintos en la regresión logística, la pérdida del SVM y el algoritmo de Gradient Boosting (ver capítulo 2). En conjunto explican las diferencias de AUC de orden 0.001 y la discrepancia del SVM.

## Limitaciones de DeLong y pruebas complementarias
- **DeLong** asume que los AUC son aproximadamente normales (razonable con muestras grandes), compara solo el AUC (no la calibración ni los costos de error) y trata el conjunto de prueba como fijo: **no incluye la variabilidad del entrenamiento ni de la búsqueda de hiperparámetros**.
- **McNemar** evalúa decisiones con un umbral fijo. Con 20 % de incumplimiento y umbral 0.5 hay pocos positivos, de modo que puede discrepar del AUC.
- **El bootstrap pareado** añade ΔAUC-PR (más sensible con clases desbalanceadas) y ΔF1, pero remuestrea solo la prueba, no repite el entrenamiento.
- **Una sola partición:** no se repitió el experimento con otras semillas; la validación cruzada interna solo sirve para elegir hiperparámetros.

## ¿A partir de qué volumen PySpark supera a scikit-learn?
**Este experimento no lo mide**, porque se probó un único tamaño (1.3 millones de filas). Lo que sí se puede afirmar:
- Con 1.3 millones de filas y 91 variables, el problema usa unos 0.5 GB. La máquina tiene 54.8 GB, por lo que scikit-learn tendría margen para volúmenes cientos de veces mayores antes de quedarse sin memoria.
- El punto de cruce depende de **tener varios nodos**: en una sola máquina Spark local no puede superar a scikit-learn si ambos usan los mismos núcleos. La ventaja aparece cuando los datos ya no caben en la memoria de un nodo o cuando el tiempo de ajuste requiere repartir el trabajo en un clúster.
- Para ubicarlo empíricamente habría que repetir el experimento replicando el conjunto (por ejemplo 2×, 5×, 10×) y ver dónde la curva de tiempo de scikit-learn crece más rápido que la de Spark, idealmente con un clúster real. Es un posible trabajo futuro.

## Qué aporta LIME
Complementa las métricas: muestra por qué se equivocan los modelos en casos concretos y revela el costo de interpretar en un entorno distribuido (unas 55 veces más lento por explicación). Sus límites (estocástico, local, fidelidad moderada) se detallan en el capítulo 5.

En los cuatro préstamos explicados, **la tasa de interés es la variable que más pesa**, seguida del FICO y del DTI (y del plazo en scikit-learn). Los errores que cometen los modelos son errores *confiados* (P = 0.74 a 0.87 en los falsos positivos y 0.02 en los falsos negativos): en los falsos negativos todas las señales disponibles indicaban bajo riesgo, de modo que el incumplimiento probablemente se debió a factores que el conjunto de datos no recoge. Una limitación práctica: en PySpark las variables salieron sin nombre (`x0`, `x1`, …), lo que obligó a identificarlas por orden y por umbral; en un trabajo futuro conviene propagar los nombres desde el *pipeline*.

## Efecto de cada condición obligatoria
| Condición | Efecto sobre el estudio |
|---|---|
| Dataset completo | Da alto poder estadístico (detecta diferencias minúsculas) y hace visibles los costos reales de entrenamiento (Random Forest de 111 min en Spark) |
| Configuración de Spark | Con 400 particiones en una máquina de 8 núcleos probablemente hay más tareas pequeñas de las necesarias (no se midió el efecto); se mantuvo por cumplir el enunciado |
| Sin `toPandas()` previo | Obliga a que el preprocesamiento y el entrenamiento ocurran dentro de Spark; las puntuaciones se traen solo después |
| `cache` tras el `VectorAssembler` | Evita recalcular el *pipeline* y releer el CSV en cada ajuste de la validación cruzada |
| Sin `randomSplit` | Garantiza la misma partición en ambos entornos, base de las pruebas pareadas |
| Sin bucles manuales | `CrossValidator` evalúa la rejilla en paralelo dentro de Spark |
| Ajustar con train solamente | Evita fuga de información; las pruebas reflejan generalización |
| Declarar hardware y medir tiempos | Permite una comparación honesta: ambos entornos en la misma máquina |

## Limitaciones del estudio
1. Spark en modo local, sin clúster real.
2. Se usó `HistGradientBoosting` en scikit-learn (algoritmo distinto al de Spark).
3. El SVM con `squared_hinge` en scikit-learn frente a `hinge` en Spark.
4. Un solo conjunto de prueba, una sola semilla y un único tamaño de datos.
5. Umbral fijo de 0.5, que penaliza el F1 y el recall con 20 % de incumplimiento; no se optimizó el umbral.
6. La población de estudio excluye préstamos sin desenlace conocido.
