# 3. Resultados

## Métricas en el conjunto de prueba (269,062 préstamos)

```{code-cell} ipython3
:tags: [remove-stderr]
import pandas as pd, numpy as np, os
from IPython.display import display, Markdown
R = 'resultados'
def leer(nombre):
    p = os.path.join(R, nombre)
    return pd.read_csv(p) if os.path.exists(p) else None
sk, sp = leer('resultados_sklearn.csv'), leer('resultados_pyspark.csv')
cols = ['modelo', 'auc', 'accuracy', 'precision', 'recall', 'f1', 't_entrenamiento_s', 't_prediccion_s']
if sk is not None and sp is not None:
    display(Markdown('**scikit-learn**')); display(sk[cols].round(4))
    display(Markdown('**PySpark**')); display(sp[cols].round(4))
else:
    print('Faltan resultados_sklearn.csv o resultados_pyspark.csv')
```

## AUC y tiempos lado a lado

```{code-cell} ipython3
if sk is not None and sp is not None:
    t = pd.DataFrame({
        'modelo': sk['modelo'],
        'AUC sklearn': sk['auc'], 'AUC PySpark': sp.set_index('modelo').loc[sk['modelo'], 'auc'].values,
        'Entren. sklearn (min)': sk['t_entrenamiento_s'] / 60,
        'Entren. PySpark (min)': sp.set_index('modelo').loc[sk['modelo'], 't_entrenamiento_s'].values / 60,
        'Pred. sklearn (s)': sk['t_prediccion_s'],
        'Pred. PySpark (s)': sp.set_index('modelo').loc[sk['modelo'], 't_prediccion_s'].values})
    t['Razón de entren. (Spark/sklearn)'] = t['Entren. PySpark (min)'] / t['Entren. sklearn (min)'].replace(0, np.nan)
    display(t.round(3))
```

```{note}
El tiempo de **entrenamiento** incluye la búsqueda de hiperparámetros con validación cruzada (varios ajustes por modelo). Spark mide además la lectura del CSV, el preprocesamiento y la transferencia de resultados; esos tiempos comunes se reportan aparte en los CSV de resultados.
```

![Tiempos](resultados/pruebas/tiempos.png)

## Curvas ROC
![ROC por entorno](resultados/pruebas/roc_por_entorno.png)

## Lectura de los resultados
- **Los modelos de mayor AUC son Gradient Boosting (0.718 en ambos entornos) y Random Forest (0.713–0.714).** Las diferencias entre los seis modelos son pequeñas en términos absolutos, salvo Naive Bayes (0.63) y el SVM de Spark (0.61).
- **La regresión logística (0.710) queda muy cerca de los ensambles.** Indica que gran parte de la señal es aproximadamente lineal en las variables disponibles.
- **scikit-learn fue más rápido en los seis modelos** con este volumen. La explicación se desarrolla en el capítulo 6.
