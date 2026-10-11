# 5. Interpretabilidad local con LIME

Se explica el **Gradient Boosting** de cada entorno (el de mayor AUC probabilístico: 0.7179 en scikit-learn y 0.7174 en PySpark) en dos préstamos mal clasificados con alta seguridad: el **falso positivo** (predicho incumplimiento, pero se pagó) con mayor probabilidad y el **falso negativo** con menor probabilidad. Cada explicación usa 5,000 perturbaciones y 10 variables.

## Resumen de costos y ajuste

```{code-cell} ipython3
import pandas as pd, os
from IPython.display import display
p = 'resultados/lime/lime_resumen.csv'
display(pd.read_csv(p).round(3) if os.path.exists(p) else 'Falta lime_resumen.csv')
```

## scikit-learn
![LIME sklearn](resultados/lime/lime_sklearn.png)

## PySpark
![LIME PySpark](resultados/lime/lime_pyspark.png)

## Qué aporta LIME y cuáles son sus límites
- **Lo que aporta:** los AUC dicen *cuánto* acierta cada entorno; LIME muestra *por qué* se equivoca en préstamos concretos y permite comprobar si dos modelos con AUC casi idéntico razonan de forma parecida.
- **Costo en el entorno distribuido:** una explicación tarda **unas 0.07 s en scikit-learn y unas 4 s en PySpark** (aproximadamente 55 veces más). Cada llamada a la función de predicción lanza un trabajo de Spark con 5,000 perturbaciones. LIME corre en el *driver*; el clúster solo calcula probabilidades.
- **Los datos de fondo viajan al *driver*:** aquí solo una muestra de 20 000 filas de entrenamiento y dos de prueba, nunca el dataset completo.
- **Fidelidad local moderada:** el R² del modelo lineal local está entre 0.39 y 0.53. Un modelo lineal reproduce solo en parte el comportamiento del modelo de árboles en ese vecindario, por lo que las explicaciones son orientativas.
- **Estocástica y en unidades estandarizadas:** las explicaciones cambian con la semilla (el cuaderno mide cuántas de las 5 variables principales se repiten) y las reglas están en desviaciones estándar del entrenamiento.
- **Variables correlacionadas y *one-hot*:** la perturbación puede crear combinaciones poco realistas, y el efecto de variables muy correlacionadas (la tasa de interés con otras de riesgo) se reparte de forma arbitraria.
- **Para explicar muchas instancias en Spark** habría que paralelizar las perturbaciones (por ejemplo con `mapInPandas`) o usar métodos aproximados como SHAP.
