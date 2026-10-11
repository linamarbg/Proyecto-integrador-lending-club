# 4. Pruebas estadísticas

Todas las puntuaciones están calculadas sobre **las mismas 269,062 observaciones de prueba**, por lo que las comparaciones son **pareadas**.

## AUC con intervalo de confianza del 95 % (DeLong)

```{code-cell} ipython3
import pandas as pd, os
from IPython.display import display, Markdown
R = 'resultados/pruebas'
def leer(n):
    p = os.path.join(R, n)
    return pd.read_csv(p) if os.path.exists(p) else None
a = leer('auc_ic95.csv')
display(a.round(4) if a is not None else 'Falta auc_ic95.csv')
```

## DeLong entre entornos (mismo modelo, 6 comparaciones, Holm)
`ΔAUC = AUC(sklearn) − AUC(PySpark)`. *Significativo* si `p_holm < 0.05`; *relevante* si `|ΔAUC| ≥ 0.005`.

```{code-cell} ipython3
d = leer('delong_entre_entornos.csv')
display(d.round(4) if d is not None else 'Falta delong_entre_entornos.csv')
```

![ΔAUC entre entornos](resultados/pruebas/delong_entre_entornos.png)

## DeLong entre modelos (15 pares por entorno, Holm)
```{code-cell} ipython3
for e in ('sklearn', 'pyspark'):
    t = leer(f'delong_entre_modelos_{e}.csv')
    display(Markdown(f'**{e}**'))
    display(t.round(4) if t is not None else f'Falta delong_entre_modelos_{e}.csv')
```

![Mapas de calor](resultados/pruebas/heatmaps_delong.png)

## McNemar y bootstrap pareado
- **McNemar:** compara las decisiones de clase (umbral 0.5; 0 para el SVM) usando solo las observaciones donde los dos modelos discrepan.
- **Bootstrap pareado** (B = 2000, semilla 42, misma remuestra para ambos modelos): intervalo percentil del 95 % para ΔAUC, ΔAUC-PR y ΔF1.
- Se aplican a las 6 comparaciones entre entornos y a los dos mejores modelos de cada entorno. Holm se aplica dentro de cada familia.

```{code-cell} ipython3
r = leer('resumen_tres_pruebas.csv')
pd.set_option('display.max_colwidth', None); pd.set_option('display.width', 250)
display(r if r is not None else 'Falta resumen_tres_pruebas.csv')
```

## Lectura
1. **Entre entornos, solo el SVM difiere de forma relevante** (ΔAUC = 0.103). La causa es la pérdida: `squared_hinge` frente a `hinge`.
2. **Naive Bayes, Random Forest y el árbol de decisión** muestran diferencias de AUC estadísticamente significativas pero **menores a 0.005** (entre 0.0005 y 0.0012): son equivalentes en la práctica.
3. **La regresión logística y Gradient Boosting** no muestran diferencia significativa de AUC.
4. **Las tres pruebas no siempre coinciden.** En Gradient Boosting el AUC es el mismo pero ΔF1 = −0.025: Spark predice más positivos con el umbral 0.5. DeLong compara cómo ordenan los modelos; McNemar y F1 miran decisiones con un umbral.
5. **Gradient Boosting supera a Random Forest en ambos entornos** (ΔAUC 0.0036 en scikit-learn y 0.0044 en Spark, p < 0.001), pero por debajo del margen de relevancia de 0.005.
6. **El ΔF1 de 0.075 a 0.105 entre esos dos modelos se debe sobre todo al umbral**: con 0.5, Random Forest casi no predice incumplimientos (recall de 2 %).
