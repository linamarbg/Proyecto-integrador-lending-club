# Proyecto Integrador de Aprendizaje Automático: Lending Club

Predicción de incumplimiento de préstamos de Lending Club comparando seis modelos en **scikit-learn** y **PySpark**, con el conjunto de datos completo, pruebas estadísticas (DeLong, McNemar y bootstrap pareado) e interpretabilidad local con LIME.

**Autores:** Lina Margarita Buelvas Gutiérrez y Camilo Molina Guerrero

**Libro (Jupyter Book):** _pega aquí el enlace de GitHub Pages cuando esté publicado_

## Contenido del repositorio
| Carpeta o archivo | Qué contiene |
|---|---|
| `index.html` y archivos HTML | El Jupyter Book ya compilado (lo que se ve en el enlace) |
| `notebooks/` | Los ocho cuadernos, en el orden de ejecución |
| `libro_fuente/` | Los textos y la configuración con los que se genera el libro |

## Orden de los cuadernos
1. `01_carga_y_verificacion`: carga y verificación del dataset
2. `02_target_y_particion`: variable objetivo y partición 80/20 común (semilla 42)
3. `03_sklearn_modelado`: modelos en scikit-learn
4. `04_pyspark_modelado`: modelos en PySpark
5. `05_pruebas_estadisticas`: DeLong, McNemar y bootstrap pareado
6. `06_eda`: análisis exploratorio
7. `07_lime`: interpretabilidad local
8. `08_construir_libro`: compilación del Jupyter Book

## Datos
Los datos provienen de Kaggle (*wordsforthewise/lending-club*, archivo `accepted_2007_to_2018Q4`). No se incluyen en este repositorio por su tamaño.

## Entorno
Todos los experimentos se ejecutaron en Google Colab Pro con RAM alta (8 núcleos, 54.8 GB), tanto para scikit-learn como para PySpark (modo local).
