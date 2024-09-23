# YOM_desafio

Instalación

```
!pip install surprise
```
```
# Used to ignore the warning given as output of the code
import warnings
warnings.filterwarnings('ignore')

# Basic libraries of python for numeric and dataframe computations
import numpy as np
import pandas as pd

# Basic library for data visualization
import matplotlib.pyplot as plt

# Slightly advanced library for data visualization
import seaborn as sns

# A dictionary output that does not raise a key error
from collections import defaultdict

# A performance metrics in surprise
from surprise import accuracy

# Class is used to parse a file containing ratings, data should be in structure - user ; item ; rating
from surprise.reader import Reader

# Class for loading datasets
from surprise.dataset import Dataset

# For model tuning model hyper-parameters
from surprise.model_selection import GridSearchCV

# For splitting the rating data in train and test dataset
from surprise.model_selection import train_test_split

# For implementing similarity based recommendation system
from surprise.prediction_algorithms.knns import KNNBasic

# For implementing matrix factorization based recommendation system
from surprise.prediction_algorithms.matrix_factorization import SVD

# For implementing cross validation
from surprise.model_selection import KFold
```

Calcular recomendaciones
```
# Df_grouped es el DataFrame con los datos de transacciones
top_recommendations = get_recommendations_for_district(df_grouped_filtered, 'Macul', new_id_commerce=12, top_n=10, algo=similarity_algo_optimized_item) #algo_knn_item es el modelo 3

# Imprimir las recomendaciones
for product_id, predicted_quantity in top_recommendations:
    print(f"Producto {product_id}: Cantidad predicha {predicted_quantity:.2f}")
```
Donde el argumento
1) df_gropued_filtered: Es la base de 4.965 registros, sin outliers y agrupadas por suma de cantidades por id_commerce, id_product
2) "Macul" es el argumento de district. Hay 5 opciones disponibles según la información contenida en el dataframe: Providencia, Nunoa, Penalolen, La Florida
3) new_id_commerce toma valores del 1 al 100. Al elegir un comercio se calculan los productos más populares junto a sus cantidades predichas
4) top_n es la cantidad de recomendaciones que se quieren generar de mayor a menor valor
5) algo es el argumento donde se inputa el modelo trabajado. Hay 6 modelos: algo_knn, similarity_algo_optimized_user, algo_knn_item, similarity_algo_optimized_item, algo_svd, svd_algo_optimized
