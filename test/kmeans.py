#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://www.leonardosapiras.com.br/op/Dados8.csv')
atributos_independentes = dataset.iloc[:, [3, 4]].values

# Estratégia do elbow para encontrar número de grupos
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(atributos_independentes)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.atributos_independenteslabel('Grupos')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(atributos_independentes)

# Visualização
plt.scatter(atributos_independentes[y_kmeans == 0, 0], atributos_independentes[y_kmeans == 0, 1], s = 100, c = 'gray', label = 'Grupo 1')
plt.scatter(atributos_independentes[y_kmeans == 1, 0], atributos_independentes[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Grupo 2')
plt.scatter(atributos_independentes[y_kmeans == 2, 0], atributos_independentes[y_kmeans == 2, 1], s = 100, c = 'orange', label = 'Grupo 3')
plt.scatter(atributos_independentes[y_kmeans == 3, 0], atributos_independentes[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Grupo 4')
plt.scatter(atributos_independentes[y_kmeans == 4, 0], atributos_independentes[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Grupo 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Grupos de clientes')
plt.xlabel('Salário anual')
plt.ylabel('Pontuação de gasto')
plt.legend()
plt.show()