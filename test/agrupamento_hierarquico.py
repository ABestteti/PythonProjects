#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://www.leonardosapiras.com.br/op/Dados8.csv')
atributos_independentes = dataset.iloc[:, [3, 4]].values

# Criação do dendrograma
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(atributos_independentes, method = 'ward'))
plt.title('Dendrograma')
plt.xlabel('Clientes')
plt.ylabel('Distância euclidiana')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(atributos_independentes)

# Visualização
plt.scatter(atributos_independentes[y_hc == 0, 0], atributos_independentes[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(atributos_independentes[y_hc == 1, 0], atributos_independentes[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(atributos_independentes[y_hc == 2, 0], atributos_independentes[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(atributos_independentes[y_hc == 3, 0], atributos_independentes[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(atributos_independentes[y_hc == 4, 0], atributos_independentes[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Grupos de clientes')
plt.xlabel('Salário anual')
plt.ylabel('Pontuação de gasto')
plt.legend()
plt.show()