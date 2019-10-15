#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://www.leonardosapiras.com.br/op/Dados8.csv')
atributos_independentes = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import DBSCAN
clustering = DBSCAN(eps=3, min_samples=2).fit(atributos_independentes)
core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_

# Quantifica grupos e ruídos
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

clustering.labels_

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    atributos_independentesy = atributos_independentes[class_member_mask & core_samples_mask]
    plt.plot(atributos_independentesy[:, 0], atributos_independentesy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    atributos_independentesy = atributos_independentes[class_member_mask & ~core_samples_mask]
    plt.plot(atributos_independentesy[:, 0], atributos_independentesy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('%d grupos estimados' % n_clusters_)
plt.show()