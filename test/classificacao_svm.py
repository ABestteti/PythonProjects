#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Extração de dados
dataset = pd.read_csv("https://www.leonardosapiras.com.br/dados.csv")
atributos_independentes = dataset.iloc[:, [2, 3]].values
atributos_dependentes = dataset.iloc[:, 4].values


# Divide em treino e teste
from sklearn.model_selection import train_test_split
atributos_independentes_treino, atributos_independentes_teste, atributos_dependentes_treino, atributos_dependentes_teste = train_test_split(atributos_independentes, atributos_dependentes, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
atributos_independentes_treino = sc.fit_transform(atributos_independentes_treino)
atributos_independentes_teste = sc.transform(atributos_independentes_teste)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classificador = SVC(kernel = 'rbf', random_state = 0)
classificador.fit(atributos_independentes_treino, atributos_dependentes_treino)

# Realiza classificação
atributos_dependentes_classificados = classificador.predict(atributos_independentes_teste)



# Matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(atributos_dependentes_teste, atributos_dependentes_classificados)
cm


# Grafico com conjunto de treino
from matplotlib.colors import ListedColormap
atributos_independentes_set, atributos_dependentes_set = atributos_independentes_treino, atributos_dependentes_treino
X1, X2 = np.meshgrid(np.arange(start = atributos_independentes_set[:, 0].min() - 1, stop = atributos_independentes_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = atributos_independentes_set[:, 1].min() - 1, stop = atributos_independentes_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'gray')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(atributos_dependentes_set)):
    plt.scatter(atributos_independentes_set[atributos_dependentes_set == j, 0], atributos_independentes_set[atributos_dependentes_set == j, 1],
                c = ListedColormap(('blue', 'gray'))(i), label = j)
    
plt.title('Kernel SVM (Conjunto de treino)')
plt.xlabel('Idade')
plt.ylabel('Salario')
plt.legend()
plt.show()

# Grafico com conjunto de teste
from matplotlib.colors import ListedColormap
atributos_independentes_set, atributos_dependentes_set = atributos_independentes_teste, atributos_dependentes_teste
X1, X2 = np.meshgrid(np.arange(start = atributos_independentes_set[:, 0].min() - 1, stop = atributos_independentes_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = atributos_independentes_set[:, 1].min() - 1, stop = atributos_independentes_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'gray')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(atributos_dependentes_set )):
    plt.scatter(atributos_independentes_set[atributos_dependentes_set  == j, 0], atributos_independentes_set[atributos_dependentes_set == j, 1],
                c = ListedColormap(('blue', 'gray'))(i), label = j)
plt.title('Kernel SVM (Conjunto de teste)')
plt.xlabel('Idade')
plt.ylabel('Salario')
plt.legend()
plt.show()