#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://www.leonardosapiras.com.br/op/Dados7.csv')
atributos_independentes = dataset.iloc[:, 1:2].values
atributos_dependentes = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(atributos_independentes, atributos_dependentes)

atributos_dependentes_classificados = regressor.predict(atributos_independentes)
y_pred = regressor.predict([[6.5]])

# Visualização
atributos_independentes_grid = np.arange(min(atributos_independentes), max(atributos_independentes), 0.01)
atributos_independentes_grid = atributos_independentes_grid.reshape((len(atributos_independentes_grid), 1))
plt.scatter(atributos_independentes, atributos_dependentes, color = 'blue')
plt.plot(atributos_independentes_grid, regressor.predict(atributos_independentes_grid), color = 'gray')
plt.title('Regressão com árvore de decisão')
plt.xlabel('Cargo')
plt.ylabel('Salário')
plt.show()


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=['Nivel'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())