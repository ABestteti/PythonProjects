#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://www.leonardosapiras.com.br/op/Dados7.csv')
atributos_independentes = dataset.iloc[:, 1:2].values
atributos_dependentes = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_atributos_independentes = StandardScaler()
sc_atributos_dependentes = StandardScaler()
atributos_independentes = sc_atributos_independentes.fit_transform(atributos_independentes)
atributos_dependentes = atributos_dependentes.reshape(-1,1)
atributos_dependentes = sc_atributos_dependentes.fit_transform(atributos_dependentes)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(atributos_independentes, atributos_dependentes)

atributos_dependentes_pred = regressor.predict([[6.5]])
atributos_dependentes_pred = sc_atributos_dependentes.inverse_transform(atributos_dependentes_pred)

# Visualização
plt.scatter(atributos_independentes, atributos_dependentes, color = 'blue')
plt.plot(atributos_independentes, regressor.predict(atributos_independentes), color = 'gray')
plt.title('Regressão com vetor de suporte')
plt.xlabel('Cargo')
plt.ylabel('Salário')
plt.show()