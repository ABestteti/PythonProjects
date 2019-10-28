#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Extração de dados
dataset = pd.read_csv('https://www.leonardosapiras.com.br/op/Dados4.csv')
atributos_independentes = dataset.iloc[:, :-1].values
atributos_dependentes = dataset.iloc[:, 1].values

# Divide em treino e teste
from sklearn.model_selection import train_test_split
atributos_independentes_treino, atributos_independentes_teste, atributos_dependentes_treino, atributos_dependentes_teste = train_test_split(atributos_independentes, atributos_dependentes, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(atributos_independentes_treino, atributos_dependentes_treino)

# Regressao
atributos_dependentes_classificado = regressor.predict(atributos_independentes_teste)

# Avaliação de resultado
acuracia = regressor.score(atributos_independentes_teste, atributos_dependentes_teste)

# Visualização de dados de treino
plt.scatter(atributos_independentes_treino, atributos_dependentes_treino, color = 'blue')
plt.plot(atributos_independentes_treino, regressor.predict(atributos_independentes_treino), color = 'gray')
plt.title('Salário e experiência - Treino')
plt.xlabel('Anos de experiência')
plt.ylabel('Salário')
plt.show()

# Visualização de dados de teste
plt.scatter(atributos_independentes_teste, atributos_dependentes_teste, color = 'blue')
plt.plot(atributos_independentes_treino, regressor.predict(atributos_independentes_treino), color = 'gray')
plt.title('Salário e experiência - Teste')
plt.xlabel('Anos de experiência')
plt.ylabel('Salário')
plt.show()
