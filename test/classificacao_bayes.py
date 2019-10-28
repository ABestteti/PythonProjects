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

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(atributos_independentes_treino, atributos_dependentes_treino)

# Realiza classificação
atributos_dependentes_classificados = classificador.predict(atributos_independentes_teste)


# Matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(atributos_dependentes_teste, atributos_dependentes_classificados)
cm

