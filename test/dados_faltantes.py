#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Pré-processamento

# Bibliotecas
import pandas
import numpy

# Extração de dados
dataset = pandas.read_csv('Dados.csv')

atributos_independentes = dataset.iloc[:, :-1].values
atributos_dependentes = dataset.iloc[:, 3].values

# Dados faltantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = numpy.nan, strategy = 'mean')
imputer = imputer.fit(atributos_independentes[:, 1:3])
atributos_independentes[:, 1:3] = imputer.transform(atributos_independentes[:, 1:3])

print(atributos_independentes)