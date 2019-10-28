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



# Tratamento de dados categoricos
# Atributos independentes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_atributos_independentes = LabelEncoder()
atributos_independentes[:, 0] = labelencoder_atributos_independentes.fit_transform(atributos_independentes[:, 0])

# Previne que uma categoria seja maior que outra, por meio de dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0])
atributos_independentes = onehotencoder.fit_transform(atributos_independentes).toarray()


# Atributos dependentes
labelencoder_atributos_dependentes = LabelEncoder()
atributos_dependentes = labelencoder_atributos_dependentes.fit_transform(atributos_dependentes)