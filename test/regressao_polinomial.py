#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://www.leonardosapiras.com.br/op/Dados7.csv')
atributos_independentes = dataset.iloc[:, 1:2].values
atributos_dependentes = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(atributos_independentes, atributos_dependentes)



from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
atributos_independentes_poly = poly_reg.fit_transform(atributos_independentes)
poly_reg.fit(atributos_independentes_poly, atributos_dependentes)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(atributos_independentes_poly, atributos_dependentes)


# Visualização com regressão linear
plt.scatter(atributos_independentes, atributos_dependentes, color = 'blue')
plt.plot(atributos_independentes, lin_reg.predict(atributos_independentes), color = 'gray')
plt.title('Regressão linear')
plt.xlabel('Cargo')
plt.ylabel('Salário')
plt.show()

# Visualização com regressão polinomial
plt.scatter(atributos_independentes, atributos_dependentes, color = 'blue')
plt.plot(atributos_independentes, lin_reg_2.predict(poly_reg.fit_transform(atributos_independentes)), color = 'gray')
plt.title('Regressão polinomial')
plt.xlabel('Cargo')
plt.ylabel('Salário')
plt.show()