#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Bibliotecas
import pandas

# Extração de dados
dataset = pandas.read_csv('Dados.csv')
atributos_independentes = dataset.iloc[:, :-1].values
atributos_dependentes = dataset.iloc[:, 3].values



#Quantidade de linhas e colunas do DataFrame
dataset.shape


#Descrição do Index
dataset.index 


#Colunas presentes no DataFrame
dataset.columns 



#Contagem de dados não-nulos
dataset.count()



#Soma dos valores de um DataFrame
dataset.sum()


#Filtros em dataframe
dataset["Salario"].sum()
dataset[dataset["Pais"] == "Brasil"]["Salario"].sum()

dataset.query('Pais == "Brasil"')
dataset.query('Idade == 44')

dataset.query('Pais == "Uruguai" and Idade == 44')



#Menor valor de um DataFrame
dataset.min()


#Maior valor
dataset.max()


#Resumo estatístico do DataFrame, com quartis, mediana, desvio padrão.
dataset.describe()


#Média dos valores
dataset.mean()
dataset[dataset["Pais"] == "Brasil"]["Salario"].mean()


#Mediana dos valores
dataset.median()


print(dataset["Pais"].value_counts())
dataset["Pais"].value_counts().plot.bar()




