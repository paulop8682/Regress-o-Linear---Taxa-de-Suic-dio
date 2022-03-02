#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importando as bibliotecas

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

import numpy as np

import seaborn as sns


# In[2]:


#Importando a base de dados
dados = pd.read_csv('master.csv', sep = ',')


# In[3]:


#Renomeando colunas
dados = dados.rename(columns = {'suicides/100k pop': 'taxa_suicidio', 'gdp_per_capita ($)': 'pib_per_capita'})


# In[4]:


#tamanho do dataframe
dados.shape


# In[5]:


#visão geral
dados.info()


# In[6]:


# verificando se há dados faltantes
dados.isna().any()


# In[7]:


#estatística descritiva
dados.describe().round(2)


# In[8]:


#Transforando em dummies
dados = dados.replace('male', 1)
dados = dados.replace('female', 0)


# In[9]:


# Grafico taxa_suicidio
fig, ax = plt.subplots(figsize=(14, 6))

sns.lineplot(data = dados, y= "taxa_suicidio", x = 'year', color = 'green')


# In[10]:


#Correlação das variáveis
fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(dados.corr(), annot = True, center=0,  cmap="rocket", vmin=-0.9, vmax=1)
ax.set(title="Tabela de Correlação Linear entre as variaveis em geral");


# In[23]:


# plotando um histograma para a variável 'suicídio'
ax = sns.distplot(dados['taxa_suicidio'])
# fazendo formatações
ax.figure.set_size_inches(12,6)
# adicionando um título ao nosso gráfico
ax.set_title('Distribuição de Frequências da Taxa de Suicídio', fontsize=20)
# identificando os eixos
ax.set_xlabel('Taxa de Suicídio', fontsize=14)
ax.set_ylabel('Frequência', fontsize=14)
ax


# In[12]:


dados.head()


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


# criando uma Series (pandas) para armazenar 
# a taxa de suicidio (y)
y = dados['taxa_suicidio']

X = dados[['sex', 'pib_per_capita']]


# In[15]:


#Separando em base de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)


# In[16]:


# importando LinearRegression e métricas
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[17]:


#Parametrizando
modelo = LinearRegression()


# In[18]:


# utilizando método fit() para estimar o modelo
modelo.fit(X_train, y_train)


# In[19]:


#Previsões
y_previsto = modelo.predict(X_test)
y_previsto


# In[21]:


#Métricas
mse = mean_squared_error(y_previsto, y_test, squared = False)

rmse =  mse **0.5

MAE  =  mean_absolute_error(y_test,y_previsto)

def adjusted_r2(y_test, y_pred,X_train):
    
  from sklearn.metrics import r2_score

  adj_r2 = (1 - ((1 - r2_score(y_test, y_pred)) * (len(y_test) - 1)) / 
          (len(y_test) - X_train.shape[1] - 1))
    
  return adj_r2

print(f"MÉTRICAS: rmse = {rmse.round(2)},", f"mse = {mse.round(2)},", f"MAE = {MAE.round(2)},", 
      f"R² = {modelo.score(X_train, y_train).round(2)},",
      f"R²-ajustado = {adjusted_r2(y_test, y_previsto,X_train).round(2)}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




