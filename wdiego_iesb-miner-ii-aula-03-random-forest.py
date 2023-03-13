# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Carregar os dados

df = pd.read_csv('../input/train.csv', parse_dates=[0])

test = pd.read_csv('../input/test.csv', parse_dates=[0])
df.shape, test.shape
# Transformando a coluna count

df['count'] = np.log(df['count'])
# Pre-processamento dos dados



# Juntar os dataframes

df = df.append(test, sort=True)
df.info()
# Transformação da coluna datetime (feature engineering)

df['year'] = df['datetime'].dt.year

df['month'] = df['datetime'].dt.month

df['day'] = df['datetime'].dt.day

df['dayofweek'] = df['datetime'].dt.dayofweek

df['hour'] = df['datetime'].dt.hour
# Ordenar os dados pela coluna datetime

df.sort_values('datetime', inplace=True)
# Criando a coluna rolling_temp

df['rolling_temp'] = df['temp'].rolling(4, min_periods=1).mean()
# Criando a coluna rolling_atemp

df['rolling_atemp'] = df['atemp'].rolling(4, min_periods=1).mean()
# Separando os dataframes

test = df[df['count'].isnull()]

df = df[~df['count'].isnull()]
df.shape, test.shape
# Separando o df em treino e validação

from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape
# Selecionar as colunas a serem usadas no trainamento e validação



# Lista das colunas não usadas

removed_cols = ['count', 'casual', 'registered', 'datetime']



# Lista das features

feats = [c for c in df.columns if c not in removed_cols]
# Usar o modelo de RandomForest



# Importar o modelo

from sklearn.ensemble import RandomForestRegressor
# Instanciar o modelo

rf = RandomForestRegressor(random_state=42)
# Treinar o modelo

rf.fit(train[feats], train['count'])
# Fazendo as previsões

preds = rf.predict(valid[feats])
# Analisar as previsões com base na métrica



# Importando a métrica

from sklearn.metrics import mean_squared_error
# Validando as previsões

mean_squared_error(valid['count'], preds) ** (1/2)
# Melhorando o modelo de RandomForest

rf = RandomForestRegressor(random_state=42, n_estimators=200, n_jobs=-1)
# Treinar o modelo

rf.fit(train[feats], train['count'])
# Fazendo as previsões

preds = rf.predict(valid[feats])
# Validando as previsões

mean_squared_error(valid['count'], preds) ** (1/2)
# Preparando os dados para o kaggle



# Criando as previsões para os dados de teste

preds_test = rf.predict(test[feats])
# Adicionar as previsões ao dataframe

test['count'] = np.exp(preds_test)
# Salvando o arquivo para o Kaggle

test[['datetime', 'count']].to_csv('rf.csv', index=False)