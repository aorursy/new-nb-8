#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




import pandas as pd




sample_submission = pd.read_csv('/kaggle/input/dont-call-me-turkey/sample_submission.csv')
sample_submission.head(15)




train= pd.read_json('/kaggle/input/dont-call-me-turkey/train.json')
test = pd.read_json('/kaggle/input/dont-call-me-turkey/test.json')




train.head()




test.head()




test.shape




# Primero vamos a dejar nuesto data set train con las características 
# audio_embeding

train_clean = train.iloc[:,:2]
train_clean.head()




X = train_clean.iloc[:,0]
y = train_clean.iloc[:,1]




len(X)




np.array(X[34]).shape




#len(np.array(X.iloc[0]).flatten())
X_flatten = []
for idx in range(len(X)):
    X_flatten.append(list(np.array(X.iloc[idx]).flatten()))
#X_flatten = np.array(X_flatten)




y = np.array(y)




unicos = []
for idx in range(len(X_flatten)):
    unicos.append(np.array(X_flatten[idx]).shape[0])




np.unique(unicos, return_counts=True)




X_flatten2 = [] #array donde vamos a poner los que tengan los 1280
y_flatten2 = [] #tiene que tener la misma cantidad de elementos que X_flatten2
for idx in range(len(X_flatten)):
    if np.array(X_flatten[idx]).shape[0]==1280:
        X_flatten2.append(np.array(X_flatten[idx]))
        y_flatten2.append(y[idx])




X_flatten2 #Datos preprocesados, NO DESAJUSTAR




y_flatten2 #Datos preprocesados, NO DESAJUSTAR




np.array(X_flatten2).shape




np.array(y_flatten2).shape




np.unique(y, return_counts=True)
percent_turke = np.unique(y, return_counts=True)[1][1]/np.unique(y, return_counts=True)[1].sum()*100
percent_turke_new = np.unique(y_flatten2, return_counts=True)[1][1]/np.unique(y_flatten2, return_counts=True)[1].sum()*100

print('Porcentaje de guajolotes antes %.2f  \nPorcentaje de guajolotes ahora  %.2f' 
      % (percent_turke, percent_turke_new))
print('El valor bajo %.2f' %(percent_turke-percent_turke_new))




from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score,f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(X_flatten2, y_flatten2,
                                                    test_size=0.20, random_state=25)




lr = LogisticRegression()




lr.fit(X_train,y_train)




y_pred = lr.predict(X_test)




print('Test Accuracy: %.3f' %lr.score(X_test,y_test))




get_ipython().run_line_magic('matplotlib', 'inline')




import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve




train_sizes, train_scores, test_scores = learning_curve(estimator= lr,
                                                       X = X_train,
                                                       y = y_train,
                                                       train_sizes= np.linspace(0.1,1.0,10),
                                                       cv = 10,
                                                       n_jobs=1)




train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)




plt.plot(train_sizes, train_mean, c = 'b', marker = 'o',
        markersize = 5, label = 'Training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,
                alpha= 0.15, color = 'b')
plt.plot(train_sizes, test_mean, color = 'g', linestyle = '--',
        marker = 's', markersize = 5, label = 'validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std,
                test_mean - test_std, alpha = 0.15, color = 'g')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.ylim([0.85,1.05])
plt.show()




from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler #estandarizar nuestros valores
from sklearn.decomposition import PCA


#param_range = [0.001,0.01, 0.1, 1.0, 10.0,100.0,1000.0,1e4,1e5,1e6,1e7]
param_range = np.logspace(-4,5, num = 10)




pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression()) #Hacemos nuestro pipeline




train_scores, test_scores = validation_curve(estimator=pipe_lr,
                                             X=X_train,
                                             y=y_train,
                                             param_name='logisticregression__C',
                                             param_range=param_range,
                                             cv=10)




train_mean = np.mean(train_scores, axis = 1)
train_std =  np.std(train_scores, axis = 1)

test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)




plt.plot(param_range, train_mean,
        color = 'b', marker = 'o',
        markersize = 5, label = 'training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha = 0.15,
                color = 'b')

plt.plot(param_range, test_mean, color = 'g',
        linestyle = '--', marker = 's',
        markersize = 5, label = 'validation accuracy')

plt.fill_between(param_range, test_mean + test_std,
                test_mean - test_std, alpha = 0.15,
                color = 'g')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.9, 1.02])
plt.show()




y_pred = pipe_lr.fit(X_train,y_train)




print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))




pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(C = 1e-3)) #Hacemos nuestro pipeline




pipe_lr.fit(X_train,y_train)




prueba = test.iloc[:,:2]
audio_emb = prueba.iloc[:,0] #Columna audio_embedding del conjunto prueba
etiquetas = prueba.iloc[:,1]
arreglo_prueba = []
for idx in range(len(audio_emb)):
#    pass
    arreglo_prueba.append(list(np.array(audio_emb.iloc[idx]).flatten()))
#arreglo_prueba = [idx for idx in list(np.array(audio_emb.iloc[idx]).flatten())]





unicos = []
for idx in range(len(arreglo_prueba)):
    unicos.append(np.array(arreglo_prueba[idx]).shape[0])




np.unique(unicos, return_counts=True)




for i in range(len(arreglo_prueba)):
    if len(np.array(arreglo_prueba)[i]) < 1280:
        print(i)
        diferencia = 1280-len(np.array(arreglo_prueba)[i])
        arreglo_prueba[i]=np.concatenate((np.array(arreglo_prueba)[i],
                                          np.zeros(diferencia)), axis = None) #juntamos los
        #arreglos que no lleguen a 1280 con arreglos de ceros
arreglo_prueba= np.array(arreglo_prueba)




for i in zip(arreglo_prueba, etiquetas):
    print(i)




predicciones1 = pd.DataFrame(zip(etiquetas,pipe_lr.predict(arreglo_prueba)))




predicciones1.to_csv('predicciones.csv')




arreglo






