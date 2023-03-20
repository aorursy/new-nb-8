# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

ignored_columns = ['ID', 'TARGET']
C = df_train.columns

# remove constant columns
eps = 1e-10
dropped_columns = set()
print('Identifing low-variance columns...', end=' ')
for c in C:
    if df_train[c].var() < eps:
        # print('.. %-30s: too low variance ... column ignored'%(c))
        dropped_columns.add(c)
print('done!')
C = list(set(C) - dropped_columns - set(ignored_columns))

# remove duplicate columns
print('Identifying duplicate columns...', end=' ')
for i, c1 in enumerate(C):
    f1 = df_train[c1].values
    for j, c2 in enumerate(C[i+1:]):
        f2 = df_train[c2].values
        if np.all(f1 == f2):
            dropped_columns.add(c2)
print('done!')

C = list(set(C) - dropped_columns - set(ignored_columns))
print('# columns dropped: %d'%(len(dropped_columns)))
print('# columns retained: %d'%(len(C)))

df_train.drop(dropped_columns, axis=1, inplace=True)
df_test.drop(dropped_columns, axis=1, inplace=True)
y_learning = df_train['TARGET'].values
X_learning = df_train.drop(['ID','TARGET'], axis=1).values

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# Split the learning set into training, validation and local-test sets.
X_train_valid, X_local_test, y_train_valid, y_local_test = train_test_split(
    X_learning, y_learning, test_size=0.168, random_state=1
)
X_fit, X_eval, y_fit, y_eval= train_test_split(
    X_train_valid, y_train_valid, test_size=0.2, random_state=1
)

print('# train: %5d (0s: %5d, 1s: %4d)'%(len(y_fit), sum(y_fit==0), sum(y_fit==1)))
print('# valid: %5d (0s: %5d, 1s: %4d)'%(len(y_eval), sum(y_eval==0), sum(y_eval==1)))
print('# test:  %5d (0s: %5d, 1s: %4d)'%((len(y_local_test), sum(y_local_test==0), sum(y_local_test==1))))
# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5,
                        n_estimators=1000, learning_rate=0.01, 
                        subsample=0.5, colsample_bytree=0.9, seed=4242)

# fitting
clf.fit(X_fit, y_fit, early_stopping_rounds=50, eval_metric="auc", eval_set=[(X_eval, y_eval)])

# compute the AUC for the learnt model on training, validation, and local test data.
auc_train = roc_auc_score(y_fit, clf.predict_proba(X_fit)[:,1])
auc_valid = roc_auc_score(y_eval, clf.predict_proba(X_eval)[:,1])
auc_test  = roc_auc_score(y_local_test, clf.predict_proba(X_local_test)[:,1])

print('\n-----------------------')
print('  AUC train: %.5f'%auc_train)
print('  AUC valid: %.5f'%auc_valid)
print('  AUC test : %.5f'%auc_test)
print('-----------------------')

print('\nModel parameters...')
print(clf.get_params())
print('\n-----------------------\n')

# predicting
y_pred= clf.predict_proba(X_test)[:,1]

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)

print('Completed!')