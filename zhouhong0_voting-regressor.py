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
from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

import xgboost as xgb

from sklearn.ensemble import VotingRegressor

from catboost import CatBoostRegressor
train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")

test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")
all_zero_columns=[i for i in train.columns if train[i].nunique()==1]

train=train[[i for i in train.columns if i not in all_zero_columns]]

test=test[[i for i in test.columns if i not in all_zero_columns]]

duplicte_columns=['d60ddde1b', 'acc5b709d', '912836770', 'f8d75792f', 'f333a5f60']

train=train[[i for i in train.columns if i not in duplicte_columns]]

test=test[[i for i in test.columns if i not in duplicte_columns]]

X = np.log1p(train.drop(["ID", "target"], axis=1))

y = np.log1p(train["target"].values)

test = np.log1p(test.drop(["ID"], axis=1))
lgbbest=lgb.LGBMRegressor(bagging_fraction=0.5, boosting_type='gbdt', class_weight=None,

              colsample_bytree=1.0, feature_fraction=0.5,

              importance_type='split', learning_rate=0.01, max_depth=-1,

              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

              n_estimators=500, n_jobs=-1, num_leaves=130,

              objective='regression', random_state=42, reg_alpha=0.0,

              reg_lambda=0.0, silent=True, subsample=1.0,

              subsample_for_bin=200000, subsample_freq=0)
xgbbest=xgb.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.9, gamma=0.3,

             importance_type='gain', learning_rate=0.02, max_delta_step=0,

             max_depth=5, min_child_weight=5, missing=0, n_estimators=500,

             n_job=4, n_jobs=1, nthread=None, objective='reg:squarederror',

             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

             seed=0, silent=None, subsample=0.7, verbosity=1)
cb_model = CatBoostRegressor(iterations=500,

                             learning_rate=0.05,

                             depth=10,

                             eval_metric='RMSE',

                             random_seed = 42,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 50,

                             od_wait=20)
softvoting = VotingRegressor(estimators=[('xgb', xgbbest),('lgb', lgbbest),('cat', cb_model)])
softvoting.fit(X, y)

y_pred = softvoting.predict(test)

sub = pd.read_csv('../input/santander-value-prediction-challenge/sample_submission.csv')

sub["target"] = np.expm1(y_pred)

sub.to_csv('submit_softvoting.csv', index=False)