__author__ = 'Tilii: https://kaggle.com/tilii7' 



import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))





train_df = pd.read_csv('../input/train.csv', dtype={'id': np.int32, 'target': np.int8})

Y = train_df['target'].values

X = train_df.drop(['target', 'id'], axis=1)

test_df = pd.read_csv('../input/test.csv', dtype={'id': np.int32})

test = test_df.drop(['id'], axis=1)

# A parameter grid for XGBoost

params = {

        'min_child_weight': [1, 5, 10],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5]

        }

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',

                    silent=True, nthread=1)

folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3, random_state=1001 )



# Here we go

start_time = timer(None) # timing starts from this point for "start_time" variable

random_search.fit(X, Y)

timer(start_time) # timing ends here for "start_time" variable

print('\n All results:')

print(random_search.cv_results_)

print('\n Best estimator:')

print(random_search.best_estimator_)

print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))

print(random_search.best_score_ * 2 - 1)

print('\n Best hyperparameters:')

print(random_search.best_params_)

results = pd.DataFrame(random_search.cv_results_)

results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

y_test = random_search.predict_proba(test)

results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})

results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)

# grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='roc_auc', n_jobs=4, cv=skf.split(X,Y), verbose=3 )

# grid.fit(X, Y)

# print('\n All results:')

# print(grid.cv_results_)

# print('\n Best estimator:')

# print(grid.best_estimator_)

# print('\n Best score:')

# print(grid.best_score_ * 2 - 1)

# print('\n Best parameters:')

# print(grid.best_params_)

# results = pd.DataFrame(grid.cv_results_)

# results.to_csv('xgb-grid-search-results-01.csv', index=False)



# y_test = grid.best_estimator_.predict_proba(test)

# results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})

# results_df.to_csv('submission-grid-search-xgb-porto-01.csv', index=False)
