import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt


from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

pd.options.display.precision = 15



import lightgbm as lgb

import xgboost as xgb

import time

import datetime

from catboost import CatBoostRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from sklearn import metrics

from sklearn import linear_model

import gc

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



from IPython.display import HTML

import json

import altair as alt



import networkx as nx

import matplotlib.pyplot as plt


import os

import time

import datetime

import json

import gc

from numba import jit



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn import metrics



from itertools import product



# import altair as alt

# from altair.vega import v3

# from IPython.display import HTML

# alt.renderers.enable('notebook')
# using ideas from this kernel: https://www.kaggle.com/notslush/altair-visualization-2018-stackoverflow-survey

def prepare_altair():

    """

    Helper function to prepare altair for working.

    """

    vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + v3.SCHEMA_VERSION

    vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'

    vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION

    vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'

    noext = "?noext"

    

    paths = {

        'vega': vega_url + noext,

        'vega-lib': vega_lib_url + noext,

        'vega-lite': vega_lite_url + noext,

        'vega-embed': vega_embed_url + noext

    }

    

    workaround = f"""    requirejs.config({{

        baseUrl: 'https://cdn.jsdelivr.net/npm/',

        paths: {paths}

    }});

    """

    

    return workaround

    



def add_autoincrement(render_func):

    # Keep track of unique <div/> IDs

    cache = {}

    def wrapped(chart, id="vega-chart", autoincrement=True):

        if autoincrement:

            if id in cache:

                counter = 1 + cache[id]

                cache[id] = counter

            else:

                cache[id] = 0

            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])

        else:

            if id not in cache:

                cache[id] = 0

            actual_id = id

        return render_func(chart, id=actual_id)

    # Cache will stay outside and 

    return wrapped

           



@add_autoincrement

def render(chart, id="vega-chart"):

    """

    Helper function to plot altair visualizations.

    """

    chart_str = """

    <div id="{id}"></div><script>

    require(["vega-embed"], function(vg_embed) {{

        const spec = {chart};     

        vg_embed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);

        console.log("anything?");

    }});

    console.log("really...anything?");

    </script>

    """

    return HTML(

        chart_str.format(

            id=id,

            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)

        )

    )

    



def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df

    



@jit

def fast_auc(y_true, y_prob):

    """

    fast roc_auc computation: https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013

    """

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    nfalse = 0

    auc = 0

    n = len(y_true)

    for i in range(n):

        y_i = y_true[i]

        nfalse += (1 - y_i)

        auc += y_i * nfalse

    auc /= (nfalse * (n - nfalse))

    return auc





def eval_auc(y_true, y_pred):

    """

    Fast auc eval function for lgb.

    """

    return 'auc', fast_auc(y_true, y_pred), True





def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):

    """

    Fast metric computation for this competition: https://www.kaggle.com/c/champs-scalar-coupling

    Code is from this kernel: https://www.kaggle.com/uberkinder/efficient-metric

    """

    maes = (y_true-y_pred).abs().groupby(types).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()

    



def train_model_regression(X, X_test, y, params, folds, model_type='lgb', eval_metric='mae', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of regression models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns is None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'sklearn_scoring_function': metrics.mean_absolute_error},

                    'group_mae': {'lgb_metric_name': 'mae',

                        'catboost_metric_name': 'MAE',

                        'scoring_function': group_mean_log_mae},

                    'mse': {'lgb_metric_name': 'mse',

                        'catboost_metric_name': 'MSE',

                        'sklearn_scoring_function': metrics.mean_squared_error}

                    }



    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros(len(X))

    

    # averaged predictions on train data

    prediction = np.zeros(len(X_test))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        if eval_metric != 'group_mae':

            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))

        else:

            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict

    





def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,

                               verbose=10000, early_stopping_rounds=200, n_estimators=50000):

    """

    A function to train a variety of regression models.

    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)

    :params: y - target

    :params: folds - folds to split data

    :params: model_type - type of model to use

    :params: eval_metric - metric to use

    :params: columns - columns to use. If None - use all columns

    :params: plot_feature_importance - whether to plot feature importance of LGB

    :params: model - sklearn model, works only for "sklearn" model type

    

    """

    columns = X.columns if columns == None else columns

    X_test = X_test[columns]

    

    # to set up scoring parameters

    metrics_dict = {'auc': {'lgb_metric_name': eval_auc,

                        'catboost_metric_name': 'AUC',

                        'sklearn_scoring_function': metrics.roc_auc_score},

                    }

    

    result_dict = {}

    

    # out-of-fold predictions on train data

    oof = np.zeros((len(X), len(set(y.values))))

    

    # averaged predictions on train data

    prediction = np.zeros((len(X_test), oof.shape[1]))

    

    # list of scores on folds

    scores = []

    feature_importance = pd.DataFrame()

    

    # split and train on folds

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print(f'Fold {fold_n + 1} started at {time.ctime()}')

        if type(X) == np.ndarray:

            X_train, X_valid = X[columns][train_index], X[columns][valid_index]

            y_train, y_valid = y[train_index], y[valid_index]

        else:

            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]

            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

            

        if model_type == 'lgb':

            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],

                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)

            

            y_pred_valid = model.predict_proba(X_valid)

            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=n_estimators, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)

            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.')

            print('')

            

            y_pred = model.predict_proba(X_test)

        

        if model_type == 'cat':

            model = CatBoostClassifier(iterations=n_estimators, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'], **params,

                                      loss_function=metrics_dict[eval_metric]['catboost_metric_name'])

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid

        scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid[:, 1]))



        prediction += y_pred    

        

        if model_type == 'lgb' and plot_feature_importance:

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= folds.n_splits

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    result_dict['oof'] = oof

    result_dict['prediction'] = prediction

    result_dict['scores'] = scores

    

    if model_type == 'lgb':

        if plot_feature_importance:

            feature_importance["importance"] /= folds.n_splits

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

            

            result_dict['feature_importance'] = feature_importance

        

    return result_dict



# setting up altair

# workaround = prepare_altair()

# HTML("".join((

#     "<script>",

#     workaround,

#     "</script>",

# )))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')

structures = pd.read_csv('../input/structures.csv')

scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')



print('Train dataset shape is -> rows: {} cols:{}'.format(train.shape[0],train.shape[1]))

print('Test dataset shape is  -> rows: {} cols:{}'.format(test.shape[0],test.shape[1]))

print('Sub dataset shape is  -> rows: {} cols:{}'.format(sub.shape[0],sub.shape[1]))

print('Structures dataset shape is  -> rows: {} cols:{}'.format(structures.shape[0],structures.shape[1]))

print('Scalar_coupling_contributions dataset shape is  -> rows: {} cols:{}'.format(scalar_coupling_contributions.shape[0],

                                                                                   scalar_coupling_contributions.shape[1]))
n_estimators_default = 4000
'''

size = round(0.10*train.shape[0])

train = train[:size]

test = test[:size]

sub = sub[:size]

structures = structures[:size]

scalar_coupling_contributions = scalar_coupling_contributions[:size]



print('Train dataset shape is now rows: {} cols:{}'.format(train.shape[0],train.shape[1]))

print('Test dataset shape is now rows: {} cols:{}'.format(test.shape[0],test.shape[1]))

print('Sub dataset shape is now rows: {} cols:{}'.format(sub.shape[0],sub.shape[1]))

print('Structures dataset shape is now rows: {} cols:{}'.format(structures.shape[0],structures.shape[1]))

print('Scalar_coupling_contributions dataset shape is now rows: {} cols:{}'.format(scalar_coupling_contributions.shape[0],

                                                                                   scalar_coupling_contributions.shape[1]))

'''
train = pd.merge(train, scalar_coupling_contributions, how = 'left',

                  left_on  = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'],

                  right_on = ['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
train.head(10)
test.head(10)
scalar_coupling_contributions.head(5)
pd.concat(objs=[train['scalar_coupling_constant'],scalar_coupling_contributions['fc'] ],axis=1)[:10]
from tqdm import tqdm_notebook as tqdm

atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor



fudge_factor = 0.05

atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}

print(atomic_radius)



electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}



#structures = pd.read_csv(structures, dtype={'atom_index':np.int8})



atoms = structures['atom'].values

atoms_en = [electronegativity[x] for x in tqdm(atoms)]

atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]



structures['EN'] = atoms_en

structures['rad'] = atoms_rad



display(structures.head())
i_atom = structures['atom_index'].values

p = structures[['x', 'y', 'z']].values

p_compare = p

m = structures['molecule_name'].values

m_compare = m

r = structures['rad'].values

r_compare = r



source_row = np.arange(len(structures))

max_atoms = 28



bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)

bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)



print('Calculating bonds')



for i in tqdm(range(max_atoms-1)):

    p_compare = np.roll(p_compare, -1, axis=0)

    m_compare = np.roll(m_compare, -1, axis=0)

    r_compare = np.roll(r_compare, -1, axis=0)

    

    mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?

    dists = np.linalg.norm(p - p_compare, axis=1) * mask

    r_bond = r + r_compare

    

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    

    source_row = source_row

    target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row

    

    source_atom = i_atom

    target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i

    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col

    

    bonds[(source_row, target_atom)] = bond

    bonds[(target_row, source_atom)] = bond

    bond_dists[(source_row, target_atom)] = dists

    bond_dists[(target_row, source_atom)] = dists



bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row

bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col

bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row

bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col



print('Counting and condensing bonds')



bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]

bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]

bond_lengths_mean = [ np.mean(x) for x in bond_lengths]

bond_lengths_median = [ np.median(x) for x in bond_lengths]

bond_lengths_std = [ np.std(x) for x in bond_lengths]

n_bonds = [len(x) for x in bonds_numeric]



#bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}

#bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})



bond_data = {'n_bonds':n_bonds, 'bond_lengths_mean': bond_lengths_mean,

             'bond_lengths_std':bond_lengths_std, 'bond_lengths_median': bond_lengths_median }

bond_df = pd.DataFrame(bond_data)

structures = structures.join(bond_df)

display(structures.head(20))
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    #df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

train['dist'] = 1/(train['dist']**3)

test['dist'] = 1/(test['dist']**3)



train['dist_x'] = (train['x_0'] - train['x_1']) ** 2

test['dist_x'] = (test['x_0'] - test['x_1']) ** 2

train['dist_y'] = (train['y_0'] - train['y_1']) ** 2

test['dist_y'] = (test['y_0'] - test['y_1']) ** 2

train['dist_z'] = (train['z_0'] - train['z_1']) ** 2

test['dist_z'] = (test['z_0'] - test['z_1']) ** 2



train['type_0'] = train['type'].apply(lambda x: x[0])

test['type_0'] = test['type'].apply(lambda x: x[0])

def create_features(df):

    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')

    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')

    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')

    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')

    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')

    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')

    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')

    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']

    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')

    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']

    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')

    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')

    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')

    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']

    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']

    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')

    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']

    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']

    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']

    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']

    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')

    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']

    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']

    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')

    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']

    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']

    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')

    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']

    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']

    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')

    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']

    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']

    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')

    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']

    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']

    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')

    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')

    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']

    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']

    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')

    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']

    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')

    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']

    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')

    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']

    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']

    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')

    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')

    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')

    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    df = reduce_mem_usage(df)

    return df



train = create_features(train)

test = create_features(test)
def map_atom_info(df_1,df_2, atom_idx):

    df = pd.merge(df_1, df_2, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)



    return df



def create_closest(df_train):

    #I apologize for my poor coding skill. Please make the better one.

    df_temp=df_train.loc[:,["molecule_name","atom_index_0","atom_index_1","dist","x_0","y_0","z_0","x_1","y_1","z_1"]].copy()

    df_temp_=df_temp.copy()

    df_temp_= df_temp_.rename(columns={'atom_index_0': 'atom_index_1',

                                       'atom_index_1': 'atom_index_0',

                                       'x_0': 'x_1',

                                       'y_0': 'y_1',

                                       'z_0': 'z_1',

                                       'x_1': 'x_0',

                                       'y_1': 'y_0',

                                       'z_1': 'z_0'})

    df_temp=pd.concat(objs=[df_temp,df_temp_],axis=0)



    df_temp["min_distance"]=df_temp.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')

    df_temp= df_temp[df_temp["min_distance"]==df_temp["dist"]]



    df_temp=df_temp.drop(['x_0','y_0','z_0','min_distance'], axis=1)

    df_temp= df_temp.rename(columns={'atom_index_0': 'atom_index',

                                     'atom_index_1': 'atom_index_closest',

                                     'distance': 'distance_closest',

                                     'x_1': 'x_closest',

                                     'y_1': 'y_closest',

                                     'z_1': 'z_closest'})



    for atom_idx in [0,1]:

        df_train = map_atom_info(df_train,df_temp, atom_idx)

        df_train = df_train.rename(columns={'atom_index_closest': f'atom_index_closest_{atom_idx}',

                                            'distance_closest': f'distance_closest_{atom_idx}',

                                            'x_closest': f'x_closest_{atom_idx}',

                                            'y_closest': f'y_closest_{atom_idx}',

                                            'z_closest': f'z_closest_{atom_idx}'})

    return df_train



#dtrain = create_closest(train)

#dtest = create_closest(test)

#print('dtrain size',dtrain.shape)

#print('dtest size',dtest.shape)
def add_cos_features(df):

    df["distance_0"]=((df['x_0']-df['x_closest_0'])**2+(df['y_0']-df['y_closest_0'])**2+(df['z_0']-df['z_closest_0'])**2)**(1/2)

    df["distance_1"]=((df['x_1']-df['x_closest_1'])**2+(df['y_1']-df['y_closest_1'])**2+(df['z_1']-df['z_closest_1'])**2)**(1/2)

    df["vec_0_x"]=(df['x_0']-df['x_closest_0'])/df["distance_0"]

    df["vec_0_y"]=(df['y_0']-df['y_closest_0'])/df["distance_0"]

    df["vec_0_z"]=(df['z_0']-df['z_closest_0'])/df["distance_0"]

    df["vec_1_x"]=(df['x_1']-df['x_closest_1'])/df["distance_1"]

    df["vec_1_y"]=(df['y_1']-df['y_closest_1'])/df["distance_1"]

    df["vec_1_z"]=(df['z_1']-df['z_closest_1'])/df["distance_1"]

    df["vec_x"]=(df['x_1']-df['x_0'])/df["dist"]

    df["vec_y"]=(df['y_1']-df['y_0'])/df["dist"]

    df["vec_z"]=(df['z_1']-df['z_0'])/df["dist"]

    df["cos_0_1"]=df["vec_0_x"]*df["vec_1_x"]+df["vec_0_y"]*df["vec_1_y"]+df["vec_0_z"]*df["vec_1_z"]

    df["cos_0"]=df["vec_0_x"]*df["vec_x"]+df["vec_0_y"]*df["vec_y"]+df["vec_0_z"]*df["vec_z"]

    df["cos_1"]=df["vec_1_x"]*df["vec_x"]+df["vec_1_y"]*df["vec_y"]+df["vec_1_z"]*df["vec_z"]

    df=df.drop(['vec_0_x','vec_0_y','vec_0_z','vec_1_x','vec_1_y','vec_1_z','vec_x','vec_y','vec_z'], axis=1)

    return df

    

#train = add_cos_features(train)

#test = add_cos_features(test)



#print('train size',train.shape)

#print('test size',test.shape)
del_cols_list = ['id','molecule_name','sd','pso','dso']

def del_cols(df, cols):

    del_cols_list_ = [l for l in del_cols_list if l in df]

    df = df.drop(del_cols_list_,axis=1)

    return df



train = del_cols(train,del_cols_list)

test = del_cols(test,del_cols_list)
def encode_categoric_single(df):

    lbl = LabelEncoder()

    cat_cols=[]

    try:

        cat_cols = df.describe(include=['O']).columns.tolist()

        for cat in cat_cols:

            df[cat] = lbl.fit_transform(list(df[cat].values))

    except Exception as e:

        print('error: ', str(e) )



    return df
def encode_categoric(dtrain,dtest):

    lbl = LabelEncoder()

    objs_n = len(dtrain)

    dfmerge = pd.concat(objs=[dtrain,dtest],axis=0)

    cat_cols=[]

    try:

        cat_cols = dfmerge.describe(include=['O']).columns.tolist()

        for cat in cat_cols:

            dfmerge[cat] = lbl.fit_transform(list(dfmerge[cat].values))

    except Exception as e:

        print('error: ', str(e) )



    dtrain = dfmerge[:objs_n]

    dtest = dfmerge[objs_n:]

    return dtrain,dtest



train = encode_categoric_single(train)

test = encode_categoric_single(test)
y_fc = train['fc']

X = train.drop(['scalar_coupling_constant','fc'],axis=1)

y = train['scalar_coupling_constant']



X_test = test.copy()
print('X size',X.shape)

print('X_test size',X_test.shape)

print('dtest size',test.shape)

print('y_fc size',y_fc.shape)



del train, test

gc.collect()

good_columns = ['type',

 'bond_lengths_mean_y',

 'bond_lengths_median_y',

 'bond_lengths_std_y',

 'bond_lengths_mean_x',

 'molecule_atom_index_0_dist_min_div',

 'molecule_atom_index_0_dist_std_div',

 'molecule_atom_index_0_dist_mean',

 'molecule_atom_index_0_dist_max',

 'dist_y',

 'molecule_atom_index_1_dist_std_diff',

 'z_0',

 'molecule_type_dist_min',

 'molecule_atom_index_0_y_1_mean_div',

 'dist_x',

 'x_0',

 'y_0',

 'molecule_type_dist_std',

 'molecule_atom_index_0_y_1_std',

 'molecule_dist_mean',

 'molecule_atom_index_0_dist_std_diff',

 'dist_z',

 'molecule_atom_index_0_dist_std',

 'molecule_atom_index_0_x_1_std',

 'molecule_type_dist_std_diff',

 'molecule_type_0_dist_std',

 'dist',

 'molecule_atom_index_0_dist_mean_diff',

 'molecule_atom_index_1_dist_min_div',

 'molecule_atom_index_1_dist_mean_diff',

 'y_1',

 'molecule_type_dist_mean_div',

 'molecule_dist_max',

 'molecule_atom_index_0_dist_mean_div',

 'z_1',

 'molecule_atom_index_0_z_1_std',

 'molecule_atom_index_1_dist_mean_div',

 'molecule_atom_index_1_dist_min_diff',

 'molecule_atom_index_1_dist_mean',

 'molecule_atom_index_1_dist_min',

 'molecule_atom_index_1_dist_max',

 'molecule_type_0_dist_std_diff',

 'molecule_atom_index_0_dist_min_diff',

 'molecule_type_dist_mean_diff',

 'x_1',

 'molecule_atom_index_0_y_1_max',

 'molecule_atom_index_0_y_1_mean_diff',

 'molecule_atom_1_dist_std_diff',

 'molecule_atom_index_0_y_1_mean',

 'molecule_atom_1_dist_std',

 'molecule_type_dist_max']



X = X[good_columns].copy()

X_test = X_test[good_columns].copy()
n_fold = 8

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
params = {'num_leaves': 50,

          'min_child_samples': 79,

          'min_data_in_leaf': 100,

          'objective': 'regression',

          'max_depth': 9,

          'learning_rate': 0.3,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3,

          'colsample_bytree': 1.0

         }
X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values, 'oof': [0] * len(X), 'target': y_fc.values})

X_short_test = pd.DataFrame({'ind': list(X_test.index), 'type': X_test['type'].values, 'prediction': [0] * len(X_test)})

result_dict_lgb_oof =  {}

for t in X['type'].unique():

    print(f'Training of type {t}')

    X_t = X.loc[X['type'] == t]

    X_test_t = X_test.loc[X_test['type'] == t]

    y_t = X_short.loc[X_short['type'] == t, 'target']

    result_dict_lgb_oof = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=False,

                                                      verbose=500, early_stopping_rounds=200, n_estimators=n_estimators_default)

    X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb_oof['oof']

    X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb_oof['prediction']
X['oof_fc'] = X_short['oof']

X_test['oof_fc'] = X_short_test['prediction']


good_columns = ['type','oof_fc',

 'bond_lengths_mean_y',

 'bond_lengths_median_y',

 'bond_lengths_std_y',

 'bond_lengths_mean_x',

 'molecule_atom_index_0_dist_min_div',

 'molecule_atom_index_0_dist_std_div',

 'molecule_atom_index_0_dist_mean',

 'molecule_atom_index_0_dist_max',

 'dist_y',

 'molecule_atom_index_1_dist_std_diff',

 'z_0',

 'molecule_type_dist_min',

 'molecule_atom_index_0_y_1_mean_div',

 'dist_x',

 'x_0',

 'y_0',

 'molecule_type_dist_std',

 'molecule_atom_index_0_y_1_std',

 'molecule_dist_mean',

 'molecule_atom_index_0_dist_std_diff',

 'dist_z',

 'molecule_atom_index_0_dist_std',

 'molecule_atom_index_0_x_1_std',

 'molecule_type_dist_std_diff',

 'molecule_type_0_dist_std',

 'dist',

 'molecule_atom_index_0_dist_mean_diff',

 'molecule_atom_index_1_dist_min_div',

 'molecule_atom_index_1_dist_mean_diff',

 'y_1',

 'molecule_type_dist_mean_div',

 'molecule_dist_max',

 'molecule_atom_index_0_dist_mean_div',

 'z_1',

 'molecule_atom_index_0_z_1_std',

 'molecule_atom_index_1_dist_mean_div',

 'molecule_atom_index_1_dist_min_diff',

 'molecule_atom_index_1_dist_mean',

 'molecule_atom_index_1_dist_min',

 'molecule_atom_index_1_dist_max',

 'molecule_type_0_dist_std_diff',

 'molecule_atom_index_0_dist_min_diff',

 'molecule_type_dist_mean_diff',

 'x_1',

 'molecule_atom_index_0_y_1_max',

 'molecule_atom_index_0_y_1_mean_diff',

 'molecule_atom_1_dist_std_diff',

 'molecule_atom_index_0_y_1_mean',

 'molecule_atom_1_dist_std',

 'molecule_type_dist_max']



X = X[good_columns].copy()

X_test = X_test[good_columns].copy()
params = {'num_leaves': 50,

          'min_child_samples': 79,

          'min_data_in_leaf': 100,

          'objective': 'regression',

          'max_depth': 9,

          'learning_rate': 0.3,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.9,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3,

          'colsample_bytree': 1.0

         }

#result_dict_lgb2 = train_model_regression(X=X, X_test=X_test, y=y, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=True,

#                                                      verbose=500, early_stopping_rounds=200, n_estimators=n_estimators_default)

#Best Features? 

''' 

feature_importance = result_dict_lgb2['feature_importance']

best_features = feature_importance[['feature','importance']].groupby(['feature']).mean().sort_values(

        by='importance',ascending=False).iloc[:50,0:0].index.tolist()

best_features'''
X_short = pd.DataFrame({'ind': list(X.index), 'type': X['type'].values, 'oof': [0] * len(X), 'target': y.values})

X_short_test = pd.DataFrame({'ind': list(X_test.index), 'type': X_test['type'].values, 'prediction': [0] * len(X_test)})

for t in X['type'].unique():

    print(f'Training of type {t}')

    X_t = X.loc[X['type'] == t]

    X_test_t = X_test.loc[X_test['type'] == t]

    y_t = X_short.loc[X_short['type'] == t, 'target']

    result_dict_lgb3 = train_model_regression(X=X_t, X_test=X_test_t, y=y_t, params=params, folds=folds, model_type='lgb', eval_metric='group_mae', plot_feature_importance=False,

                                                      verbose=500, early_stopping_rounds=200, n_estimators=n_estimators_default)

    X_short.loc[X_short['type'] == t, 'oof'] = result_dict_lgb3['oof']

    X_short_test.loc[X_short_test['type'] == t, 'prediction'] = result_dict_lgb3['prediction']
#Training models for type

sub['scalar_coupling_constant'] = X_short_test['prediction']

sub.to_csv('submission_type.csv', index=False)

sub.head()
