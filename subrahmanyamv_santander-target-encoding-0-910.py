import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm



import os

print(os.listdir("../input"))

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

features = [c for c in train_df.columns if c not in ['ID_code', 'target']] #basic features

target = train_df['target']
def t_encoding(tr,va,by):

    df = tr.groupby(by).agg({'target':['sum','count']})

    cols = ['sum_y','count_y']

    df.columns = cols

    df = df.reset_index()

    df = df.sort_values(by)



    df['r'] = df['sum_y'].cumsum() / df['count_y'].cumsum()  # I think this is the key operation

    df.drop(['sum_y','count_y'],axis=1,inplace=True)

    return va.merge(df,on=by,how='left')['r'].values



var_cols = features



for col in tqdm(var_cols):

    te_r = t_encoding(train_df,test_df,col) 

    test_df.loc[:,col+'_r'] = te_r





folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=99999)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    

    tr = train_df.loc[trn_idx,var_cols+['target']]

    va = train_df.loc[val_idx,var_cols+['target']]



    for col in tqdm(var_cols):

        encoded = t_encoding(tr,va,col)

        train_df.loc[val_idx,col+'_r'] = encoded

                

    

features = [col for col in train_df.columns if ('var' in col)]



train_df = train_df[features].reset_index(drop=True)

print('data prepared: {}'.format(train_df.shape))
param = {

    'bagging_freq': 5,          

    'bagging_fraction': 0.38,   'boost_from_average':'false',   

    'boost': 'gbdt',             'feature_fraction': 1,     'learning_rate': 0.0085,

    'max_depth': -1,             'metric':'auc',                'min_data_in_leaf': 80,     'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 13,            'num_threads': 8,              'tree_learner': 'serial',   'objective': 'binary',

    'reg_alpha': 0.1302650970728192, 'reg_lambda': 0.3603427518866501,'verbosity': 1

}

folds = StratifiedKFold(n_splits=12, shuffle=False, random_state=99999)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold {}".format(fold_))

    

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

    val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

    clf = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 2000)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

    predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = predictions

sub.to_csv("submission.csv", index=False)