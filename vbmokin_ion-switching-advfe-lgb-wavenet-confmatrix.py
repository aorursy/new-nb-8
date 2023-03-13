# FE-4



# Seed for random_state

seed_random = 42



# FE-2 : window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000, 50000, 100000, 500000]

# About selection of these values see my post: https://www.kaggle.com/c/liverpool-ion-switching/discussion/135073

window_sizes = [10, 50]

# with new my features: signal_shift_+2, signal_shift_-2

# without ratio-, diff-features

# without 'medianbatch', 'abs_avgbatch', 'abs_maxbatch' with smallest feature importance



# LGB-2 model tuning

lr_lgb = 0.05

num_leaves = 200

num_iterations = 1500  # recommended 2000

# 'metric'is 'logloss'



# Wavenet with SHIFTED-RFC Proba and CBR model tuning (the pilot version)

lr_wn = 0.0015  # recommended 0.0015

EPOCHS_wn = 15  # recommended 180

SPLITS = 2      # recommended 6



# XGB-1 model tuning

# lr_xgb = 0.05

# max_depth_xgb = 10

# num_boost_round_xgb = 1000

#'eval_metric'is 'logloss'



# Set weight of models

w_lgb = 0.5

w_wnet = 1 - w_lgb

print(w_wnet)

# without LogReg, MLP, XGB, Ridge Regression and SGDRegressor models
import os


import tensorflow as tf

from tensorflow.keras.layers import *

import pandas as pd

import numpy as np

import random

from tensorflow.keras.callbacks import Callback, LearningRateScheduler

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

from tensorflow.keras import losses, models, optimizers

import tensorflow_addons as tfa

import math



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, KFold, train_test_split, GroupKFold

from sklearn.utils import shuffle

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, make_scorer

import lightgbm as lgb

import xgboost as xgb

#from pykalman import KalmanFilter



from functools import partial

import scipy as sp



import time

import datetime



import gc



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:

        if col != 'time':

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
train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
train.head()
train.shape
test.head()

for window in window_sizes:

    train["rolling_mean_" + str(window)] = train['signal'].rolling(window=window).mean()

    train["rolling_std_" + str(window)] = train['signal'].rolling(window=window).std()

    train["rolling_var_" + str(window)] = train['signal'].rolling(window=window).var()

    train["rolling_min_" + str(window)] = train['signal'].rolling(window=window).min()

    train["rolling_max_" + str(window)] = train['signal'].rolling(window=window).max()

    

    #train["rolling_min_max_ratio_" + str(window)] = train["rolling_min_" + str(window)] / train["rolling_max_" + str(window)]

    #train["rolling_min_max_diff_" + str(window)] = train["rolling_max_" + str(window)] - train["rolling_min_" + str(window)]

    

    a = (train['signal'] - train['rolling_min_' + str(window)]) / (train['rolling_max_' + str(window)] - train['rolling_min_' + str(window)])

    train["norm_" + str(window)] = a * (np.floor(train['rolling_max_' + str(window)]) - np.ceil(train['rolling_min_' + str(window)]))

    

train = train.replace([np.inf, -np.inf], np.nan)    

train.fillna(0, inplace=True)

for window in window_sizes:

    test["rolling_mean_" + str(window)] = test['signal'].rolling(window=window).mean()

    test["rolling_std_" + str(window)] = test['signal'].rolling(window=window).std()

    test["rolling_var_" + str(window)] = test['signal'].rolling(window=window).var()

    test["rolling_min_" + str(window)] = test['signal'].rolling(window=window).min()

    test["rolling_max_" + str(window)] = test['signal'].rolling(window=window).max()

    

    #test["rolling_min_max_ratio_" + str(window)] = test["rolling_min_" + str(window)] / test["rolling_max_" + str(window)]

    #test["rolling_min_max_diff_" + str(window)] = test["rolling_max_" + str(window)] - test["rolling_min_" + str(window)]



    

    a = (test['signal'] - test['rolling_min_' + str(window)]) / (test['rolling_max_' + str(window)] - test['rolling_min_' + str(window)])

    test["norm_" + str(window)] = a * (np.floor(test['rolling_max_' + str(window)]) - np.ceil(test['rolling_min_' + str(window)]))



test = test.replace([np.inf, -np.inf], np.nan)    

test.fillna(0, inplace=True)

def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)

    

    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    

    # add shifts_1

    df['signal_shift_+1'] = [0,] + list(df['signal'].values[:-1])

    df['signal_shift_-1'] = list(df['signal'].values[1:]) + [0]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+1'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-1'][i] = np.nan

    

    # add shifts_2 - my upgrade

    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])

    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0] + [1]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==1].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-2'][i] = np.nan

    for i in df[df['batch_index']==49998].index:

        df['signal_shift_-2'][i] = np.nan

    

    df = df.drop(columns=['batch', 'batch_index', 'batch_slices', 'batch_slices2'])



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels']]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    df = df.replace([np.inf, -np.inf], np.nan)    

    df.fillna(0, inplace=True)

    gc.collect()

    return df



train = features(train)

test = features(test)
train.info()
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
y = train['open_channels']

col = [c for c in train.columns if c not in ['time', 'open_channels', 'group', 'medianbatch', 'abs_avgbatch', 'abs_maxbatch']]
train.info()
train.head()
test.info()
test.head()
# %%time

# scaler = StandardScaler()

# train_reg = pd.DataFrame(

#     scaler.fit_transform(train[col]),

#     columns=train[col].columns,

#     index=train[col].index

# )

# test_reg = pd.DataFrame(

#     scaler.transform(test[col]),

#     columns=test[col].columns,

#     index=test[col].index

# )
# %%time

# ridge_reg = Ridge(tol=5e-5, fit_intercept=False, random_state=seed_random)

# ridge_reg.fit(train_reg, y)
# y_train_ridge = ridge_reg.predict(train_reg)

# y_preds_ridge = ridge_reg.predict(test_reg)

# y_train_ridge = np.clip(y_train_ridge, 0, 10).astype(int)

# y_preds_ridge = np.clip(y_preds_ridge, 0, 10).astype(int)
# %%time

# sgd = SGDRegressor(learning_rate = 'adaptive', fit_intercept=False, random_state=seed_random)

# sgd.fit(train_reg, y)
# y_train_sgd = sgd.predict(train_reg)

# y_preds_sgd = sgd.predict(test_reg)

# y_train_sgd = np.clip(y_train_sgd, 0, 10).astype(int)

# y_preds_sgd = np.clip(y_preds_sgd, 0, 10).astype(int)
# %%time

# logreg = LogisticRegression(random_state=seed_random)

# logreg.fit(train_reg, y)
# y_logreg_train = logreg.predict(train_reg)

# y_logreg_pred = logreg.predict(test_reg)

# y_logreg_train = np.clip(y_logreg_train, 0, 10).astype(int)

# y_logreg_pred = np.clip(y_logreg_pred, 0, 10).astype(int)
# # https://www.kaggle.com/vbmokin/automatic-selection-from-20-classifier-models

# # https://www.kaggle.com/vbmokin/bod-prediction-in-river-15-regression-models

# mlp = MLPClassifier()

# f1scoring = make_scorer(f1_score, average = 'macro')

# param_grid = {'hidden_layer_sizes': [i for i in range(n_layer_0, n_layer_max)],

#               'activation': ['relu'],

#               'solver': ['adam'],

#               'learning_rate': ['constant'],

#               'learning_rate_init': [lr_mlp],

#               'power_t': [0.5],

#               'alpha': [0.0001],

#               'max_iter': [1000],

#               'early_stopping': [True],

#               'warm_start': [False],

#               'random_state': [seed_random]}

# mlp_GS = GridSearchCV(mlp, param_grid=param_grid, scoring=f1scoring,

#                    cv=cv_mlp, verbose=True, pre_dispatch='2*n_jobs')

# mlp_GS.fit(train_reg, y)

# print("Best parameters set:", mlp_GS.best_params_)
# y_mlp_pred = mlp_GS.predict(test_reg)

# y_pred_train_mlp = mlp_GS.predict(train_reg)
# del train_reg, test_reg

# gc.collect()
# Thanks to https://www.kaggle.com/siavrez/simple-eda-model

def MacroF1Metric(preds, dtrain):

    labels = dtrain.get_label()

    preds = np.round(np.clip(preds, 0, 10)).astype(int)

    score = f1_score(labels, preds, average = 'macro')

    return ('MacroF1Metric', score, True)

# Thanks to https://www.kaggle.com/jazivxt/physically-possible with tuning from https://www.kaggle.com/siavrez/simple-eda-model and my tuning

X_train, X_valid, y_train, y_valid = train_test_split(train[col], y, test_size=0.3, random_state=seed_random)

params = {'learning_rate': lr_lgb, 

          'max_depth': -1, 

          'num_leaves': num_leaves,

          'metric': 'logloss', 

          'random_state': seed_random, 

          'n_jobs':-1, 

          'sample_fraction':0.33}

model = lgb.train(params, lgb.Dataset(X_train, y_train), num_iterations, lgb.Dataset(X_valid, y_valid), verbose_eval=100, early_stopping_rounds=200, feval=MacroF1Metric)

gc.collect()

y_lgb_pred = model.predict(test[col], num_iteration=model.best_iteration)

y_pred_train_lgb = model.predict(train[col], num_iteration=model.best_iteration)

gc.collect()
print('LGB score {0:.4f}'.format(np.mean(f1_score(y, np.round(np.clip(y_pred_train_lgb,0,10)).astype(int), average="macro"))))
fig =  plt.figure(figsize = (15,15))

axes = fig.add_subplot(111)

lgb.plot_importance(model,ax = axes,height = 0.5)

plt.show();plt.close()

gc.collect()
# # Thanks to https://www.kaggle.com/teejmahal20/3-simple-ideas-ensemble

# train_set = xgb.DMatrix(X_train, y_train)

# val_set = xgb.DMatrix(X_valid, y_valid)

# del X_train, X_valid, y_train, y_valid

# gc.collect()
# %%time

# params_xgb = {'colsample_bytree': 0.375,

#               'learning_rate': lr_xgb,

#               'max_depth': max_depth_xgb, 

#               'subsample': 1, 

#               'objective':'reg:squarederror',

#               'eval_metric':'logloss'}



# modelx = xgb.train(params_xgb, train_set, num_boost_round=num_boost_round_xgb, evals=[(train_set, 'train'), (val_set, 'val')], 

#                                      verbose_eval=50, early_stopping_rounds=200)

# del train_set, val_set

# gc.collect()
# %%time

# y_xgb_pred = modelx.predict(xgb.DMatrix(test[col]))

# y_pred_train_xgb = modelx.predict(xgb.DMatrix(train[col]))

# gc.collect()
# print('XGB score {0:.4f}'.format(np.mean(f1_score(y, np.round(np.clip(y_pred_train_xgb,0,10)).astype(int), average="macro"))))
# Thanks to https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr

# configurations and main hyperparammeters

EPOCHS = EPOCHS_wn

NNBATCHSIZE = 16

GROUP_BATCH_SIZE = 4000

SEED = 321

LR = lr_wn

# Thanks to https://www.kaggle.com/nxrprime/wavenet-with-shifted-rfc-proba-and-cbr - my upgrade: improved FE and simplified model

# Create batches of 4000 observations

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)



# read data

def read_data():

    train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})

    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})

    

    Y_train_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")

    Y_test_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")

    

    for i in range(11):

        train[f"proba_{i}"] = Y_train_proba[:, i]

        test[f"proba_{i}"] = Y_test_proba[:, i]



    return train, test, sub



# create batches of 4000 observations

def batching(df, batch_size):

    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values

    df['group'] = df['group'].astype(np.uint16)

    return df



# normalize the data (standard scaler). We can also try other scalers for a better score!

def normalize(train, test):

    train_input_mean = train.signal.mean()

    train_input_sigma = train.signal.std()

    train['signal'] = (train.signal - train_input_mean) / train_input_sigma

    test['signal'] = (test.signal - train_input_mean) / train_input_sigma

    return train, test



# get lead and lags features

def lag_with_pct_change(df, windows):

    for window in windows:    

        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)

        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)

    return df



# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).

def run_feat_engineering(df, batch_size):

    # create batches

    df = batching(df, batch_size = batch_size)

    # create leads and lags (1, 2, 3 making them 6 features)

    df = lag_with_pct_change(df, [1, 2, 3])

    # create signal ** 2 (this is the new feature)

    df['signal_2'] = df['signal'] ** 2

    

    # 'batch_index' from https://www.kaggle.com/jazivxt/physically-possible and https://www.kaggle.com/siavrez/simple-eda-model

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    

    # add shifts_2 - my upgrade

    df['signal_shift_+2'] = [0,] + [1,] + list(df['signal'].values[:-2])

    df['signal_shift_-2'] = list(df['signal'].values[2:]) + [0] + [1]

    for i in df[df['batch_index']==0].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==1].index:

        df['signal_shift_+2'][i] = np.nan

    for i in df[df['batch_index']==49999].index:

        df['signal_shift_-2'][i] = np.nan

    for i in df[df['batch_index']==49998].index:

        df['signal_shift_-2'][i] = np.nan    

    df = df.drop(columns=['batch', 'batch_index'])

    

    return df



# fillna with the mean and select features for training

def feature_selection(train, test):

    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]

    train = train.replace([np.inf, -np.inf], np.nan)

    test = test.replace([np.inf, -np.inf], np.nan)

    for feature in features:

        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()

        train[feature] = train[feature].fillna(feature_mean)

        test[feature] = test[feature].fillna(feature_mean)

    return train, test, features



# model function (very important, you can try different arquitectures to get a better score. I believe that top public leaderboard is a 1D Conv + RNN style)

def Classifier(shape_):

    

    def cbr(x, out_layer, kernel, stride, dilation):

        x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)

        x = BatchNormalization()(x)

        x = Activation("relu")(x)

        return x

    

    def wave_block(x, filters, kernel_size, n):

        dilation_rates = [2**i for i in range(n)]

        x = Conv1D(filters = filters,

                   kernel_size = 1,

                   padding = 'same')(x)

        res_x = x

        for dilation_rate in dilation_rates:

            tanh_out = Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same', 

                              activation = 'tanh', 

                              dilation_rate = dilation_rate)(x)

            sigm_out = Conv1D(filters = filters,

                              kernel_size = kernel_size,

                              padding = 'same',

                              activation = 'sigmoid', 

                              dilation_rate = dilation_rate)(x)

            x = Multiply()([tanh_out, sigm_out])

            x = Conv1D(filters = filters,

                       kernel_size = 1,

                       padding = 'same')(x)

            res_x = Add()([res_x, x])

        return res_x

    

    inp = Input(shape = (shape_))

    x = cbr(inp, 64, 7, 1, 1)

    x = BatchNormalization()(x)

    x = wave_block(x, 64, 3, 4)

    x = BatchNormalization()(x)

    x = wave_block(x, 128, 3, 1)

    x = cbr(x, 32, 7, 1, 1)

    x = BatchNormalization()(x)

    x = wave_block(x, 64, 3, 1)

    x = cbr(x, 32, 7, 1, 1)

    x = BatchNormalization()(x)

    x = Dropout(0.2)(x)  

#     x = cbr(inp, 64, 7, 1, 1)

#     x = BatchNormalization()(x)

#     x = wave_block(x, 16, 3, 12)

#     x = BatchNormalization()(x)

#     x = wave_block(x, 32, 3, 8)

#     x = BatchNormalization()(x)

#     x = wave_block(x, 64, 3, 4)

#     x = BatchNormalization()(x)

#     x = wave_block(x, 128, 3, 1)

#     x = cbr(x, 32, 7, 1, 1)

#     x = BatchNormalization()(x)

#     x = wave_block(x, 64, 3, 1)

#     x = cbr(x, 32, 7, 1, 1)

#     x = BatchNormalization()(x)

#     x = Dropout(0.2)(x)

    out = Dense(11, activation = 'softmax', name = 'out')(x)

    

    model = models.Model(inputs = inp, outputs = out)

    

    opt = Adam(lr = LR)

    opt = tfa.optimizers.SWA(opt)

    model.compile(loss = losses.CategoricalCrossentropy(), optimizer = opt, metrics = ['accuracy'])

    return model



# function that decrease the learning as epochs increase (i also change this part of the code)

def lr_schedule(epoch):

    if epoch < 3:

        lr = LR

    elif epoch < 8:

        lr = LR / 3

    elif epoch < 30:

        lr = LR / 5

    elif epoch < 60:

        lr = LR / 7

    elif epoch < 70:

        lr = LR / 9

    elif epoch < 80:

        lr = LR / 11

    elif epoch < 90:

        lr = LR / 13

    else:

        lr = LR / 100

    return lr



# class to get macro f1 score. This is not entirely necessary but it's fun to check f1 score of each epoch (be carefull, if you use this function early stopping callback will not work)

class MacroF1(Callback):

    def __init__(self, model, inputs, targets):

        self.model = model

        self.inputs = inputs

        self.targets = np.argmax(targets, axis = 2).reshape(-1)

        

    def on_epoch_end(self, epoch, logs):

        pred = np.argmax(self.model.predict(self.inputs), axis = 2).reshape(-1)

        score = f1_score(self.targets, pred, average = 'macro')

        print(f'F1 Macro Score: {score:.5f}')



# main function to perfrom groupkfold cross validation (we have 1000 vectores of 4000 rows and 8 features (columns)). Going to make 5 groups with this subgroups.

def run_cv_model_by_batch(train, test, splits, batch_col, feats, sample_submission, nn_epochs, nn_batch_size):

    

    seed_everything(SEED)

    K.clear_session()

    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)

    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)

    tf.compat.v1.keras.backend.set_session(sess)

    oof_ = np.zeros((len(train), 11)) # build out of folds matrix with 11 columns, they represent our target variables classes (from 0 to 10)

    preds_ = np.zeros((len(test), 11))

    preds_train = np.zeros((len(train), 11))

    target = ['open_channels']

    group = train['group']

    kf = GroupKFold(n_splits=SPLITS)

    splits = [x for x in kf.split(train, train[target], group)]



    new_splits = []

    for sp in splits:

        new_split = []

        new_split.append(np.unique(group[sp[0]]))

        new_split.append(np.unique(group[sp[1]]))

        new_split.append(sp[1])    

        new_splits.append(new_split)

    # pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)

    tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)



    tr.columns = ['target_'+str(i) for i in range(11)] + ['group']

    target_cols = ['target_'+str(i) for i in range(11)]

    train_tr = np.array(list(tr.groupby('group').apply(lambda x: x[target_cols].values))).astype(np.float32)

    train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))

    test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))



    for n_fold, (tr_idx, val_idx, val_orig_idx) in enumerate(new_splits[0:], start=0):

        train_x, train_y = train[tr_idx], train_tr[tr_idx]

        valid_x, valid_y = train[val_idx], train_tr[val_idx]

        print(f'Our training dataset shape is {train_x.shape}')

        print(f'Our validation dataset shape is {valid_x.shape}')



        gc.collect()

        shape_ = (None, train_x.shape[2]) # input is going to be the number of feature we are using (dimension 2 of 0, 1, 2)

        model = Classifier(shape_)

        # using our lr_schedule function

        cb_lr_schedule = LearningRateScheduler(lr_schedule)

        model.fit(train_x,train_y,

                  epochs = nn_epochs,

                  callbacks = [cb_lr_schedule, MacroF1(model, valid_x, valid_y)], # adding custom evaluation metric for each epoch

                  batch_size = nn_batch_size,verbose = 2,

                  validation_data = (valid_x,valid_y))

        preds_f = model.predict(valid_x)

        f1_score_ = f1_score(np.argmax(valid_y, axis=2).reshape(-1),  np.argmax(preds_f, axis=2).reshape(-1), average = 'macro') # need to get the class with the biggest probability

        print(f'Training fold {n_fold + 1} completed. macro f1 score : {f1_score_ :1.5f}')

        preds_f = preds_f.reshape(-1, preds_f.shape[-1])

        oof_[val_orig_idx,:] += preds_f

        te_preds = model.predict(test)

        te_train_preds = model.predict(train)

        model.save("model-wavenet.h5")

        te_preds = te_preds.reshape(-1, te_preds.shape[-1])           

        preds_ += te_preds / SPLITS

        te_train_preds = te_train_preds.reshape(-1, te_train_preds.shape[-1])           

        preds_train += te_train_preds / SPLITS

        

    # calculate the oof macro f1_score

    f1_score_ = f1_score(np.argmax(train_tr, axis = 2).reshape(-1),  np.argmax(oof_, axis = 1), average = 'macro') # axis 2 for the 3 Dimension array and axis 1 for the 2 Domension Array (extracting the best class)

    print(f'Training completed. oof macro f1 score : {f1_score_:1.5f}')

    y_wnet_pred = np.argmax(preds_, axis = 1)

    y_pred_train_wnet = np.argmax(preds_train, axis = 1)

    #sample_submission.to_csv('submission_wavenet.csv', index=False, float_format='%.4f')

    return y_wnet_pred, y_pred_train_wnet



# this function run our entire program

print('Reading Data Started...')

train, test, sample_submission = read_data()

train, test = normalize(train, test)

print('Reading and Normalizing Data Completed')



print('Creating Features')

print('Feature Engineering Started...')

train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)

test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)

train, test, features = feature_selection(train, test)

print('Feature Engineering Completed...')

print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')

y_wnet_pred, y_pred_train_wnet = run_cv_model_by_batch(train, test, SPLITS, 'group', features, sample_submission, EPOCHS, NNBATCHSIZE)

print('Training Wavenet model completed...')
print('Wavenet score {0:.4f}'.format(np.mean(f1_score(y, np.round(np.clip(y_pred_train_wnet,0,10)).astype(int), average="macro"))))
# Showing Confusion Matrix

# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud

def plot_cm(y_true, y_pred, title):

    figsize=(14,14)

    y_pred = y_pred.astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
# Showing Confusion Matrix for Ridge model

#plot_cm(y, y_train_ridge, 'Confusion matrix for Ridge model')
# Showing Confusion Matrix for SGDRegressor model

#plot_cm(y, y_train_sgd, 'Confusion matrix for SGDRegressor model')
# # Showing Confusion Matrix for Logistic Regression

# plot_cm(y, y_logreg_train, 'Confusion matrix for Logistic Regression')
# Showing Confusion Matrix for LGB model

plot_cm(y, y_pred_train_lgb, 'Confusion matrix for LGB model')
# Showing Confusion Matrix for Wavenet model

plot_cm(y, y_pred_train_wnet, 'Confusion matrix for Wavenet model')
# # Showing Confusion Matrix for XGB model

# plot_cm(y, y_pred_train_xgb, 'Confusion matrix for XGB model')
# # Showing Confusion Matrix for MLP model

# plot_cm(y, y_pred_train_mlp, 'Confusion matrix for MLP model')
gc.collect()
# Showing Confusion Matrix for the main solution before rounded

y_train_preds = w_lgb*y_pred_train_lgb + w_wnet*y_pred_train_wnet

plot_cm(y, y_train_preds, 'Confusion matrix for the main solution before rounded')
del y_pred_train_lgb, y_pred_train_wnet

gc.collect()
#y_preds = w_ridge*y_preds_ridge + w_sgd*y_preds_sgd + w_logreg*y_logreg_pred + w_lgb*y_lgb_pred

#y_preds = w_logreg*y_logreg_pred + w_lgb*y_lgb_pred + w_mlp*y_mlp_pred

#y_preds = w_lgb*y_lgb_pred + w_xgb*y_xgb_pred

y_preds = w_lgb*y_lgb_pred + w_wnet*y_wnet_pred
del y_lgb_pred, y_wnet_pred

gc.collect()
def pred_proc(pred):

    pred = np.round(np.clip(pred, 0, 10))

    return pred.astype(int)
# Prediction processing for the main solution

y_preds = pred_proc(y_preds)

y_train_preds = pred_proc(y_train_preds)
# Showing Confusion Matrix for the processing main solution

plot_cm(y, y_train_preds, 'Confusion matrix for the processing main solution')
print('The main solution score {0:.4f}'.format(np.mean(f1_score(y, y_train_preds, average="macro"))))
del y_train_preds

gc.collect()
test['open_channels'] = y_preds

test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')
test.head(3)
print(test.time.values[:10])

print(test['open_channels'].mean())

print(test['open_channels'].hist())