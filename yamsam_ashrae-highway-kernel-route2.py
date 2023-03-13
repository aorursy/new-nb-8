folds = 3



debug = False

num_rounds=1000



use_ucf  = False # use ucf data in train

use_sort = False # sort by month

del_2016 = False # delete site0 2016 from train

copy_site2 = False

ucf_year = [2017, 2018] # ucf data year used in train 
import gc

import os

from pathlib import Path

import random

import sys



from tqdm import tqdm_notebook as tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from IPython.core.display import display, HTML



# --- plotly ---

from plotly import tools, subplots

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.figure_factory as ff



# --- models ---

from sklearn import preprocessing

from sklearn.model_selection import KFold

import lightgbm as lgb

import xgboost as xgb

import catboost as cb



from sklearn.metrics import mean_squared_error
# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified to support timestamp type, categorical type

# Modified to add option to use float16 or not. feather format does not support float16.

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            # skip datetime type or categorical type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df

root = Path('../input/ashrae-feather-format-for-fast-loading')



train_df = pd.read_feather(root/'train.feather')

weather_train_df = pd.read_feather(root/'weather_train.feather')

weather_test_df = pd.read_feather(root/'weather_test.feather')

building_meta_df = pd.read_feather(root/'building_metadata.feather')
# load site 0 data

ucf_root = Path('../input/ashrae-ucf-spider-and-eda-full-test-labels')

leak0_df = pd.read_pickle(ucf_root/'site0.pkl') 

leak0_df['meter_reading'] = leak0_df.meter_reading_scraped

leak0_df.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)

leak0_df.fillna(0, inplace=True)

leak0_df.loc[leak0_df.meter_reading < 0, 'meter_reading'] = 0

leak0_df = leak0_df[leak0_df.timestamp.dt.year > 2016]

print(len(leak0_df))
leak0_df.head()
# load site 1 data

ucl_root = Path('../usr/lib/ucl_data_leakage_episode_2')

leak1_df = pd.read_pickle(ucl_root/'site1.pkl') 

leak1_df['meter_reading'] = leak1_df.meter_reading_scraped

leak1_df.drop(['meter_reading_scraped'], axis=1, inplace=True)

leak1_df.fillna(0, inplace=True)

leak1_df.loc[leak1_df.meter_reading < 0, 'meter_reading'] = 0

leak1_df = leak1_df[leak1_df.timestamp.dt.year > 2016]

print(len(leak1_df))
leak1_df.head()
# load site 2 data

leak2_df = pd.read_csv('/kaggle/input/asu-buildings-energy-consumption/asu_2016-2018.csv')

leak2_df['timestamp'] = pd.to_datetime(leak2_df['timestamp'])

leak2_df.fillna(0, inplace=True)

leak2_df.loc[leak2_df.meter_reading < 0, 'meter_reading'] = 0



if copy_site2: # some data looks better than train in 2016

    train_df = train_df.merge(leak2_df, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")

    train_df.loc[train_df.meter_reading_y.isnull() == False, 'meter_reading_x'] = train_df[train_df.meter_reading_y.isnull() == False].meter_reading_y

    train_df.drop('meter_reading_y', axis=1, inplace=True)

    train_df.rename(columns={'meter_reading_x': 'meter_reading'}, inplace=True)



leak2_df = leak2_df[leak2_df.timestamp.dt.year > 2016]

leak2_df = leak2_df[leak2_df.building_id!=245] # building 245 is missing now.



print(len(leak2_df))
train_df.head()
leak2_df.head()
print ('leak rate = {}'.format((len(leak0_df) + len(leak1_df) + len(leak2_df))  / len(train_df)))
if use_ucf:

    leak0_df = leak0_df[leak0_df.timestamp.dt.year.isin(ucf_year)]

    leak1_df = leak1_df[leak1_df.timestamp.dt.year.isin(ucf_year)]

    leak2_df = leak2_df[leak2_df.timestamp.dt.year.isin(ucf_year)]

    

    train_df = pd.concat([train_df, leak0_df, leak1_df, leak2_df])

    train_df.reset_index(inplace=True)

    

    weather_train_df = pd.concat([weather_train_df, weather_test_df])

    weather_train_df.reset_index(inplace=True)
del weather_test_df, leak0_df, leak1_df, leak2_df #leak_meta_df

gc.collect()
train_df['date'] = train_df['timestamp'].dt.date

train_df['meter_reading_log1p'] = np.log1p(train_df['meter_reading'])
def plot_date_usage(train_df, meter=0, building_id=0):

    train_temp_df = train_df[train_df['meter'] == meter]

    train_temp_df = train_temp_df[train_temp_df['building_id'] == building_id]    

    train_temp_df_meter = train_temp_df.groupby('date')['meter_reading_log1p'].sum()

    train_temp_df_meter = train_temp_df_meter.to_frame().reset_index()

    fig = px.line(train_temp_df_meter, x='date', y='meter_reading_log1p')

    fig.show()
plot_date_usage(train_df, meter=0, building_id=0)
plot_date_usage(train_df, meter=0, building_id=105)
plot_date_usage(train_df, meter=0, building_id=156)
building_meta_df[building_meta_df.site_id == 0]
train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')
zone_dict={0:4,1:0,2:7,3:4,4:7,5:0,6:4,7:4,8:4,9:5,10:7,11:4,12:0,13:5,14:4,15:4} 



def set_localtime(df):

    for sid, zone in zone_dict.items():

        sids = df.site_id == sid

        df.loc[sids, 'timestamp'] = df[sids].timestamp - pd.offsets.Hour(zone)
def preprocess(df):

    df["hour"] = df["timestamp"].dt.hour

#     df["day"] = df["timestamp"].dt.day

    df["weekend"] = df["timestamp"].dt.weekday

    df["month"] = df["timestamp"].dt.month

    df["dayofweek"] = df["timestamp"].dt.dayofweek



#     hour_rad = df["hour"].values / 24. * 2 * np.pi

#     df["hour_sin"] = np.sin(hour_rad)

#     df["hour_cos"] = np.cos(hour_rad)
preprocess(train_df)
# sort train. i dont know it is best

if use_ucf and use_sort:

    train_df = train_df.sort_values('month')

    train_df = train_df.reset_index()
df_group = train_df.groupby('building_id')['meter_reading_log1p']

#building_mean = df_group.mean().astype(np.float16)

building_median = df_group.median().astype(np.float16)

#building_min = df_group.min().astype(np.float16)

#building_max = df_group.max().astype(np.float16)

#building_std = df_group.std().astype(np.float16)



#train_df['building_mean'] = train_df['building_id'].map(building_mean)

train_df['building_median'] = train_df['building_id'].map(building_median)

#train_df['building_min'] = train_df['building_id'].map(building_min)

#train_df['building_max'] = train_df['building_id'].map(building_max)

#train_df['building_std'] = train_df['building_id'].map(building_std)

del df_group
#building_mean.head()
weather_train_df.head()
# weather_train_df.describe()
weather_train_df.isna().sum()
weather_train_df.shape
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
weather_train_df = weather_train_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))
weather_train_df.groupby('site_id').apply(lambda group: group.isna().sum())
def add_lag_feature(weather_df, window=3):

    group_df = weather_df.groupby('site_id')

    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']

    rolled = group_df[cols].rolling(window=window, min_periods=0)

    lag_mean = rolled.mean().reset_index().astype(np.float16)

    lag_max = rolled.max().reset_index().astype(np.float16)

    lag_min = rolled.min().reset_index().astype(np.float16)

    lag_std = rolled.std().reset_index().astype(np.float16)

    for col in cols:

        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]

        weather_df[f'{col}_max_lag{window}'] = lag_max[col]

        weather_df[f'{col}_min_lag{window}'] = lag_min[col]

        weather_df[f'{col}_std_lag{window}'] = lag_std[col]
# skip lag feature to save memory

#add_lag_feature(weather_train_df, window=3)

#add_lag_feature(weather_train_df, window=72)
set_localtime(weather_train_df)
weather_train_df.head()
weather_train_df.columns
# categorize primary_use column to reduce memory on merge...



primary_use_list = building_meta_df['primary_use'].unique()

primary_use_dict = {key: value for value, key in enumerate(primary_use_list)} 

print('primary_use_dict: ', primary_use_dict)

building_meta_df['primary_use'] = building_meta_df['primary_use'].map(primary_use_dict)



gc.collect()
train_df = reduce_mem_usage(train_df, use_float16=True)

building_meta_df = reduce_mem_usage(building_meta_df, use_float16=True)

weather_train_df = reduce_mem_usage(weather_train_df, use_float16=True)
building_meta_df.head()
category_cols = ['building_id', 'site_id', 'primary_use']  # , 'meter'

feature_cols = ['square_feet', 'year_built'] + [

    'hour', 'weekend', # 'month' , 'dayofweek'

    'building_median'] + [

    'air_temperature', 'cloud_coverage',

    'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

    'wind_direction', 'wind_speed',]

#     'air_temperature_mean_lag72',

#     'air_temperature_max_lag72', 'air_temperature_min_lag72',

#     'air_temperature_std_lag72', 'cloud_coverage_mean_lag72',

#     'dew_temperature_mean_lag72', 'precip_depth_1_hr_mean_lag72',

#     'sea_level_pressure_mean_lag72', 'wind_direction_mean_lag72',

#     'wind_speed_mean_lag72', 'air_temperature_mean_lag3',

#     'air_temperature_max_lag3',

#     'air_temperature_min_lag3', 'cloud_coverage_mean_lag3',

#     'dew_temperature_mean_lag3',

#     'precip_depth_1_hr_mean_lag3', 'sea_level_pressure_mean_lag3',

#     'wind_direction_mean_lag3', 'wind_speed_mean_lag3']
def create_X_y(train_df, target_meter):

    target_train_df = train_df[train_df['meter'] == target_meter]

    target_train_df = target_train_df.merge(building_meta_df, on='building_id', how='left')

    target_train_df = target_train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    X_train = target_train_df[feature_cols + category_cols]

    y_train = target_train_df['meter_reading_log1p'].values



    del target_train_df

    return X_train, y_train
def fit_lgbm(train, val, devices=(-1,), seed=None, cat_features=None, num_rounds=1500, lr=0.1, bf=0.1):

    """Train Light GBM model"""

    X_train, y_train = train

    X_valid, y_valid = val

    metric = 'l2'

    params = {'num_leaves': 31,

              'objective': 'regression',

#               'max_depth': -1,

              'learning_rate': lr,

              "boosting": "gbdt",

              "bagging_freq": 5,

              "bagging_fraction": bf,

              "feature_fraction": 0.9,

              "metric": metric,

#               "verbosity": -1,

#               'reg_alpha': 0.1,

#               'reg_lambda': 0.3

              }

    device = devices[0]

    if device == -1:

        # use cpu

        pass

    else:

        # use gpu

        print(f'using gpu device_id {device}...')

        params.update({'device': 'gpu', 'gpu_device_id': device})



    params['seed'] = seed



    early_stop = 20

    verbose_eval = 20



    d_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_features)

    d_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_features)

    watchlist = [d_train, d_valid]



    print('training LGB:')

    model = lgb.train(params,

                      train_set=d_train,

                      num_boost_round=num_rounds,

                      valid_sets=watchlist,

                      verbose_eval=verbose_eval,

                      early_stopping_rounds=early_stop)



    # predictions

    y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

    

    print('best_score', model.best_score)

    log = {'train/mae': model.best_score['training']['l2'],

           'valid/mae': model.best_score['valid_1']['l2']}

    return model, y_pred_valid, log
#folds = 5

seed = 666

shuffle = False

kf = KFold(n_splits=folds, shuffle=shuffle, random_state=seed)

oof_total = 0
target_meter = 0

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])

gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models0 = []

for train_idx, valid_idx in kf.split(X_train, y_train):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    print('train', len(train_idx), 'valid', len(valid_idx))

#     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])

    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,

                                        num_rounds=num_rounds, lr=0.05, bf=0.7)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models0.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

sns.distplot(y_valid_pred_total)



oof0 = mean_squared_error(y_train, y_valid_pred_total)

oof_total += oof0 * len(y_train)



del X_train, y_train

gc.collect()
def plot_feature_importance(model):

    importance_df = pd.DataFrame(model.feature_importance(),

                                 index=feature_cols + category_cols,

                                 columns=['importance']).sort_values('importance')

    fig, ax = plt.subplots(figsize=(8, 8))

    importance_df.plot.barh(ax=ax)

    fig.show()
target_meter = 1

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])

gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models1 = []

for train_idx, valid_idx in kf.split(X_train, y_train):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    print('train', len(train_idx), 'valid', len(valid_idx))

#     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])

    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=num_rounds,

                                       lr=0.05, bf=0.5)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models1.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

sns.distplot(y_valid_pred_total)



oof1 = mean_squared_error(y_train, y_valid_pred_total)

oof_total += oof1 * len(y_train)



del X_train, y_train

gc.collect()
target_meter = 2

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])



gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models2 = []

for train_idx, valid_idx in kf.split(X_train, y_train):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    print('train', len(train_idx), 'valid', len(valid_idx))

#     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])

    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols,

                                        num_rounds=num_rounds, lr=0.05, bf=0.8)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models2.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

sns.distplot(y_valid_pred_total)



oof2 = mean_squared_error(y_train, y_valid_pred_total)

oof_total += oof2 * len(y_train)



del X_train, y_train

gc.collect()
target_meter = 3

X_train, y_train = create_X_y(train_df, target_meter=target_meter)

y_valid_pred_total = np.zeros(X_train.shape[0])



gc.collect()

print('target_meter', target_meter, X_train.shape)



cat_features = [X_train.columns.get_loc(cat_col) for cat_col in category_cols]

print('cat_features', cat_features)



models3 = []

for train_idx, valid_idx in kf.split(X_train, y_train):

    train_data = X_train.iloc[train_idx,:], y_train[train_idx]

    valid_data = X_train.iloc[valid_idx,:], y_train[valid_idx]



    print('train', len(train_idx), 'valid', len(valid_idx))

#     model, y_pred_valid, log = fit_cb(train_data, valid_data, cat_features=cat_features, devices=[0,])

    model, y_pred_valid, log = fit_lgbm(train_data, valid_data, cat_features=category_cols, num_rounds=num_rounds,

                                       lr=0.03, bf=0.9)

    y_valid_pred_total[valid_idx] = y_pred_valid

    models3.append(model)

    gc.collect()

    if debug:

        break



sns.distplot(y_train)

sns.distplot(y_valid_pred_total)



oof3 = mean_squared_error(y_train, y_valid_pred_total)

oof_total += oof3 * len(y_train)



del X_train, y_train

gc.collect()
oof_total = oof_total / len(train_df)
del train_df, weather_train_df, building_meta_df 

gc.collect()
print('loading...')

test_df = pd.read_feather(root/'test.feather')

weather_test_df = pd.read_feather(root/'weather_test.feather')

building_meta_df = pd.read_feather(root/'building_metadata.feather')



set_localtime(weather_test_df)



print('preprocessing building...')

test_df['date'] = test_df['timestamp'].dt.date

preprocess(test_df)

#test_df['building_mean'] = test_df['building_id'].map(building_mean)

test_df['building_median'] = test_df['building_id'].map(building_median)

# test_df['building_min'] = test_df['building_id'].map(building_min)

# test_df['building_max'] = test_df['building_id'].map(building_max)

# test_df['building_std'] = test_df['building_id'].map(building_std)



print('preprocessing weather...')

weather_test_df = weather_test_df.groupby('site_id').apply(lambda group: group.interpolate(limit_direction='both'))

weather_test_df.groupby('site_id').apply(lambda group: group.isna().sum())



# add_lag_feature(weather_test_df, window=3)

# add_lag_feature(weather_test_df, window=72)



print('reduce mem usage...')

test_df = reduce_mem_usage(test_df, use_float16=True)

weather_test_df = reduce_mem_usage(weather_test_df, use_float16=True)



gc.collect()
sample_submission = pd.read_feather(os.path.join(root, 'sample_submission.feather'))

sample_submission = reduce_mem_usage(sample_submission)
def create_X(test_df, target_meter):

    target_test_df = test_df[test_df['meter'] == target_meter]

    target_test_df = target_test_df.merge(building_meta_df, on='building_id', how='left')

    target_test_df = target_test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')

    X_test = target_test_df[feature_cols + category_cols]

    return X_test
def pred(X_test, models, batch_size=1000000):

    iterations = (X_test.shape[0] + batch_size -1) // batch_size

    print('iterations', iterations)



    y_test_pred_total = np.zeros(X_test.shape[0])

    for i, model in enumerate(models):

        print(f'predicting {i}-th model')

        for k in tqdm(range(iterations)):

            y_pred_test = model.predict(X_test[k*batch_size:(k+1)*batch_size], num_iteration=model.best_iteration)

            y_test_pred_total[k*batch_size:(k+1)*batch_size] += y_pred_test



    y_test_pred_total /= len(models)

    return y_test_pred_total


X_test = create_X(test_df, target_meter=0)

gc.collect()



y_test0 = pred(X_test, models0)



sns.distplot(y_test0)



del X_test

gc.collect()

X_test = create_X(test_df, target_meter=1)

gc.collect()



y_test1 = pred(X_test, models1)

sns.distplot(y_test1)



del X_test

gc.collect()

X_test = create_X(test_df, target_meter=2)

gc.collect()



y_test2 = pred(X_test, models2)

sns.distplot(y_test2)



del X_test

gc.collect()
X_test = create_X(test_df, target_meter=3)

gc.collect()



y_test3 = pred(X_test, models3)

sns.distplot(y_test3)



del X_test

gc.collect()
sample_submission.loc[test_df['meter'] == 0, 'meter_reading'] = np.expm1(y_test0)

sample_submission.loc[test_df['meter'] == 1, 'meter_reading'] = np.expm1(y_test1)

sample_submission.loc[test_df['meter'] == 2, 'meter_reading'] = np.expm1(y_test2)

sample_submission.loc[test_df['meter'] == 3, 'meter_reading'] = np.expm1(y_test3)
if not debug:

    sample_submission.to_csv('submission.csv', index=False, float_format='%.4f')
# sie 0

leak_score0 = 0



leak_df = pd.read_pickle(ucf_root/'site0.pkl') 

leak_df['meter_reading'] = leak_df.meter_reading_scraped

leak_df.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)

leak_df.fillna(0, inplace=True)

leak_df = leak_df[leak_df.timestamp.dt.year > 2016]

leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values



sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0



for bid in leak_df.building_id.unique():

    temp_df = leak_df[(leak_df.building_id == bid)]

    for m in temp_df.meter.unique():

        v0 = sample_submission.loc[(test_df.building_id == bid)&(test_df.meter==m), 'meter_reading'].values

        v1 = temp_df[temp_df.meter==m].meter_reading.values

        

        leak_score0 += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)

        

        sample_submission.loc[(test_df.building_id == bid)&(test_df.meter==m), 'meter_reading'] = temp_df[temp_df.meter==m].meter_reading.values

        

leak_score0 /= len(leak_df)
# site 1

leak_score1 = 0



leak_df = pd.read_pickle(ucl_root/'site1.pkl') 

leak_df['meter_reading'] = leak_df.meter_reading_scraped

leak_df.drop(['meter_reading_scraped'], axis=1, inplace=True)

leak_df.fillna(0, inplace=True)

leak_df = leak_df[leak_df.timestamp.dt.year > 2016]

leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values



#sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0



for bid in leak_df.building_id.unique():

    temp_df = leak_df[(leak_df.building_id == bid)]

    for m in temp_df.meter.unique():

        v0 = sample_submission.loc[(test_df.building_id == bid)&(test_df.meter==m), 'meter_reading'].values

        v1 = temp_df[temp_df.meter==m].meter_reading.values

        

        leak_score1 += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)

        

        sample_submission.loc[(test_df.building_id == bid)&(test_df.meter==m), 'meter_reading'] = temp_df[temp_df.meter==m].meter_reading.values



leak_score1 /= len(leak_df)        
# site 2

leak_score2 = 0



# load site 2 data

leak_df = pd.read_csv('/kaggle/input/asu-buildings-energy-consumption/asu_2016-2018.csv')

leak_df['timestamp'] = pd.to_datetime(leak_df['timestamp'])

leak_df.fillna(0, inplace=True)

leak_df.loc[leak_df.meter_reading < 0, 'meter_reading'] = 0

leak_df = leak_df[leak_df.timestamp.dt.year > 2016]



leak_df = leak_df[leak_df.building_id!=245] # building 245 is missing now.



#sample_submission.loc[sample_submission.meter_reading < 0, 'meter_reading'] = 0



for bid in leak_df.building_id.unique():

    temp_df = leak_df[(leak_df.building_id == bid)]

    for m in temp_df.meter.unique():

        v0 = sample_submission.loc[(test_df.building_id == bid)&(test_df.meter==m), 'meter_reading'].values

        v1 = temp_df[temp_df.meter==m].meter_reading.values

        

        leak_score2 += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)

        

        sample_submission.loc[(test_df.building_id == bid)&(test_df.meter==m), 'meter_reading'] = temp_df[temp_df.meter==m].meter_reading.values



leak_score2 /= len(leak_df)        
if not debug:

    sample_submission.to_csv('submission_ucf_replaced.csv', index=False, float_format='%.4f')
sample_submission.head()
np.log1p(sample_submission['meter_reading']).hist(bins=100)
print ('oof score meter0 =', np.sqrt(oof0))

print ('oof score meter1 =', np.sqrt(oof1))

print ('oof score meter2 =', np.sqrt(oof2))

print ('oof score meter3 =', np.sqrt(oof3))

print ('oof score total  =', np.sqrt(oof_total))
print ('UCF score = ', np.sqrt(leak_score0))

print ('UCL score = ', np.sqrt(leak_score1))

print ('ASU score = ', np.sqrt(leak_score2))
plot_feature_importance(models0[0])
plot_feature_importance(models1[0])
plot_feature_importance(models2[0])
plot_feature_importance(models3[0])