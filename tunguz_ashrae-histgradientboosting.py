




import sklearn

sklearn.__version__

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from tqdm import tqdm_notebook as tqdm

import datetime

from meteocalc import feels_like, Temp

from sklearn import metrics

import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

building_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

weather_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')
# Original code from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude



def fill_weather_dataset(weather_df):

    

    # Find Missing Dates

    time_format = "%Y-%m-%d %H:%M:%S"

    start_date = datetime.datetime.strptime(weather_df['timestamp'].min(),time_format)

    end_date = datetime.datetime.strptime(weather_df['timestamp'].max(),time_format)

    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)

    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]



    missing_hours = []

    for site_id in range(16):

        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])

        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])

        new_rows['site_id'] = site_id

        weather_df = pd.concat([weather_df,new_rows])



        weather_df = weather_df.reset_index(drop=True)           



    # Add new Features

    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])

    weather_df["day"] = weather_df["datetime"].dt.day

    weather_df["week"] = weather_df["datetime"].dt.week

    weather_df["month"] = weather_df["datetime"].dt.month

    

    # Reset Index for Fast Update

    weather_df = weather_df.set_index(['site_id','day','month'])



    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])

    weather_df.update(air_temperature_filler,overwrite=False)



    # Step 1

    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()

    # Step 2

    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])



    weather_df.update(cloud_coverage_filler,overwrite=False)



    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])

    weather_df.update(due_temperature_filler,overwrite=False)



    # Step 1

    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()

    # Step 2

    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])



    weather_df.update(sea_level_filler,overwrite=False)



    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])

    weather_df.update(wind_direction_filler,overwrite=False)



    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])

    weather_df.update(wind_speed_filler,overwrite=False)



    # Step 1

    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()

    # Step 2

    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])



    weather_df.update(precip_depth_filler,overwrite=False)



    weather_df = weather_df.reset_index()

    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)

    

    def get_meteorological_features(data):

        def calculate_rh(df):

            df['relative_humidity'] = 100 * (np.exp((17.625 * df['dew_temperature']) / (243.04 + df['dew_temperature'])) / np.exp((17.625 * df['air_temperature'])/(243.04 + df['air_temperature'])))

        def calculate_fl(df):

            flike_final = []

            flike = []

            # calculate Feels Like temperature

            for i in range(len(df)):

                at = df['air_temperature'][i]

                rh = df['relative_humidity'][i]

                ws = df['wind_speed'][i]

                flike.append(feels_like(Temp(at, unit = 'C'), rh, ws))

            for i in range(len(flike)):

                flike_final.append(flike[i].f)

            df['feels_like'] = flike_final

            del flike_final, flike, at, rh, ws

        calculate_rh(data)

        calculate_fl(data)

        return data



    weather_df = get_meteorological_features(weather_df)

    

    return weather_df



# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin



from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

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

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def features_engineering(df):

    

    # Sort by timestamp

    df.sort_values("timestamp")

    df.reset_index(drop=True)

    

    # Add more features

    df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S")

    df["hour"] = df["timestamp"].dt.hour

    df["dayofweek"] = df["timestamp"].dt.weekday

    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",

                    "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",

                    "2017-01-02", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",

                    "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",

                    "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",

                    "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",

                    "2019-01-01"]

    df['group'] = df['timestamp'].dt.month

    df['group'].replace((1, 2, 3, 4), 1, inplace = True)

    df['group'].replace((5, 6, 7, 8), 2, inplace = True)

    df['group'].replace((9, 10, 11, 12), 3, inplace = True)

    df["is_holiday"] = (df.timestamp.isin(holidays)).astype(int)

    df['square_feet'] =  np.log1p(df['square_feet'])

    

    

    # Remove Unused Columns

    drop = ["timestamp"]

    df = df.drop(drop, axis=1)

    gc.collect()

    

    # Encode Categorical Data

    le = LabelEncoder()

    df["primary_use"] = le.fit_transform(df["primary_use"])

    

    return df
# weather manipulation

weather_df = fill_weather_dataset(weather_df)



# memory reduction

train_df = reduce_mem_usage(train_df,use_float16=True)

building_df = reduce_mem_usage(building_df,use_float16=True)

weather_df = reduce_mem_usage(weather_df,use_float16=True)



# merge data

train_df = train_df.merge(building_df, left_on='building_id',right_on='building_id',how='left')

train_df = train_df.merge(weather_df,how='left',left_on=['site_id','timestamp'],right_on=['site_id','timestamp'])

del weather_df

gc.collect()



# feature engineering

train_df = features_engineering(train_df)



# transform target variable

train_df['meter_reading'] = np.log1p(train_df["meter_reading"])
# eliminate bad rows

bad_rows = pd.read_csv('../input/bad-rows-ashrae/rows_to_drop.csv')

train_df.drop(bad_rows.loc[:, '0'], inplace = True)

train_df.reset_index(drop = True, inplace = True)
# declare target, categorical and numeric columns

target = 'meter_reading'

categorical = ['building_id', 'site_id', 'primary_use', 'meter', 'is_holiday', 'dayofweek']

numeric_cols = [col for col in train_df.columns if col not in categorical + [target, 'timestamp', 'group']]

features = categorical + numeric_cols
def run_hgbr(train, cat_features = categorical, num_rounds = 1200, folds = 3):

    kf = GroupKFold(n_splits = folds)

    models = []



    oof = np.zeros(len(train))

    for tr_idx, val_idx in tqdm(kf.split(train, groups = train['group']), total = folds):

        tr_x, tr_y = train[features].iloc[tr_idx], train[target].iloc[tr_idx]

        vl_x, vl_y = train[features].iloc[val_idx], train[target].iloc[val_idx]

        #tr_data = lgb.Dataset(tr_x, label = tr_y,  categorical_feature = categorical)

        #vl_data = lgb.Dataset(vl_x, label = vl_y,  categorical_feature = categorical)

        clf = HistGradientBoostingRegressor(learning_rate = 0.1, max_iter=num_rounds, n_iter_no_change=60, verbose = 25)

        clf.fit(tr_x, tr_y)

        models.append(clf)

        oof[val_idx] = clf.predict(vl_x)

        gc.collect()

    score = np.sqrt(metrics.mean_squared_error(train[target], np.clip(oof, a_min=0, a_max=None)))

    print('Our oof cv is :', score)

    return models

models = run_hgbr(train_df)
# read test

test_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

row_ids = test_df["row_id"]

test_df.drop("row_id", axis=1, inplace=True)

test_df = reduce_mem_usage(test_df)



# merge with building info

test_df = test_df.merge(building_df,left_on='building_id',right_on='building_id',how='left')

del building_df

gc.collect()



# fill test weather data

weather_df = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

weather_df = fill_weather_dataset(weather_df)

weather_df = reduce_mem_usage(weather_df)



# merge weather data

test_df = test_df.merge(weather_df,how='left',on=['timestamp','site_id'])

del weather_df

gc.collect()



# feature engineering

test_df = features_engineering(test_df)
def predictions(models, iterations = 50):

    # split test data into batches

    set_size = len(test_df)

    batch_size = set_size // iterations

    meter_reading = []

    for i in tqdm(range(iterations)):

        pos = i*batch_size

        fold_preds = [np.expm1(model.predict(test_df[features].iloc[pos : pos+batch_size])) for model in models]

        meter_reading.extend(np.mean(fold_preds, axis=0))



    print(len(meter_reading))

    assert len(meter_reading) == set_size

    submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

    submission['meter_reading'] = np.clip(meter_reading, a_min=0, a_max=None) # clip min at zero

    submission.to_csv('fe2_lgbm.csv', index=False)

    print('We are done!')

predictions(models)