import numpy as np

import pandas as pd

import gc

import os



from os import path



pd.set_option("max_columns", 500)

def create_d_n_feature(df):

    data_dict = {0: [ 10, 11, 12, 13, 14, 15, 16], 1: [0,1,2,3,4,5, 22, 23],

                 2:[6,  7,  8,  9,17,18, 19, 20, 21]}

    day_night_hrs_dict = {value:key for key in data_dict for value in data_dict[key]}

    df["d_n"]=df['timestamp'].dt.hour.map(day_night_hrs_dict)

    df.d_n = df.d_n.astype(np.int8)

    return df



def create_date_feature(df):

    df['date'] = df.timestamp.dt.date

    return df



def create_max_min_by_day_night(df):

    df_temp = df.groupby(["building_id","date","d_n"]).agg(max_at =("air_temperature", "max"),

                                                                 min_at =("air_temperature", "min"),

                                                                 mean_at=("air_temperature", "mean"),

                                                                 max_dt =("dew_temperature", "max"),

                                                                 min_dt =("dew_temperature", "min"),

                                                                 mean_dt =("dew_temperature", "mean") ).reset_index()



    df = df.merge(df_temp, on=["building_id","date","d_n"], how='left')



    del df_temp

    df = df.drop(columns='date')

    

    gc.collect()

    

    return df





def create_lag_features(df, window):

    """

    Creating lag-based features looking back in time.

    """

    df = df.sort_values(['site_id','timestamp'])

    

    feature_cols = ["air_temperature", "dew_temperature"]

    df_site = df.groupby("site_id")



    df_rolled = df_site[feature_cols].rolling(window=window, min_periods=0)



    df_mean = df_rolled.mean().reset_index()

    df_median = df_rolled.median().reset_index()

    df_min = df_rolled.min().reset_index()

    df_max = df_rolled.max().reset_index()



    for feature in feature_cols:

        df[f"{feature}_mean_lag{window}"] = df_mean[feature]

        df[f"{feature}_median_lag{window}"] = df_median[feature]

        df[f"{feature}_min_lag{window}"] = df_min[feature]

        df[f"{feature}_max_lag{window}"] = df_max[feature]

        

    del df_mean

    del df_median

    del df_min

    del df_max

    del df_rolled

    

    gc.collect()

        

    return df







def fill_medians_with_filling_value(df, fill_value=-1):

    for col in df.columns:

        if col.startswith('had'):

            df.loc[df[col[len('had_'):]] == 0, col] = fill_value

            

    return df
def compress_dataframe(df):

    result = df.copy()

    for col in result.columns:

        col_data = result[col]

        dn = col_data.dtype.name

        if dn == "object":

            result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="integer")

        elif dn == "bool":

            result[col] = col_data.astype("int8")

        elif dn.startswith("int") or (col_data.round() == col_data).all():

            result[col] = pd.to_numeric(col_data, downcast="integer")

        else:

            result[col] = pd.to_numeric(col_data, downcast='float')

    return result
def make_is_bad_zero(Xy_subset, min_interval=48, summer_start=3000, summer_end=7500):

    """Helper routine for 'find_bad_zeros'.

    

    This operates upon a single dataframe produced by 'groupby'. We expect an 

    additional column 'meter_id' which is a duplicate of 'meter' because groupby 

    eliminates the original one."""

    meter = Xy_subset.meter_id.iloc[0]

    is_zero = Xy_subset.meter_reading == 0

    if meter == 0:

        # Electrical meters should never be zero. Keep all zero-readings in this table so that

        # they will all be dropped in the train set.

        return is_zero



    transitions = (is_zero != is_zero.shift(1))

    all_sequence_ids = transitions.cumsum()

    ids = all_sequence_ids[is_zero].rename("ids")

    if meter in [2, 3]:

        # It's normal for steam and hotwater to be turned off during the summer

        keep = set(ids[(Xy_subset.timestamp < summer_start) |

                       (Xy_subset.timestamp > summer_end)].unique())

        is_bad = ids.isin(keep) & (ids.map(ids.value_counts()) >= min_interval)

    elif meter == 1:

        time_ids = ids.to_frame().join(Xy_subset.timestamp).set_index("timestamp").ids

        is_bad = ids.map(ids.value_counts()) >= min_interval



        # Cold water may be turned off during the winter

        jan_id = time_ids.get(0, False)

        dec_id = time_ids.get(8283, False)

        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and

                dec_id == time_ids.get(8783, False)):

            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))

    else:

        raise Exception(f"Unexpected meter type: {meter}")



    result = is_zero.copy()

    result.update(is_bad)

    return result



def find_bad_zeros(X, y):

    """Returns an Index object containing only the rows which should be deleted."""

    Xy = X.assign(meter_reading=y, meter_id=X.meter)

    is_bad_zero = Xy.groupby(["building_id", "meter"]).apply(make_is_bad_zero)

    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])



def find_bad_sitezero(X):

    """Returns indices of bad rows from the early days of Site 0 (UCF)."""

    return X[(X.timestamp < 3378) & (X.site_id == 0) & (X.meter == 0)].index



def find_bad_building1099(X, y):

    """Returns indices of bad rows (with absurdly high readings) from building 1099."""

    return X[(X.building_id == 1099) & (X.meter == 2) & (y > 3e4)].index



def find_bad_rows(X, y):

    return find_bad_zeros(X, y).union(find_bad_sitezero(X)).union(find_bad_building1099(X, y))
def read_building_metadata(nan_filling=-1):

    return compress_dataframe(pd.read_csv(

        input_file("building_metadata.csv")).fillna(nan_filling)).set_index("building_id")
site_GMT_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]



def read_weather_train(fix_timestamps=True, interpolate_na=True, add_na_indicators=True, create_rolling_features=True):

    df = pd.read_csv(input_file("weather_train.csv"), parse_dates=["timestamp"])

    

    

    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    if fix_timestamps:

        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}

        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)

    if interpolate_na:

        site_dfs = []

        for site_id in df.site_id.unique():

            # Make sure that we include all possible hours so that we can interpolate evenly

            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784))

            site_df.site_id = site_id

            for col in [c for c in site_df.columns if c != "site_id"]:

                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()

                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')

                # Some sites are completely missing some columns, so use this fallback

                site_df[col] = site_df[col].fillna(df[col].median())

            site_dfs.append(site_df)

        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column

    elif add_na_indicators:

        for col in df.columns:

            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()

    

    if create_rolling_features:

        df = create_lag_features(df, window=24)

        

    df = compress_dataframe(df)

    

    return df.set_index(["site_id", "timestamp"])





def read_weather_test(fix_timestamps=True, interpolate_na=True, add_na_indicators=True, create_rolling_features=True):

    df = pd.read_csv(input_file("weather_test.csv"), parse_dates=["timestamp"])

    

    

    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    if fix_timestamps:

        GMT_offset_map = {site: offset for site, offset in enumerate(site_GMT_offsets)}

        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)

    if interpolate_na:

        site_dfs = []

        for site_id in df.site_id.unique():

            # Make sure that we include all possible hours so that we can interpolate evenly

            site_df = df[df.site_id == site_id].set_index("timestamp").reindex(range(8784, 26304))

            site_df.site_id = site_id

            for col in [c for c in site_df.columns if c != "site_id"]:

                if add_na_indicators: site_df[f"had_{col}"] = ~site_df[col].isna()

                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')

                # Some sites are completely missing some columns, so use this fallback

                site_df[col] = site_df[col].fillna(df[col].median())

            site_dfs.append(site_df)

        df = pd.concat(site_dfs).reset_index()  # make timestamp back into a regular column

    elif add_na_indicators:

        for col in df.columns:

            if df[col].isna().any(): df[f"had_{col}"] = ~df[col].isna()

                

    if create_rolling_features:

        df = create_lag_features(df, window=24)

    df = compress_dataframe(df)

    

    return df.set_index(["site_id", "timestamp"])
def _add_time_features(X):

    return X.assign(tm_day_of_week=((X.timestamp // 24) % 7), tm_hour_of_day=(X.timestamp % 24))
def input_file(file):

    path = f"../input/ashrae-energy-prediction/{file}"

    if not os.path.exists(path): return path + ".gz"

    return path
def read_train(df):    

    df = create_d_n_feature(df)

    df = create_date_feature(df)

    

    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    return compress_dataframe(df)



def combined_train_data(df, fix_timestamps=True, interpolate_na=True, add_na_indicators=True, create_rolling_features=True):

    Xy = compress_dataframe(read_train(df).join(read_building_metadata(), on="building_id").join(

        read_weather_train(fix_timestamps, interpolate_na, add_na_indicators, create_rolling_features=create_rolling_features),

        on=["site_id", "timestamp"]).fillna(-1))

    return Xy.drop(columns=["meter_reading"]), Xy.meter_reading
def read_test(df): 

    df = create_d_n_feature(df)

    df = create_date_feature(df)

    

    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600

    return compress_dataframe(df).set_index("row_id")



def combined_test_data(df, fix_timestamps=True, interpolate_na=True, add_na_indicators=True, create_rolling_features=True):

    X = compress_dataframe(read_test(df).join(read_building_metadata(), on="building_id").join(

        read_weather_test(fix_timestamps, interpolate_na, add_na_indicators, create_rolling_features=create_rolling_features),

        on=["site_id", "timestamp"]).fillna(-1))

    return X
NUM_FOLDS = 5



from sklearn.model_selection import StratifiedKFold





def create_stratify_kfold(df):

    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    df['k_folds'] = 0

    

    for idx, (train_idx, test_idx) in enumerate(kf.split(df, df['building_id'])):

        df['k_folds'].iloc[test_idx] = idx

        

    return df
cat_columns = [

    "building_id", "meter", "site_id", "primary_use", "had_air_temperature", "had_cloud_coverage",

    "had_dew_temperature", "had_precip_depth_1_hr", "had_sea_level_pressure", "had_wind_direction",

    "had_wind_speed", "tm_day_of_week", "tm_hour_of_day", "d_n"

]
X_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv', parse_dates=['timestamp'])

    

X_test = combined_test_data(X_test)

X_test = create_max_min_by_day_night(X_test)

X_test = compress_dataframe(_add_time_features(X_test))



X_test = X_test.drop(columns="timestamp") 

X_test['row_id'] = X_test.index



gc.collect()



X_test.to_parquet('X_test.parquet.gzip', compression='gzip')



del X_test

gc.collect()



print('Test ready!')
leak_df = pd.read_csv('/kaggle/input/leakaggregator/leak_df.csv', parse_dates=['timestamp'])

X_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv', parse_dates=['timestamp'])





X_train = pd.concat([X_train, leak_df[leak_df['timestamp'].dt.year == 2016]], axis=0).drop_duplicates(['timestamp','building_id', 'meter']).reset_index()



del leak_df

gc.collect()



X_train, y_train = combined_train_data(X_train)

X_train = create_max_min_by_day_night(X_train)



bad_rows = find_bad_rows(X_train, y_train)



X_train = X_train.drop(index=bad_rows)

y_train = y_train.reindex_like(X_train)



# Additional preprocessing

X_train = compress_dataframe(_add_time_features(X_train))



X_train = X_train.drop(columns="timestamp")  # Raw timestamp doesn't help when prediction

y_train = np.log1p(y_train)



X_train['meter_reading'] = y_train



del y_train

gc.collect()



X_train = create_stratify_kfold(X_train)



gc.collect()



X_train.to_parquet('X_train.parquet.gzip', compression='gzip')



del X_train

gc.collect()



print('Train ready!')