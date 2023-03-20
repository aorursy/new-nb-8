import numpy as np

import pandas as pd

import os, sys, gc, time, warnings, pickle, psutil, random

from math import ceil

from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')

pd.set_option("display.max_columns", 500)
#メモリの使用量を調べる。

def get_memory_usage():

    process_id   = os.getpid()                #自分のプロセス番号

    process      = psutil.Process(process_id) #プロセスを指定

    memory_usage = process.memory_info()[0]/2. ** 30

    memory_usage = np.round(memory_usage, 2)

    return memory_usage





#メモリのサイズを調整する。

def sizeof_fmt(num, suffix='B'):

    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:

        if abs(num) < 1024.0:

            return "{:3.1f}{unit}{suffix}".format(num, unit=unit, suffix=suffix)

        num /=  1024.0

    return "{:.1f}{unit}{suffix}".format(num, unit='Yi', suffix=suffix)





#メモリのサイズを節約する。

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

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f} % reduction)'.format(end_mem, (start_mem - end_mem) * 100 / start_mem))

    return df
# 2つのdfを繋げる。

def merge_by_concat(df1, df2, merge_on):

    merged_gf = df1[merge_on]

    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')

    new_columns = [col for col in list(merged_gf) if col not in merge_on]

    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)

    return df1
TARGET = 'sales'         

END_TRAIN = 1913         

MAIN_INDEX = ['id','d'] 
# データの読み込み

train_df    = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')

prices_df   = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

calendar_df = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
# train_dfをmeltする。

index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

grid_df = pd.melt(train_df, 

                  id_vars = index_columns, 

                  var_name = 'd', 

                  value_name = TARGET)

print('Train rows: {} to {} '.format(len(train_df), len(grid_df)))





# predict用の、1914日目から1942日目のrowを追加。

add_grid = pd.DataFrame()

for i in range(1,29):

    temp_df = train_df[index_columns]

    temp_df = temp_df.drop_duplicates()

    temp_df['d'] = 'd_'+ str(END_TRAIN+i)

    temp_df[TARGET] = np.nan

    add_grid = pd.concat([add_grid,temp_df])



grid_df = pd.concat([grid_df,add_grid])

grid_df = grid_df.reset_index(drop=True)





# メモリの節約のため、①今後使わないdfを削除。②columnsのtype: 'object' → 'category'に変更

del temp_df, add_grid, train_df



print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

for col in index_columns:

    grid_df[col] = grid_df[col].astype('category')

print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
# 各店舗での各商品の発売日(release)を調べる。

release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()

release_df.columns = ['store_id', 'item_id', 'release']

grid_df = merge_by_concat(grid_df, release_df, ['store_id','item_id'])

del release_df





# releaseより前の情報は必要ないので、削除する。

grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])

grid_df = grid_df[grid_df['wm_yr_wk']>=grid_df['release']]

grid_df = grid_df.reset_index(drop=True)





# releaseカラムは、差分だけ保持することで、メモリを節約する。

print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

grid_df['release'] = grid_df['release'] - grid_df['release'].min()

grid_df['release'] = grid_df['release'].astype(np.int16)

print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
grid_df.to_pickle('grid_part_1.pkl')

print('Size:', grid_df.shape)
prices_df['price_max']     = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')

prices_df['price_min']     = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')

prices_df['price_std']     = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')

prices_df['price_mean']    = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')

prices_df['price_norm']    = prices_df['sell_price']/prices_df['price_max']

prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')

prices_df['item_nunique']  = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')



# 瞬間の、月毎の、年毎の、価格の変動を特徴量として加えたい。そのために、calendar_dfから、month, yearを持ってくる。

calendar_prices = calendar_df[['wm_yr_wk','month','year']]

calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])

prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')

del calendar_prices



prices_df['price_momentum']   = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))

prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')

prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')



del prices_df['month'], prices_df['year']
original_columns = list(grid_df)

grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')

keep_columns = [col for col in list(grid_df) if col not in original_columns]

grid_df = grid_df[MAIN_INDEX + keep_columns]

grid_df = reduce_mem_usage(grid_df)

grid_df.to_pickle('grid_part_2.pkl')

print('Size:', grid_df.shape)

del prices_df
grid_df = pd.read_pickle('grid_part_1.pkl')
grid_df = grid_df[MAIN_INDEX]





# merge

icols = ['date',

         'd',

         'event_name_1',

         'event_type_1',

         'event_name_2',

         'event_type_2',

         'snap_CA',

         'snap_TX',

         'snap_WI']

grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')





# メモリの節約

icols = ['event_name_1',

         'event_type_1',

         'event_name_2',

         'event_type_2',

         'snap_CA',

         'snap_TX',

         'snap_WI']

for col in icols:

    grid_df[col] = grid_df[col].astype('category')

    

    

# 新しい特徴量の作成

grid_df['date']  = pd.to_datetime(grid_df['date'])

grid_df['tm_d']  = grid_df['date'].dt.day.astype(np.int8)

grid_df['tm_w']  = grid_df['date'].dt.week.astype(np.int8)

grid_df['tm_m']  = grid_df['date'].dt.month.astype(np.int8)

grid_df['tm_y']  = grid_df['date'].dt.year

grid_df['tm_y']  = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)

grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x/7)).astype(np.int8) #月の何周目か。7日:1周目、8日:２周目

grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8) #曜日。月曜日:0、日曜日:6

grid_df['tm_w_end'] = (grid_df['tm_dw']>=5).astype(np.int8) #週末かどうか。祝日も加えたい。



del grid_df['date']  
grid_df.to_pickle('grid_part_3.pkl')

print('Size:', grid_df.shape)



del calendar_df, grid_df
# パート1のメモリをさらに減らす。

grid_df = pd.read_pickle('grid_part_1.pkl')

grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)



del grid_df['wm_yr_wk']

grid_df.to_pickle('grid_part_1.pkl')



del grid_df
grid_df = pd.concat([pd.read_pickle('grid_part_1.pkl'),

                     pd.read_pickle('grid_part_2.pkl').iloc[:,2:],

                     pd.read_pickle('grid_part_3.pkl').iloc[:,2:]],

                     axis=1)



print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

print('Size', grid_df.shape)
# 2.4GBは学習に扱うのに大きすぎる。

# state_id, shop_idで区切って、学習すれば、よりメモリを減らせる。



state_id = 'CA'

grid_df = grid_df[grid_df['state_id']==state_id]

print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))



store_id = 'CA_1'

grid_df = grid_df[grid_df['store_id']==store_id]

print("{:>20}: {:>8}".format('Full Grid',sizeof_fmt(grid_df.memory_usage(index=True).sum())))
grid_df.head()
grid_df.info()