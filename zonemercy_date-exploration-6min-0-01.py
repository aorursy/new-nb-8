
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



train_date_part = pd.read_csv('../input/train_date.csv', nrows=10000)

print(train_date_part.shape)

print(1.0 * train_date_part.count().sum() / train_date_part.size)

print(train_date_part[:2])
# Let's check the min and max times for each station

def get_station_times(dates, withId=False):

    times = []

    cols = list(dates.columns)

    if 'Id' in cols:

        cols.remove('Id')

    for feature_name in cols:

        if withId:

            df = dates[['Id', feature_name]].copy()

            df.columns = ['Id', 'time']

        else:

            df = dates[[feature_name]].copy()

            df.columns = ['time']

        df['station'] = feature_name.split('_')[1][1:]

        df = df.dropna()

        times.append(df)

    return pd.concat(times)



station_times = get_station_times(train_date_part, withId=True).sort_values(by=['Id', 'station'])

#print(station_times[:5])

#print(station_times.shape)

min_station_times = station_times.groupby(['Id', 'station']).min()['time']

max_station_times = station_times.groupby(['Id', 'station']).max()['time']

print(np.mean(1. * (min_station_times == max_station_times)))
resp = pd.read_csv('../input/train_numeric.csv', usecols=['Id','Response'])
# Read station times for train and test

date_cols = train_date_part.drop('Id', axis=1).count().reset_index().sort_values(by=0, ascending=False)

date_cols['station'] = date_cols['index'].apply(lambda s: s.split('_')[1])

date_cols = date_cols.drop_duplicates('station', keep='first')['index'].tolist()

print(date_cols) # selected features



train_date = pd.read_csv('../input/train_date.csv', usecols=['Id']+date_cols)

print(train_date.shape)

train_station_times = get_station_times(train_date, withId=True)

#print(train_station_times.shape)

#train_time_cnt = train_station_times.groupby('time').count()[['station']].reset_index()

#train_time_cnt.columns = ['time', 'cnt']

#print(train_time_cnt.shape)
train_station_times = train_station_times.merge(resp,how='left',on=['Id'])

del train_date_part, station_times

del resp, train_date
train_station_times
x1 = train_station_times.groupby('station').count()[['Response']].reset_index()

x2 = train_station_times.groupby('station').sum()[['Response']].reset_index()

x1 = x1.merge(x2,how='left',on=['station'])

x1['f'] = x1['Response_y'] / x1['Response_x']

print (x1)

plt.plot(x1.Response_x.values, x1.f.values, 'b.', alpha=0.5)

x1 = train_station_times.groupby('time').count()[['station']].reset_index()

train_station_times = train_station_times.merge(x1,how='left',on=['time'])

del x1

x1 = train_station_times.groupby('time').sum()[['Response']].reset_index()

train_station_times = train_station_times.merge(x1,how='left',on=['time'])

del x1
train_station_times['f'] = train_station_times['Response_y'] / train_station_times['station_y']

#train_station_times.sort_values(by=['f'], ascending=False)

train_station_times = train_station_times.drop_duplicates('Id', keep='first')

train_station_times.shape
week_duration = 1679

train_station_times['week_part'] = ((train_station_times['time'].values * 100) % week_duration).astype(np.int64)

train_week_part = train_station_times.groupby(['week_part'])[['Response_y','station_y']].sum().reset_index()

train_week_part['f'] = train_week_part['Response_y'] / train_week_part['station_y']

train_week_part
#fig = plt.figure()

a = plt.plot(train_station_times.Id.values, train_station_times.f.values, 'b.', alpha=0.1, label='train count')

#y_train = train_station_times['f'].rolling(window=100, center=True).mean().values

#plt.plot(train_station_times.Id.values, y_train, 'r-', linewidth=4, alpha=0.5, label='train count smooth')



#plt.xlim(0, 1680)

#plt.ylim(0, 0.05)
fig = plt.figure()

plt.plot(train_week_part.week_part.values, train_week_part.f.values, 'b.', alpha=0.5, label='train count')

y_train = train_week_part['f'].rolling(window=50, center=True).mean().values

plt.plot(train_week_part.week_part.values, y_train, 'r-', linewidth=4, alpha=0.5, label='train count smooth')



plt.xlim(0, 1680)

plt.ylim(0, 0.04)

#fig.savefig('week_duration.png', dpi=300)
fig = plt.figure()

plt.plot(train_time_cnt['time'].values, train_time_cnt['cnt'].values, 'b.', alpha=0.1, label='train')

plt.plot(test_time_cnt['time'].values, test_time_cnt['cnt'].values, 'r.', alpha=0.1, label='test')

plt.title('Original date values')

plt.ylabel('Number of records')

plt.xlabel('Time')

fig.savefig('original_date_values.png', dpi=300)

plt.show()



print((train_time_cnt['time'].min(), train_time_cnt['time'].max()))

print((test_time_cnt['time'].min(), test_time_cnt['time'].max()))
time_ticks = np.arange(train_time_cnt['time'].min(), train_time_cnt['time'].max() + 0.01, 0.01)

time_ticks = pd.DataFrame({'time': time_ticks})

time_ticks = pd.merge(time_ticks, train_time_cnt, how='left', on='time')

time_ticks = time_ticks.fillna(0)

# Autocorrelation

x = time_ticks['cnt'].values

max_lag = 8000

auto_corr_ks = range(1, max_lag)

auto_corr = np.array([1] + [np.corrcoef(x[:-k], x[k:])[0, 1] for k in auto_corr_ks])

fig = plt.figure()

plt.plot(auto_corr, 'k.', label='autocorrelation by 0.01')

plt.title('Train Sensor Time Auto-correlation')

period = 25

auto_corr_ks = list(range(period, max_lag, period))

auto_corr = np.array([1] + [np.corrcoef(x[:-k], x[k:])[0, 1] for k in auto_corr_ks])

plt.plot([0] + auto_corr_ks, auto_corr, 'go', alpha=0.5, label='strange autocorrelation at 0.25')

period = 1675

auto_corr_ks = list(range(period, max_lag, period))

auto_corr = np.array([1] + [np.corrcoef(x[:-k], x[k:])[0, 1] for k in auto_corr_ks])

plt.plot([0] + auto_corr_ks, auto_corr, 'ro', markersize=10, alpha=0.5, label='one week = 16.75?')

plt.xlabel('k * 0.01 -  autocorrelation lag')

plt.ylabel('autocorrelation')

plt.legend(loc=0)

fig.savefig('train_time_auto_correlation.png', dpi=300)
week_duration = 1679

train_time_cnt['week_part'] = ((train_time_cnt['time'].values * 100) % week_duration).astype(np.int64)

# Aggregate weekly stats

train_week_part = train_time_cnt.groupby(['week_part'])[['cnt']].sum().reset_index()

fig = plt.figure()

plt.plot(train_week_part.week_part.values, train_week_part.cnt.values, 'b.', alpha=0.5, label='train count')

y_train = train_week_part['cnt'].rolling(window=20, center=True).mean().values

plt.plot(train_week_part.week_part.values, y_train, 'b-', linewidth=4, alpha=0.5, label='train count smooth')

plt.title('Relative Part of week')

plt.ylabel('Number of records')

plt.xlim(0, 1680)

fig.savefig('week_duration.png', dpi=300)