
import pandas as pd

import numpy as np

from fbprophet import Prophet



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
N = 60 # number of days for test split

i = 1800 # one example time series to train
all_data = pd.read_csv("../input/train_1.csv").T

key = pd.read_csv("../input/key_1.csv")
#Handle N/A

train, test = all_data.iloc[0:-N,:], all_data.iloc[-N:,:]



test_cleaned = test.T.fillna(method='ffill').T

train_cleaned = train.T.iloc[:,1:].fillna(method='ffill').T
#fill outliers that are out of 1.5*std with rolling median of 56 days

data=train_cleaned.iloc[:,i].to_frame()

data.columns = ['visits']

data['median'] = pd.rolling_median(data.visits,50,min_periods=1)

std_mult = 1.5

data.ix[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'visits'] = data.ix[np.abs(data.visits-data.visits.median())>=(std_mult*data.visits.std()),'median']

data.index = pd.to_datetime(data.index)



print(data.tail())
#prophet expects the folllwing label names

X = pd.DataFrame(index=range(0,len(data)))

X['ds'] = data.index

X['y'] = data['visits'].values

X.tail()
m = Prophet(yearly_seasonality=True)

m.fit(X)

future = m.make_future_dataframe(periods=N)

future.tail()
forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

m.plot(forecast);

y_truth = test_cleaned.iloc[:,i].values

y_forecasted = forecast.iloc[-N:,2].values





denominator = (np.abs(y_truth) + np.abs(y_forecasted))

diff = np.abs(y_truth - y_forecasted) / denominator

diff[denominator == 0] = 0.0

print(200 * np.mean(diff))
print(200 * np.median(diff))