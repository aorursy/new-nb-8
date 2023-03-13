# Data processing
import numpy as np
import pandas as pd
import datetime as dt

# Visualization libaries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Read data
train = pd.read_csv('../input/train.csv', nrows = 10_000_000)
test = pd.read_csv('../input/test.csv')
combine = [train, test]

test.dtypes
print(test.head())
print(train.head())
for dataset in combine:
    #So Fare Prices are directly influenced by Distance
    dataset['longitude_distance'] = abs(dataset['pickup_longitude'] - dataset['dropoff_longitude'])
    
    dataset['latitude_distance'] = abs(dataset['pickup_latitude'] - dataset['dropoff_latitude'])
    
    # Straight distance
    dataset['distance_travelled'] = (dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5
    dataset['distance_travelled_sin'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_cos'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5)
    dataset['distance_travelled_sin_sqrd'] = np.sin((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    dataset['distance_travelled_cos_sqrd'] = np.cos((dataset['longitude_distance'] ** 2 * dataset['latitude_distance'] ** 2) ** .5) ** 2
    # Haversine formula for distance
    # Haversine formula:	a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    # c = 2 ⋅ atan2( √a, √(1−a) )
    # d = R ⋅ c
    R = 6371e3 # Metres
    phi1 = np.radians(dataset['pickup_latitude'])
    phi2 = np.radians(dataset['dropoff_latitude'])
    phi_chg = np.radians(dataset['pickup_latitude'] - dataset['dropoff_latitude'])
    delta_chg = np.radians(dataset['pickup_longitude'] - dataset['dropoff_longitude'])
    a = np.sin(phi_chg / 2) + np.cos(phi1) * np.cos(phi2) * np.sin(delta_chg / 2)
    c = 2 * np.arctan2(a ** .5, (1-a) ** .5)
    d = R * c
    dataset['haversine'] = d
    
    # Bearing
    # Formula:	θ = atan2( sin Δλ ⋅ cos φ2 , cos φ1 ⋅ sin φ2 − sin φ1 ⋅ cos φ2 ⋅ cos Δλ )
    y = np.sin(delta_chg * np.cos(phi2))
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_chg)
    dataset['bearing'] = np.degrees(np.arctan2(y, x))
    
    # Rhumb lines
    psi_chg = np.log(np.tan(np.pi / 4 + phi2 / 2) / np.tan(np.pi / 4 + phi1 / 2))
    q = phi_chg / psi_chg
    d = (phi_chg + q ** 2 * delta_chg ** 2) ** .5 * R
    dataset['rhumb_lines'] = d
    
    # Maybe time of day matters? Obviously duration is a factor, but there is no data for time arrival
    # Features: hour of day (night vs day), month (some months may be in higher demand) 
    dataset['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])
    dataset['hour_of_day'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['week'] = dataset.pickup_datetime.dt.week
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['day_of_year'] = dataset.pickup_datetime.dt.dayofyear
    dataset['week_of_year'] = dataset.pickup_datetime.dt.weekofyear
    
train.head(3)
# Let's check how the features correlate
colormap = plt.cm.RdBu
plt.figure(figsize=(20,20))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
# Let's drop all the irrelevant features
train_features_to_keep = ['haversine', 'fare_amount']
train.drop(train.columns.difference(train_features_to_keep), 1, inplace=True)
train= train.dropna()

test_features_to_keep = ['key', 'haversine']
test.drop(test.columns.difference(test_features_to_keep), 1, inplace=True)
# Let's prepare the test set
x_pred = test.drop('key', axis=1)

# Let's run XGBoost and predict those fares!
x_train,x_test,y_train,y_test = train_test_split(train.drop('fare_amount',axis=1),train.pop('fare_amount'),random_state=123,test_size=0.2)

def XGBmodel(x_train,x_test,y_train,y_test):
    matrix_train = xgb.DMatrix(x_train,label=y_train)
    matrix_test = xgb.DMatrix(x_test,label=y_test)
    model=xgb.train(params={'objective':'reg:linear','eval_metric':'rmse'}
                    ,dtrain=matrix_train,num_boost_round=300, 
                    early_stopping_rounds=30,evals=[(matrix_test,'test')],)
    return model

model=XGBmodel(x_train,x_test,y_train,y_test)
prediction = model.predict(xgb.DMatrix(x_pred), ntree_limit = model.best_ntree_limit)
# Add to submission
submission = pd.DataFrame({
        "key": test['key'],
        "fare_amount": prediction.round(2)
})

submission.to_csv('sub_fare.csv',index=False)
