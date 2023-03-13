# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
operation_mode = 'final';

# operation_mode = 'validation';
from sklearn.base import TransformerMixin, BaseEstimator

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_log_error
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv');

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv');

df_train.shape, df_test.shape
df_train
df_train['Date'] = pd.to_datetime(df_train['Date'], format = '%Y-%m-%d');

df_test['Date'] = pd.to_datetime(df_test['Date'], format = '%Y-%m-%d');
# Combine Country and Province

def combine_country_province(df):

    df.loc[:,'Province_State'] = df['Province_State'].fillna("")

    df.loc[:,'Region'] = df['Country_Region'] + " " + df['Province_State']

    df.loc[:,'Region'] = df.loc[:,'Region'].str.strip();

    return df;

    

df_train = combine_country_province(df_train);

df_test = combine_country_province(df_test);

df_train
df_test
# A transformer which will give the number of days as integer for ML methods to work efficiently.

class Days_Since_P0_World(BaseEstimator, TransformerMixin):  

    """Add num of days column based on date column , since a integer column will fit Data Techniques better.

    """

    def __init__(self):

        self.p_zero_date = None;

        self.col_name = 'days_since_p0_world';

        pass



    def fit(self, X, y=None ):

        self.p_zero_date = X['Date'].min()

        return self;

    

    def transform(self, X ):

        X[self.col_name] = X['Date']  -  self.p_zero_date;

        X.loc[:,self.col_name] = X[self.col_name].dt.days;

        return X;
days_since_p0_world = Days_Since_P0_World()

df_train = days_since_p0_world.fit_transform(df_train)

df_test = days_since_p0_world.transform(df_test)

df_train
# A transformer which sets Day 0 to when the first patient was discovered in the region

class Days_Since_P0_Country(BaseEstimator, TransformerMixin):  

    """A transformer which sets Day 0 to when the first patient was discovered in the COUNTRY.

    DOES NOT DROP THE ROWS , RETURNS FULL DATA.

    

    gets the min date for P1 to appear in train data set and calculates difference to this date.

    """



    def __init__(self, y_col_name = 'ConfirmedCases'):

        self.p_zero_date = {};

        self.col_name = 'days_since_p0_country';

        self.y_col_name = y_col_name;

        pass



    def fit(self, X, y=None ):

        regions = X['Country_Region'].unique();

        for this_region in regions:

            this_region_X = X.loc[X['Country_Region'] == this_region,:];

            self.p_zero_date[this_region] = min(this_region_X.loc[this_region_X[self.y_col_name]>0,'Date']);

        return self;

    

    def transform(self, X ):

        regions = X['Country_Region'].unique();

        X[self.col_name] = 0;

        answer = pd.DataFrame();

        for this_region in regions:

            this_region_X = None; # To prevent the bugging warning message.

            this_region_X = X.loc[X['Country_Region'] == this_region,:];

            this_region_X.loc[:,self.col_name] = this_region_X['Date'] -  self.p_zero_date[this_region];

            this_region_X.loc[:,self.col_name] = this_region_X[self.col_name].dt.days;

            answer = pd.concat([answer, this_region_X], axis='index');

        return answer;
days_since_p0_country = Days_Since_P0_Country()

df_train = days_since_p0_country.fit_transform(df_train)

df_test = days_since_p0_country.transform(df_test)

df_train
# A transformer which sets Day 0 to when the first patient was discovered in the region

class Days_Since_P0_Region(BaseEstimator, TransformerMixin):  

    """A transformer which sets Day 0 to when the first patient was discovered in the region.

    DOES NOT DROP THE ROWS , RETURNS FULL DATA.

    

    gets the min date for P1 to appear in train data set and calculates difference to this date.

    """



    def __init__(self, y_col_name = 'ConfirmedCases'):

        self.p_zero_date = {};

        self.col_name = 'days_since_p0_region';

        self.y_col_name = y_col_name;

        pass



    def fit(self, X, y=None ):

        regions = X['Region'].unique();

        for this_region in regions:

            this_region_X = X.loc[X['Region'] == this_region,:];

            self.p_zero_date[this_region] = min(this_region_X.loc[this_region_X[self.y_col_name]>0,'Date']);

        return self;

    

    def transform(self, X ):

        regions = X['Region'].unique();

        X[self.col_name] = 0;

        answer = pd.DataFrame();

        for this_region in regions:

            this_region_X = None; # To prevent the bugging warning message.

            this_region_X = X.loc[X['Region'] == this_region,:];

            this_region_X.loc[:,self.col_name] = this_region_X['Date'] -  self.p_zero_date[this_region];

            this_region_X.loc[:,self.col_name] = this_region_X[self.col_name].dt.days;

            answer = pd.concat([answer, this_region_X], axis='index');

        return answer;
days_since_p0_region = Days_Since_P0_Region()

df_train = days_since_p0_region.fit_transform(df_train)

df_test = days_since_p0_region.transform(df_test)

df_train
df_train.loc[df_train['Province_State'] == 'Alabama',:]

df_train.loc[(df_train['Date'] == '2020-01-22') & (df_train['Country_Region'] == 'China'),:]

df_train.loc[(df_train['Date'] == '2020-01-23') & (df_train['Country_Region'] == 'China'),:]

# Check #4873 'China Gansu'
# regions = df_train['Region'].unique();

# regions = regions[5:15]

# x_name = 'days_since_p0_region'



# for idx, region in enumerate(regions):

#     plt.figure(idx);

#     f, ax = plt.subplots(1, 2, figsize=(20,5*1));

#     text = "*"*10+'INDEX='+str(idx)+"*"*10+"REGION <---->"+region+"*"*10

#     plt.figtext(.5,.9,text, fontsize=20, color='red', ha='center')

#     df = df_train.loc[df_train['Region'] == region,:]

#     sns.regplot(data = df, x=x_name, y='ConfirmedCases', ax=ax[0],order=3)

#     sns.regplot(data = df, x=x_name, y='Fatalities', ax=ax[1],order=3)
# from collections import Counter

# India = df_train[df_train['Region'] == 'India'];

# cc_list = list(India.loc[India['days_since_p0_country'] >= 0,'ConfirmedCases']);

# cc_rise_day_on_day = []

# print(len(cc_list))

# for idx, cc_n in enumerate(cc_list):

#     if(idx == len(cc_list)-1):

#         break;

#     cc_np1= cc_list[idx+1];

#     ratio=cc_np1/cc_n;

#     cc_rise_day_on_day.append(ratio);



# cc_rise_day_on_day

# Counter(cc_rise_day_on_day)
def rmsle(y_true, y_pred):

    return mean_squared_log_error(y_true, y_pred)**(1/2);
# PARAMS

degree = 3

# MODEL

poly = PolynomialFeatures(degree = degree, include_bias=False)

model1 = LinearRegression()

model2 = LinearRegression()
X_cols = ['days_since_p0_region','days_since_p0_country','days_since_p0_world'];

y1_col = ['ConfirmedCases']

y2_col = ['Fatalities']



all_pred_train = pd.DataFrame();

all_pred_test = pd.DataFrame();



regions = df_train['Region'].unique();



if operation_mode == 'validation':

    train_test_split_date = '2020-04-01';

    train = df_train.loc[(df_train['Date'] < train_test_split_date),:];

    test = df_train.loc[~(df_train['Date'] < train_test_split_date),:];

elif operation_mode == 'final':

    train = df_train.copy();

    test = df_test.copy();

    

# TRAIN ON ONLY NON ZEROES

# train = train.loc[train['days_since_p0_region'] >= 0,:]
for idx, region in enumerate(regions):

    scaler = StandardScaler();

    

    this_region_train = train['Region'] == region;

    this_region_test = test['Region'] == region;

    

    X0_train_iter = train.loc[this_region_train,X_cols];

    y1_train_iter = train.loc[this_region_train,y1_col];

    y2_train_iter = train.loc[this_region_train,y2_col];

    

    X0_test_iter = test.loc[this_region_test,X_cols];



    X0_train_iter = poly.fit_transform(X0_train_iter);

    X0_test_iter = poly.fit_transform(X0_test_iter);

    

    X0_train_iter = scaler.fit_transform(X0_train_iter)

    X0_test_iter = scaler.transform(X0_test_iter)

    

#     scaler_y1 = StandardScaler();

#     scaler_y1.fit_transform(y1_train_iter);

    

    model1.fit(X0_train_iter, y1_train_iter);

    y1_train_iter_pred = model1.predict(X0_train_iter);

    y1_test_iter_pred = model1.predict(X0_test_iter);



#     scaler_y2 = StandardScaler();

#     scaler_y2.fit_transform(y2_train_iter);

    

    model2.fit(X0_train_iter, y2_train_iter);

    y2_train_iter_pred = model2.predict(X0_train_iter);

    y2_test_iter_pred = model2.predict(X0_test_iter);

    

    pred_iter_train = pd.DataFrame({

        'Id': train.loc[this_region_train,'Id'],

        'ConfirmedCases': y1_train_iter_pred.reshape(-1),

        'Fatalities': y2_train_iter_pred.reshape(-1)

    })

    all_pred_train = pd.concat([all_pred_train, pred_iter_train], axis = 0);

    

    if (operation_mode == 'validation'):

        pred_iter_test = pd.DataFrame({

            'Id': test.loc[this_region_test,'Id'],

            'ConfirmedCases': y1_test_iter_pred.reshape(-1),

            'Fatalities': y2_test_iter_pred.reshape(-1)

        })

    elif operation_mode == 'final':

        pred_iter_test = pd.DataFrame({

            'ForecastId': test.loc[this_region_test,'ForecastId'],

            'ConfirmedCases': y1_test_iter_pred.reshape(-1),

            'Fatalities': y2_test_iter_pred.reshape(-1)

        })

    all_pred_test = pd.concat([all_pred_test, pred_iter_test], axis = 0);

    

print(all_pred_train)

print(all_pred_test)
all_pred_test = all_pred_test.astype('int')

all_pred_test.to_csv("submission.csv", index = False);

all_pred_test
# answer = pd.merge(df_test,all_pred_test, left_on = 'ForecastId',right_on = 'ForecastId');

# days_col = 'days_since_p0_world';



# train_max_date = df_train[days_col].max();

# test_min_date = test[days_col].min();



# for idx, region in enumerate(regions):

#     sel1 = ((df_train[days_col] >= test_min_date) & (df_train['Region'] == region));

#     to_paste = df_train.loc[sel1,['ConfirmedCases','Fatalities']].copy();

#     sel2 = ((answer[days_col] <= train_max_date) & (answer['Region'] == region))

#     if to_paste.shape != answer.loc[sel2,['ConfirmedCases','Fatalities']] .shape:

#         print(df_train.loc[sel1,:]);

#         print(answer.loc[sel2,:])

#     answer.loc[sel2,['ConfirmedCases','Fatalities']] = to_paste.loc[:,['ConfirmedCases','Fatalities']].values;
if operation_mode == 'validation':

    answer = pd.merge(df_train,all_pred_test, left_on = 'Id',right_on = 'Id');

if operation_mode == 'final':

    answer = pd.merge(df_train,all_pred_test, left_on = 'Id',right_on = 'ForecastId');



answer.loc[answer['ConfirmedCases_y'] < 0,:] = 0;

answer.loc[answer['Fatalities_y'] < 0,:] = 0;

print(rmsle(answer['ConfirmedCases_x'],answer['ConfirmedCases_y']))

print(rmsle(answer['Fatalities_x'],answer['Fatalities_y']))

# answer.loc[answer['Fatalities_y'] < 0,:]

# answer.loc[answer['Region'] == 'China Beijing']
# answer = answer.loc[:,['ForecastId','ConfirmedCases','Fatalities']]

# answer = answer.astype('int');

# answer.to_csv("submission.csv", index = False);

# answer.dtypes
model1.coef_, model1.intercept_
poly.powers_