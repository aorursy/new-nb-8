# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt


import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

import scipy.optimize as opt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
train.head()
train.describe()
print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())
confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_noChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_noChina.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases excluding China", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
plt.figure(figsize=(20,6))

sns.barplot(x='Province_State', y='ConfirmedCases', data=train)

plt.ylim(0, 3000)

plt.xticks(rotation=90)



plt.figure(figsize=(20,6))

sns.barplot(x='Province_State', y='Fatalities', data=train)

plt.ylim(0, 100)

plt.xticks(rotation=90)
plt.figure(figsize=(20,6))

sns.barplot(x='Country_Region', y='ConfirmedCases', data=train)

plt.ylim(0, 3000)

plt.xticks(rotation=90)



plt.figure(figsize=(20,6))

sns.barplot(x='Country_Region', y='Fatalities', data=train)

plt.ylim(0, 100)

plt.xticks(rotation=90)
plt.figure(figsize=(20,6))

sns.barplot(x='Date', y='ConfirmedCases', data=train)

plt.ylim(0, 3000)

plt.xticks(rotation=90)



plt.figure(figsize=(20,6))

sns.barplot(x='Date', y='Fatalities', data=train)

plt.xticks(rotation=90)
(train['Fatalities'].sum() / train['ConfirmedCases'].sum()) * 100
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state



train['Province_State'].fillna(EMPTY_VAL, inplace=True)

train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

test['Province_State'].fillna(EMPTY_VAL, inplace=True)

test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fillState(x['Province_State'], x['Country_Region']), axis=1)

test.head()
train['row_number'] = train.groupby(['Country_Region', 'Province_State']).cumcount()

x = train[train["Country_Region"] == 'China'][train["Province_State"] == 'Hubei']['row_number']

y = train[train["Country_Region"] == 'China'][train["Province_State"] == 'Hubei']['ConfirmedCases']

y_ = train[train["Country_Region"] == 'China'][train["Province_State"] == 'Hubei']['Fatalities']



def f(x, L, b, k, x_0):

    return L / (1. + np.exp(-k * (x - x_0))) + b





def logistic(xs, L, k, x_0):

    result = []

    for x in xs:

        xp = k*(x-x_0)

        if xp >= 0:

            result.append(L / ( 1. + np.exp(-xp) ) )

        else:

            result.append(L * np.exp(xp) / ( 1. + np.exp(xp) ) )

    return result



p0 = [max(y), 0.0,max(x)]

p0_ = [max(y_), 0.0,max(x)]

x_ = np.arange(0, 100, 1).tolist()

try:

    popt, pcov = opt.curve_fit(logistic, x, y,p0)

    yfit = logistic(x_, *popt)

    popt_, pcov_ = opt.curve_fit(logistic, x, y_,p0_)

    yfit_ = logistic(x_, *popt_)

except:

    popt, pcov = opt.curve_fit(f, x, y, method="lm", maxfev=5000)

    yfit = f(x_, *popt)

    popt_, pcov_ = opt.curve_fit(f, x, y_, method="lm", maxfev=5000)

    yfit_ = f(x_, *popt_)

    #print("problem")





fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.plot(x, y, 'o', label ='Actual Cases')

ax.plot(x_, yfit, '-', label ='Fitted Cases')



ax.plot(x, y_, 'o', label ='Actual Fatalities')

ax.plot(x_, yfit_, '-', label ='Fitted fatalities')

ax.title.set_text('China - Hubei province')

plt.legend(loc="center right")

plt.show()
unique = pd.DataFrame(train.groupby(['Country_Region', 'Province_State'],as_index=False).count())

unique.head()
import datetime as dt



def date_day_diff(d1, d2):

    delta = dt.datetime.strptime(d1, "%Y-%m-%d") - dt.datetime.strptime(d2, "%Y-%m-%d")

    return delta.days



log_regions = []



for index, region in unique.iterrows():

    st = region['Province_State']

    co = region['Country_Region']

    

    rdata = train[(train['Province_State']==st) & (train['Country_Region']==co)]



    t = rdata['Date'].values

    t = [float(date_day_diff(d, t[0])) for d in t]

    y = rdata['ConfirmedCases'].values

    y_ = rdata['Fatalities'].values



    p0 = [max(y), 0.0, max(t)]

    p0_ = [max(y_), 0.0, max(t)]

    try:

        popt, pcov = opt.curve_fit(logistic, t, y, p0, maxfev=10000)

        try:

            popt_, pcov_ = opt.curve_fit(logistic, t, y_, p0_, maxfev=10000)

        except:

            popt_, pcov_ = opt.curve_fit(f, t, y_,method="trf", maxfev=10000)

        log_regions.append((co,st,popt,popt_))

    except:

        popt, pcov = opt.curve_fit(f, t, y,method="trf", maxfev=10000)

        popt_, pcov_ = opt.curve_fit(f, t, y_,method="trf", maxfev=10000)

        log_regions.append((co,st,popt,popt_))
log_regions = pd.DataFrame(log_regions)

log_regions.head()
log_regions.columns = ['Country_Region','Province_State','ConfirmedCases','Fatalities']

log_regions.head(1)
data = log_regions['ConfirmedCases'].str[1]

bins = np.arange(0, 10, 0.01)

plt.hist(data,bins=bins, alpha=0.5)

plt.xlim([0,1])

plt.ylabel('count')

plt.show()

log_regions['ConfirmedCases'].str[1].quantile([.1, .25, .5, .75, .95, .99])
T = np.arange(0, 100, 1).tolist()

popt = list(log_regions[log_regions["Country_Region"] == 'Italy'][log_regions["Province_State"] == 'Italy']['ConfirmedCases'])[0]

popt_ = list(log_regions[log_regions["Country_Region"] == 'Italy'][log_regions["Province_State"] == 'Italy']['Fatalities'])[0]



try:

    yfit = logistic(T, *popt)

    yfit_ = logistic(T, *popt_)

except:

    yfit = f(T, *popt)

    yfit_ = f(T, *popt_)

    



fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.plot(T, yfit, label="Fitted ConfirmedCases")

ax.plot(T, yfit_, label="Fitted Fatalities")

ax.title.set_text('Italy fitted params')

plt.legend(loc="upper left")

plt.show()
for index, rt in log_regions.iterrows():

    st = rt['Province_State']

    co = rt['Country_Region']

    popt = list(['ConfirmedCases'])

    popt_ = list(rt['Fatalities'])

    print(co,st,popt,popt_)
data0 = log_regions['Fatalities'].str[0]/log_regions['ConfirmedCases'].str[0]

data1 = log_regions['Fatalities'].str[1]/log_regions['ConfirmedCases'].str[1]

data2 = log_regions['Fatalities'].str[2]/log_regions['ConfirmedCases'].str[2]

bins = np.arange(0, 3, 0.01)

plt.hist(data,bins=bins, alpha=0.5)

plt.xlim([0,3])

plt.ylabel('count')

plt.ylim([0,5])

plt.show()

fp = np.array([data0.median(),data1.median(),data2.median()])

fp
for index, rt in log_regions.iterrows():

    st = rt['Province_State']

    co = rt['Country_Region']

    popt = list(rt['ConfirmedCases'])

    popt_ = list(rt['Fatalities'])

    

    if popt_ == [0.0,0.0,69.0]:

        popt_ = np.multiply(fp,popt)

        print(co,st,popt,popt_)
submission = []



for index, rt in log_regions.iterrows():

    st = rt['Province_State']

    co = rt['Country_Region']

    popt = list(rt['ConfirmedCases'])

    popt_ = list(rt['Fatalities'])

    if popt_ == [0.0,0.0,69.0]:

        popt_ = np.multiply(fp,popt)

    print(co,st,popt,popt_)

    rtest = test[(test['Province_State']==st) & (test['Country_Region']==co)]

    for index, rt in rtest.iterrows():

        try:

            tdate = rt['Date']

            ca = logistic([date_day_diff(tdate, min(train[(train['Province_State']==st) & (train['Country_Region']==co)]['Date'].values))], *popt)

            try:

                fa = logistic([date_day_diff(tdate, min(train[(train['Province_State']==st) & (train['Country_Region']==co)]['Date'].values))], *popt_)

            except:

                fa = f([date_day_diff(tdate, min(train[(train['Province_State']==st) & (train['Country_Region']==co)]['Date'].values))], *popt_)

            submission.append((rt['ForecastId'], int(ca[0]), int(fa[0])))

        except:

            tdate = rt['Date']

            ca = f([date_day_diff(tdate, min(train[(train['Province_State']==st) & (train['Country_Region']==co)]['Date'].values))], *popt)

            fa = f([date_day_diff(tdate, min(train[(train['Province_State']==st) & (train['Country_Region']==co)]['Date'].values))], *popt_)

            submission.append((rt['ForecastId'], int(ca[0]), int(fa[0])))

submission = pd.DataFrame(submission)

submission.columns = ['ForecastId','ConfirmedCases','Fatalities']

submission.to_csv('./submission.csv', index = False)