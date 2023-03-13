import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import datetime

import xgboost as xgb

from sklearn import model_selection, preprocessing

color = sns.color_palette()


# result transform

def num2log(arr):

    return np.log(arr + 1)

# result retransform

def log2num(arr):

    return np.exp(arr) - 1

# treat date

def trans_date(arr): 

    return datetime.datetime.strptime(arr, '%Y-%m-%d').date()
train_df = pd.read_csv('../input/train.csv')

print('In trainset, there are {} rows and {} columns'.format(train_df.shape[0],train_df.shape[1]))

train_df.head()
print('check if id is unique: ' + str(len(pd.unique(train_df.id)) == train_df.shape[0]))
plt.figure(figsize=(12,8))

sns.distplot(num2log(train_df.price_doc.values), bins=100, kde=False)

plt.xlabel('price', fontsize=14)

plt.title('Distribution of log price_doc', fontsize = 18)

plt.show()
train_df['date'] = train_df['timestamp'].apply(trans_date)

train_df['year'] = train_df.date.apply(lambda x: x.year)

train_df['month_num'] = train_df.date.apply(lambda x: x.month + 12*(x.year-2011))

train_df['month'] = train_df.date.apply(lambda x: x.month)

start_date = train_df.date.min()

train_df['diff_date'] = train_df['date'].apply(lambda x: (x - start_date).days)

train_df['log_price'] = num2log(train_df.price_doc.values)
month_num_count = train_df.groupby('month_num').count().reset_index()

month_num_count = train_df.groupby('month_num').size().reset_index()

month_num_count.columns = ['month_num', 'Count']

plt.figure(figsize=(12,8))

sns.boxplot(x = 'month_num', y = 'log_price', data = train_df)

plt.ylabel('Log Price', fontsize=12)

plt.xlabel('Month', fontsize=12)

plt.title('Boxplot of log_price on Month # from 2011-01', fontsize = 18)

plt.show()

plt.figure(figsize=(12,8))

sns.distplot(month_num_count.Count.values, bins=50, kde=False)

plt.xlabel('count of samples per month_num', fontsize=14)

plt.title('Distribution of count', fontsize = 18)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x = 'month', y = 'log_price', data = train_df)

plt.ylabel('Log Price', fontsize=12)

plt.xlabel('Month', fontsize=12)

plt.title('Boxplot of log_price on Month', fontsize = 18)

plt.show()
date_count = train_df.groupby('date').count().reset_index()

date_count = train_df.groupby('date').size().reset_index()

date_count.columns = ['date', 'Count']

plt.figure(figsize=(12,8))

sns.boxplot(x = 'date', y = 'log_price', data = train_df)

plt.ylabel('Log Price', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.title('Boxplot of log_price on Date', fontsize = 18)

plt.show()

plt.figure(figsize=(12,8))

sns.distplot(date_count.Count.values, bins=100, kde=False)

plt.xlabel('count of samples per date', fontsize=14)

plt.title('Distribution of count', fontsize = 18)

plt.show()
missing = train_df.isnull().sum(0).reset_index()

missing.columns = ['column', 'count']

missing = missing.sort_values(by = 'count', ascending = False).loc[missing['count'] > 0]

missing['percentage'] = missing['count'] / float(train_df.shape[0]) * 100

ind = np.arange(missing.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(10,18))

rects = ax.barh(ind, missing.percentage.values, color='r')

ax.set_yticks(ind)

ax.set_yticklabels(missing.column.values, rotation='horizontal')

ax.set_xlabel("Precentage of missing values %", fontsize = 14)

ax.set_title("Number of missing values in each column", fontsize = 18)

plt.show()
delete_col = missing.loc[missing['percentage'] >= 20].column.values

train_set = train_df.drop(delete_col, axis=1)
for f in train_set.columns:

    if train_set[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_set[f].values)) 

        train_set[f] = lbl.transform(list(train_set[f].values))

dtype_df = train_set.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df.groupby("Column Type").aggregate('count').reset_index()
corrmat = train_set.drop(["id", "timestamp", "price_doc"], axis=1).corr(method='pearson', min_periods=1000)

# plot absolute values

corrmat = np.abs(corrmat)

sns.set(context="paper", font="monospace")

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True, xticklabels = False, yticklabels = False)
remain_num = 50

corr_target = corrmat['log_price'].reset_index()[:-2]

corr_target.columns = ['feature','abs_corr']

corr_target = corr_target.sort_values(by = 'abs_corr', ascending = True)[:remain_num].loc[corr_target['abs_corr'] >0.01]

ind = np.arange(corr_target.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(10,18))

rects = ax.barh(ind, corr_target.abs_corr.values, color='r')

ax.set_yticks(ind)

ax.set_yticklabels(corr_target.feature.values, rotation='horizontal')

ax.set_xlabel("absolute corr", fontsize = 14)

ax.set_title("Correlations between features and log_price ", fontsize = 18)

plt.show()
corr_target_f = list(corr_target.feature.values)
corr_target_f2 = corr_target_f

corr_target_f2.append('log_price')

high_corr = train_set[corr_target_f2].corr(method='pearson', min_periods=1000)

high_corr = np.abs(high_corr)*100

f, ax = plt.subplots(figsize=(11, 11))

sns.heatmap(high_corr, cbar=False, annot=True, square=True, fmt='.0f', 

            annot_kws={'size': 8})

plt.title('High-corrlation Features')

plt.show()
remove_list = ['cafe_sum_5000_max_price_avg','16_29_male', 'female_f','full_all', 'male_f']

corr_target_f = [x for x in corr_target_f if x not in remove_list]
train_y = train_set.log_price.values

train_X = train_set.drop(["id", "timestamp", "price_doc", "log_price"], axis=1)

xgb_params = {

    'eta': 0.05,

    'max_depth': 10,

    'subsample': 1.0,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}

dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

remain_num = 99
fig, ax = plt.subplots(figsize=(10,18))

xgb.plot_importance(model, max_num_features=remain_num, height=0.8, ax=ax)

plt.show()
importance = model.get_score(importance_type='weight')

tuples = sorted([(k, importance[k]) for k in importance], key=lambda x: x[1], reverse=True)[:remain_num]

xgb_imp_f = [x[0] for x in tuples]