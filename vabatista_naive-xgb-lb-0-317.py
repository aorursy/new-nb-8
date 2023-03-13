import numpy as np

import pandas as pd

import xgboost as xgb

import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])



df_train.head()
df_train['price_doc'].hist(bins=50)
y_train = df_train['price_doc'].values

id_test = df_test['id']



df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

df_test.drop(['id'], axis=1, inplace=True)



# Build df_all = (df_train+df_test).join(df_macro)

num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

df_all = pd.merge_ordered(df_all, df_macro, on='timestamp', how='left')

print(df_all.shape)



# Add month-year

month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)

month_year_cnt_map = month_year.value_counts().to_dict()

df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)



# Add week-year count

week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)

week_year_cnt_map = week_year.value_counts().to_dict()

df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)



# Add month and day-of-week

df_all['month'] = df_all.timestamp.dt.month

df_all['dow'] = df_all.timestamp.dt.dayofweek



# Other feature engineering

df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)

df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)



# Remove timestamp column (may overfit the model in train)

df_all.drop(['timestamp'], axis=1, inplace=True)
# Deal with categorical values

df_numeric = df_all.select_dtypes(exclude=['object'])

df_obj = df_all.select_dtypes(include=['object']).copy()



for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]



df_values = pd.concat([df_numeric, df_obj], axis=1)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2, interaction_only=True)

interactions = poly.fit_transform(df_values[['full_sq','life_sq','floor', 'build_year', 'max_floor']].fillna(0))

df_interactions = pd.DataFrame(data=interactions, columns=['poly' + str(i) for i in range(interactions.shape[1])])

df_values = pd.concat([df_values, df_interactions], axis=1)

df_values.head()
# Convert to numpy values

X_all = df_values.values

print(X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]



df_columns = df_values.columns
print(df_columns.values)
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)

dtest = xgb.DMatrix(X_test, feature_names=df_columns)
# Uncomment to tune XGB `num_boost_rounds`



#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

#    verbose_eval=True, show_stdv=False)

#cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()

#num_boost_rounds = len(cv_result)



num_boost_round = 409
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)
fig, ax = plt.subplots(1, 1, figsize=(8, 16))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
y_pred = model.predict(dtest)



df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



df_sub.to_csv('sub.csv', index=False)