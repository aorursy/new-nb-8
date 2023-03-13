# 

#   First part based on Reynaldo's script with a linear transformation of y_train

#   that happens to fit the public test data well

#   and may also fit the private test data well

#   but almost certainly won't generalize to later data

#   Second contribution  is based on Bruno do Amaral's very early entry but

#   with an outlier that Andy Harless deleted early in the competition

#   Third contribution  is based on a legitimate data cleaning,

#   probably by gunja agarwal (or actually by Jason Benner, it seems,

#   but there's also a small transformation applied ot the predictions,

#   so also probably not generalizable),

#   This combo being made by Andy Harless on June 4



#   Final contribution  is a Сatboost - made by Valeriy Babushkin. It makes solution more stable and improve a private score 

#   Geom average Private score  is  0.31508   Public is 0.31201   that is 73-rd place

#   Simle avrage Private score  is  0.31495  Public is 0.31193  that is 67-th place

#   One can tune a weights of 2 submissions and achieve a better final score





#   What is even more important is that by mixing with Catboost  we  receive a more robust solution

#   The gap between Public and Private is much smaller in case of Catboost - I personally shifted just 2 places up 

#   and finished 53-rd



###################################### ATTENTION! #################################################

#You have to change num_boost_rounds (in 3 places) and iterations (1 place) according to comments  

###################################### ATTENTION! #################################################





#  Data can be Accuired here https://www.kaggle.com/c/sberbank-russian-housing-market/dataB 
from catboost import  CatBoostRegressor



import numpy as np

import pandas as pd



from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime


res_ven = []

for ixi in [1]:

    #load files

    train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

    test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

    macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])

    id_test = test.id





    #clean data

    bad_index = train[train.life_sq > train.full_sq].index

    train.loc[bad_index, "life_sq"] = np.NaN

    equal_index = [601,1896,2791]

    test.loc[equal_index, "life_sq"] = test.loc[equal_index, "full_sq"]

    bad_index = test[test.life_sq > test.full_sq].index

    test.loc[bad_index, "life_sq"] = np.NaN

    bad_index = train[train.life_sq < 5].index

    train.loc[bad_index, "life_sq"] = np.NaN

    bad_index = test[test.life_sq < 5].index

    test.loc[bad_index, "life_sq"] = np.NaN

    bad_index = train[train.full_sq < 5].index

    train.loc[bad_index, "full_sq"] = np.NaN

    bad_index = test[test.full_sq < 5].index

    test.loc[bad_index, "full_sq"] = np.NaN

    kitch_is_build_year = [13117]

    train.loc[kitch_is_build_year, "build_year"] = train.loc[kitch_is_build_year, "kitch_sq"]

    bad_index = train[train.kitch_sq >= train.life_sq].index

    train.loc[bad_index, "kitch_sq"] = np.NaN

    bad_index = test[test.kitch_sq >= test.life_sq].index

    test.loc[bad_index, "kitch_sq"] = np.NaN

    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index

    train.loc[bad_index, "kitch_sq"] = np.NaN

    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index

    test.loc[bad_index, "kitch_sq"] = np.NaN

    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index

    train.loc[bad_index, "full_sq"] = np.NaN

    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index

    test.loc[bad_index, "full_sq"] = np.NaN

    bad_index = train[train.life_sq > 300].index

    train.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN

    bad_index = test[test.life_sq > 200].index

    test.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN

    train.product_type.value_counts(normalize= True)

    test.product_type.value_counts(normalize= True)

    bad_index = train[train.build_year < 1500].index

    train.loc[bad_index, "build_year"] = np.NaN

    bad_index = test[test.build_year < 1500].index

    test.loc[bad_index, "build_year"] = np.NaN

    bad_index = train[train.num_room == 0].index

    train.loc[bad_index, "num_room"] = np.NaN

    bad_index = test[test.num_room == 0].index

    test.loc[bad_index, "num_room"] = np.NaN

    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]

    train.loc[bad_index, "num_room"] = np.NaN

    bad_index = [3174, 7313]

    test.loc[bad_index, "num_room"] = np.NaN

    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index

    train.loc[bad_index, ["max_floor", "floor"]] = np.NaN

    bad_index = train[train.floor == 0].index

    train.loc[bad_index, "floor"] = np.NaN

    bad_index = train[train.max_floor == 0].index

    train.loc[bad_index, "max_floor"] = np.NaN

    bad_index = test[test.max_floor == 0].index

    test.loc[bad_index, "max_floor"] = np.NaN

    bad_index = train[train.floor > train.max_floor].index

    train.loc[bad_index, "max_floor"] = np.NaN

    bad_index = test[test.floor > test.max_floor].index

    test.loc[bad_index, "max_floor"] = np.NaN

    train.floor.describe(percentiles= [0.9999])

    bad_index = [23584]

    train.loc[bad_index, "floor"] = np.NaN

    train.material.value_counts()

    test.material.value_counts()

    train.state.value_counts()

    bad_index = train[train.state == 33].index

    train.loc[bad_index, "state"] = np.NaN

    test.state.value_counts()



    # brings error down a lot by removing extreme price per sqm

    train.loc[train.full_sq == 0, 'full_sq'] = 50

    train = train[train.price_doc/train.full_sq <= 600000]

    train = train[train.price_doc/train.full_sq >= 10000]



    # Add month-year

    month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)

    month_year_cnt_map = month_year.value_counts().to_dict()

    train['month_year_cnt'] = month_year.map(month_year_cnt_map)



    month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)

    month_year_cnt_map = month_year.value_counts().to_dict()

    test['month_year_cnt'] = month_year.map(month_year_cnt_map)



    # Add week-year count

    week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)

    week_year_cnt_map = week_year.value_counts().to_dict()

    train['week_year_cnt'] = week_year.map(week_year_cnt_map)



    week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)

    week_year_cnt_map = week_year.value_counts().to_dict()

    test['week_year_cnt'] = week_year.map(week_year_cnt_map)



    # Add month and day-of-week

    train['month'] = train.timestamp.dt.month

    train['dow'] = train.timestamp.dt.dayofweek



    test['month'] = test.timestamp.dt.month

    test['dow'] = test.timestamp.dt.dayofweek



    # Other feature engineering

    train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)

    train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)



    test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)

    test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)



    train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)

    test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)



    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)

    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)



    rate_2016_q2 = 1

    rate_2016_q1 = rate_2016_q2 / .99903

    rate_2015_q4 = rate_2016_q1 / .9831

    rate_2015_q3 = rate_2015_q4 / .9834

    rate_2015_q2 = rate_2015_q3 / .9815

    rate_2015_q1 = rate_2015_q2 / .9932

    rate_2014_q4 = rate_2015_q1 / 1.0112

    rate_2014_q3 = rate_2014_q4 / 1.0169

    rate_2014_q2 = rate_2014_q3 / 1.0086

    rate_2014_q1 = rate_2014_q2 / 1.0126

    rate_2013_q4 = rate_2014_q1 / 0.9902

    rate_2013_q3 = rate_2013_q4 / 1.0041

    rate_2013_q2 = rate_2013_q3 / 1.0044

    rate_2013_q1 = rate_2013_q2 / 1.0104

    rate_2012_q4 = rate_2013_q1 / 0.9832

    rate_2012_q3 = rate_2012_q4 / 1.0277

    rate_2012_q2 = rate_2012_q3 / 1.0279

    rate_2012_q1 = rate_2012_q2 / 1.0279

    rate_2011_q4 = rate_2012_q1 / 1.076

    rate_2011_q3 = rate_2011_q4 / 1.0236

    rate_2011_q2 = rate_2011_q3 / 1

    rate_2011_q1 = rate_2011_q2 / 1.011



    # test data

    test['average_q_price'] = 1



    test_2016_q2_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month <= 7].index

    test.loc[test_2016_q2_index, 'average_q_price'] = rate_2016_q2

    # test.loc[test_2016_q2_index, 'year_q'] = '2016_q2'



    test_2016_q1_index = test.loc[test['timestamp'].dt.year == 2016].loc[test['timestamp'].dt.month >= 1].loc[test['timestamp'].dt.month < 4].index

    test.loc[test_2016_q1_index, 'average_q_price'] = rate_2016_q1

    # test.loc[test_2016_q2_index, 'year_q'] = '2016_q1'



    test_2015_q4_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 10].loc[test['timestamp'].dt.month < 12].index

    test.loc[test_2015_q4_index, 'average_q_price'] = rate_2015_q4

    # test.loc[test_2015_q4_index, 'year_q'] = '2015_q4'



    test_2015_q3_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 7].loc[test['timestamp'].dt.month < 10].index

    test.loc[test_2015_q3_index, 'average_q_price'] = rate_2015_q3

    # test.loc[test_2015_q3_index, 'year_q'] = '2015_q3'



    # test_2015_q2_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index

    # test.loc[test_2015_q2_index, 'average_q_price'] = rate_2015_q2



    # test_2015_q1_index = test.loc[test['timestamp'].dt.year == 2015].loc[test['timestamp'].dt.month >= 4].loc[test['timestamp'].dt.month < 7].index

    # test.loc[test_2015_q1_index, 'average_q_price'] = rate_2015_q1





    # train 2015

    train['average_q_price'] = 1



    train_2015_q4_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

    # train.loc[train_2015_q4_index, 'price_doc'] = train.loc[train_2015_q4_index, 'price_doc'] * rate_2015_q4

    train.loc[train_2015_q4_index, 'average_q_price'] = rate_2015_q4



    train_2015_q3_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

    #train.loc[train_2015_q3_index, 'price_doc'] = train.loc[train_2015_q3_index, 'price_doc'] * rate_2015_q3

    train.loc[train_2015_q3_index, 'average_q_price'] = rate_2015_q3



    train_2015_q2_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

    #train.loc[train_2015_q2_index, 'price_doc'] = train.loc[train_2015_q2_index, 'price_doc'] * rate_2015_q2

    train.loc[train_2015_q2_index, 'average_q_price'] = rate_2015_q2



    train_2015_q1_index = train.loc[train['timestamp'].dt.year == 2015].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

    #train.loc[train_2015_q1_index, 'price_doc'] = train.loc[train_2015_q1_index, 'price_doc'] * rate_2015_q1

    train.loc[train_2015_q1_index, 'average_q_price'] = rate_2015_q1





    # train 2014

    train_2014_q4_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

    #train.loc[train_2014_q4_index, 'price_doc'] = train.loc[train_2014_q4_index, 'price_doc'] * rate_2014_q4

    train.loc[train_2014_q4_index, 'average_q_price'] = rate_2014_q4



    train_2014_q3_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

    #train.loc[train_2014_q3_index, 'price_doc'] = train.loc[train_2014_q3_index, 'price_doc'] * rate_2014_q3

    train.loc[train_2014_q3_index, 'average_q_price'] = rate_2014_q3



    train_2014_q2_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

    #train.loc[train_2014_q2_index, 'price_doc'] = train.loc[train_2014_q2_index, 'price_doc'] * rate_2014_q2

    train.loc[train_2014_q2_index, 'average_q_price'] = rate_2014_q2



    train_2014_q1_index = train.loc[train['timestamp'].dt.year == 2014].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

    #train.loc[train_2014_q1_index, 'price_doc'] = train.loc[train_2014_q1_index, 'price_doc'] * rate_2014_q1

    train.loc[train_2014_q1_index, 'average_q_price'] = rate_2014_q1





    # train 2013

    train_2013_q4_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

    # train.loc[train_2013_q4_index, 'price_doc'] = train.loc[train_2013_q4_index, 'price_doc'] * rate_2013_q4

    train.loc[train_2013_q4_index, 'average_q_price'] = rate_2013_q4



    train_2013_q3_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

    # train.loc[train_2013_q3_index, 'price_doc'] = train.loc[train_2013_q3_index, 'price_doc'] * rate_2013_q3

    train.loc[train_2013_q3_index, 'average_q_price'] = rate_2013_q3



    train_2013_q2_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

    # train.loc[train_2013_q2_index, 'price_doc'] = train.loc[train_2013_q2_index, 'price_doc'] * rate_2013_q2

    train.loc[train_2013_q2_index, 'average_q_price'] = rate_2013_q2



    train_2013_q1_index = train.loc[train['timestamp'].dt.year == 2013].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

    # train.loc[train_2013_q1_index, 'price_doc'] = train.loc[train_2013_q1_index, 'price_doc'] * rate_2013_q1

    train.loc[train_2013_q1_index, 'average_q_price'] = rate_2013_q1





    # train 2012

    train_2012_q4_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

    # train.loc[train_2012_q4_index, 'price_doc'] = train.loc[train_2012_q4_index, 'price_doc'] * rate_2012_q4

    train.loc[train_2012_q4_index, 'average_q_price'] = rate_2012_q4



    train_2012_q3_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

    # train.loc[train_2012_q3_index, 'price_doc'] = train.loc[train_2012_q3_index, 'price_doc'] * rate_2012_q3

    train.loc[train_2012_q3_index, 'average_q_price'] = rate_2012_q3



    train_2012_q2_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

    # train.loc[train_2012_q2_index, 'price_doc'] = train.loc[train_2012_q2_index, 'price_doc'] * rate_2012_q2

    train.loc[train_2012_q2_index, 'average_q_price'] = rate_2012_q2



    train_2012_q1_index = train.loc[train['timestamp'].dt.year == 2012].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

    # train.loc[train_2012_q1_index, 'price_doc'] = train.loc[train_2012_q1_index, 'price_doc'] * rate_2012_q1

    train.loc[train_2012_q1_index, 'average_q_price'] = rate_2012_q1





    # train 2011

    train_2011_q4_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 10].loc[train['timestamp'].dt.month <= 12].index

    # train.loc[train_2011_q4_index, 'price_doc'] = train.loc[train_2011_q4_index, 'price_doc'] * rate_2011_q4

    train.loc[train_2011_q4_index, 'average_q_price'] = rate_2011_q4



    train_2011_q3_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 7].loc[train['timestamp'].dt.month < 10].index

    # train.loc[train_2011_q3_index, 'price_doc'] = train.loc[train_2011_q3_index, 'price_doc'] * rate_2011_q3

    train.loc[train_2011_q3_index, 'average_q_price'] = rate_2011_q3



    train_2011_q2_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 4].loc[train['timestamp'].dt.month < 7].index

    # train.loc[train_2011_q2_index, 'price_doc'] = train.loc[train_2011_q2_index, 'price_doc'] * rate_2011_q2

    train.loc[train_2011_q2_index, 'average_q_price'] = rate_2011_q2



    train_2011_q1_index = train.loc[train['timestamp'].dt.year == 2011].loc[train['timestamp'].dt.month >= 1].loc[train['timestamp'].dt.month < 4].index

    # train.loc[train_2011_q1_index, 'price_doc'] = train.loc[train_2011_q1_index, 'price_doc'] * rate_2011_q1

    train.loc[train_2011_q1_index, 'average_q_price'] = rate_2011_q1



    train['price_doc'] = train['price_doc'] * train['average_q_price']

    # train.drop('average_q_price', axis=1, inplace=True)



    print('price changed done')



    y_train = train["price_doc"]

    # x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

    # x_test = test.drop(["id", "timestamp"], axis=1)



    x_train = train.drop(["id", "timestamp", "price_doc", "average_q_price"], axis=1)

    x_test = test.drop(["id", "timestamp", "average_q_price"], axis=1)



    num_train = len(x_train)

    x_all = pd.concat([x_train, x_test])



    for c in x_all.columns:

        if x_all[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(x_all[c].values))

            x_all[c] = lbl.transform(list(x_all[c].values))

            #x_train.drop(c,axis=1,inplace=True)



    x_train = x_all[:num_train]

    x_test = x_all[num_train:]





    xgb_params = {

        'eta': 0.05,

        'max_depth': 6,

        'subsample': 0.6,

        'colsample_bytree': 1,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'silent': 1

    }



    dtrain = xgb.DMatrix(x_train, y_train)

    dtest = xgb.DMatrix(x_test)



    # cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    #     verbose_eval=20, show_stdv=False)

    #cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

    # cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)

    # print('best num_boost_rounds = ', len(cv_output))

    # num_boost_rounds = len(cv_output) 



    num_boost_rounds = 1 #change to 422

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)



    #fig, ax = plt.subplots(1, 1, figsize=(8, 13))

    #xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)



    y_predict = model.predict(dtest)

    # y_predict = np.round(y_predict)

    gunja_output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

    # gunja_output['price_doc'] = gunja_output['price_doc'] * gunja_output['average_q_price']

    # gunja_output.drop('average_q_price', axis=1, inplace=True)

    # gunja_output.head()



    train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

    test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

    id_test = test.id



    mult = .969



    y_train = train["price_doc"] * mult + 10

    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

    x_test = test.drop(["id", "timestamp"], axis=1)



    for c in x_train.columns:

        if x_train[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(x_train[c].values))

            x_train[c] = lbl.transform(list(x_train[c].values))



    for c in x_test.columns:

        if x_test[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(x_test[c].values))

            x_test[c] = lbl.transform(list(x_test[c].values))



    xgb_params = {

        'eta': 0.05,

        'max_depth': 5,

        'subsample': 0.7,

        'colsample_bytree': 0.7,

        'objective': 'reg:linear',

        'eval_metric': 'rmse',

        'silent': 1

    }



    dtrain = xgb.DMatrix(x_train, y_train)

    dtest = xgb.DMatrix(x_test)



    # cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)

    # print('best num_boost_rounds = ', len(cv_output))

    # num_boost_rounds = len(cv_output) # 382



    num_boost_rounds = 1  # change to 385

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)



    y_predict = model.predict(dtest)

    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

    # output.drop('average_q_price', axis=1, inplace=True)

    # output.head()



    # Any results you write to the current directory are saved as output.

    df_train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])

    df_test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])

    df_macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])



    df_train.drop(df_train[df_train["life_sq"] > 7000].index, inplace=True)



    mult = 0.969

    y_train = df_train['price_doc'].values * mult + 10

    id_test = df_test['id']



    df_train.drop(['id', 'price_doc'], axis=1, inplace=True)

    df_test.drop(['id'], axis=1, inplace=True)



    num_train = len(df_train)

    df_all = pd.concat([df_train, df_test])

    # Next line just adds a lot of NA columns (becuase "join" only works on indexes)

    # but somewhow it seems to affect the result

    df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')

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



    train['building_name'] = pd.factorize(train.sub_area + train['metro_km_avto'].astype(str))[0]

    test['building_name'] = pd.factorize(test.sub_area + test['metro_km_avto'].astype(str))[0]



    def add_time_features(col):

       col_month_year = pd.Series(pd.factorize(train[col].astype(str) + month_year.astype(str))[0])

       train[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())



       col_week_year = pd.Series(pd.factorize(train[col].astype(str) + week_year.astype(str))[0])

       train[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())



    add_time_features('building_name')

    add_time_features('sub_area')



    def add_time_features(col):

       col_month_year = pd.Series(pd.factorize(test[col].astype(str) + month_year.astype(str))[0])

       test[col + '_month_year_cnt'] = col_month_year.map(col_month_year.value_counts())



       col_week_year = pd.Series(pd.factorize(test[col].astype(str) + week_year.astype(str))[0])

       test[col + '_week_year_cnt'] = col_week_year.map(col_week_year.value_counts())



    add_time_features('building_name')

    add_time_features('sub_area')





    # Remove timestamp column (may overfit the model in train)

    df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)





    factorize = lambda t: pd.factorize(t[1])[0]



    df_obj = df_all.select_dtypes(include=['object'])



    X_all = np.c_[

        df_all.select_dtypes(exclude=['object']).values,

        np.array(list(map(factorize, df_obj.iteritems()))).T

    ]

    print(X_all.shape)



    X_train = X_all[:num_train]

    X_test = X_all[num_train:]





    # Deal with categorical values

    df_numeric = df_all.select_dtypes(exclude=['object'])

    df_obj = df_all.select_dtypes(include=['object']).copy()



    for c in df_obj:

        df_obj[c] = pd.factorize(df_obj[c])[0]



    df_values = pd.concat([df_numeric, df_obj], axis=1)





    # Convert to numpy values

    X_all = df_values.values

    print(X_all.shape)



    X_train = X_all[:num_train]

    X_test = X_all[num_train:]



    df_columns = df_values.columns





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



    # cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20, verbose_eval=25, show_stdv=False)

    # print('best num_boost_rounds = ', len(cv_output))

    # num_boost_rounds = len(cv_output) #



    num_boost_rounds = 1  # change it to 420

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)



    y_pred = model.predict(dtest)



    df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})



    df_sub.head()

    first_result = output.merge(df_sub, on="id", suffixes=['_louis','_bruno'])

    first_result["price_doc"] = np.exp( .714*np.log(first_result.price_doc_louis) +

                                        .286*np.log(first_result.price_doc_bruno) )  # multiplies out to .5 & .2

    result = first_result.merge(gunja_output, on="id", suffixes=['_follow','_gunja'])



    result["price_doc"] = np.exp( .78*np.log(result.price_doc_follow) +

                                  .22*np.log(result.price_doc_gunja) )



    result.drop(["price_doc_louis","price_doc_bruno","price_doc_follow","price_doc_gunja"],axis=1,inplace=True)

    result.head()

    result.to_csv('submit_1.csv', index=False)

    res_ven.append(result)

ntrain = X_train.shape[0]

xtr_te = pd.concat((pd.DataFrame(X_train),pd.DataFrame(X_test)),axis =0)

xtr_te.fillna(-999999999,inplace=True) 

xtr_te = np.array(xtr_te)

X_train = xtr_te[:ntrain:,:]

X_test = xtr_te[ntrain:,:]
cat = [] # we would like to fiand a categorical data

cat_test = pd.DataFrame(xtr_te)

for i in range(cat_test.shape[1]):

    cat.append(((len(cat_test[i].unique()))))

print('Minimum number of distinct values in a column is', (min(cat)))

if min(cat)<100:

    print("Looks like there is  categorical features")

else:

    print("Looks like there is no categorical features")

cat = []

cat_test = pd.DataFrame(xtr_te)

for i in range(cat_test.shape[1]):

    cat.append(((len(cat_test[i].unique()),i)))

cat.sort()
number = [] # get exact position of categorical columns

for i in range(394):

    if cat[i][0]<xtr_te.shape[0]/100 and cat[i][0]>2:

        number.append(cat[i][1])

print (len(number), 'categorical columns')
categories = xtr_te[:,number].astype('str',copy = False) # there is a float values - we need to convert to str or int

xtr_te = pd.DataFrame(xtr_te)

xtr_te.drop(number,axis = 1,inplace=True)

xtr_te = np.concatenate((np.array(xtr_te),categories),axis = 1)

X_train = xtr_te[:ntrain:,:]

X_test = xtr_te[ntrain:,:]
reserv_X = X_train

reserv_Y = y_train

reserv_t = X_test

def rmsle(pred,real):

    return sum((np.log(pred+1) - np.log(real+1))**2)/len(pred)
from sklearn.cross_validation import KFold



folds = 10

cv_sum = 0

fpred = []

kf = KFold(reserv_X.shape[0], n_folds=folds)

for i, (train_index, test_index) in enumerate(kf):

    print('\n Fold %d\n' % (i + 1))

    X_train, X_val = reserv_X[train_index], reserv_X[test_index]

    y_train, y_val = reserv_Y[train_index], reserv_Y[test_index]

    

    evaluate = (X_val,y_val)

    res = []

    cd  = list(range(xtr_te.shape[1] -len(number),xtr_te.shape[1]))

    for d in [7]:

        for it in [1]: #change it to 6000

            for shr in [1]:

                train_data = X_train

                train_labels = y_train

                model = CatBoostRegressor(depth=d,iterations=it,random_seed=1342,use_best_model=True,thread_count=16

                                         )

                model.fit(X = train_data, y = train_labels, cat_features=cd,eval_set = evaluate,use_best_model=True)

                pred = model.predict(X_val)

                pred = np.array(pred)

                print (rmsle(pred,y_val),'depth',d,'iterations',model.get_tree_count())

    res.append((rmsle(pred,y_val),'depth',d,'iterations',model.get_tree_count()))

    cv_sum+=rmsle(pred,y_val)            

    y_pred = np.array(model.predict(X_test))

    

                

    fpred.append(y_pred)

score =  cv_sum/folds

print('\n Average RMSLE: %.6f' % score)



final = np.zeros_like(fpred[0])

for i in fpred:

    final+=i

final = final/folds
sub = pd.read_csv('../input/sample_submission.csv')

sub.price_doc = final

sub.to_csv('submit_2.csv',index=None)

x1 = result.copy()

x2 = sub.copy()

result = (x1.price_doc * x2.price_doc)**0.5

sub = pd.read_csv('../input/sample_submission.csv')

sub.price_doc = result

sub.to_csv('submit_final_geometrix_average.csv',index=None)

result = (x1.price_doc*0.5+ x2.price_doc*0.5)

sub = pd.read_csv('../input/sample_submission.csv')

sub.price_doc = result

sub.to_csv('submit_final_average.csv',index=None)