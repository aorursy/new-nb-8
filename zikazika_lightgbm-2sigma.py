import numpy as np
import pandas as pd
import lightgbm as lgb
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(marketdf, newsdf) = env.get_training_data()

print('Preparation:')
def prepare_data(marketdf, newsdf):
    #feature engineering
    marketdf['time'] = marketdf.time.dt.strftime("%Y%m%d").astype(int)
    marketdf['bartrend'] = marketdf['close'] / marketdf['open']
    marketdf['average'] = (marketdf['close'] + marketdf['open'])/2
    marketdf['pricevolume'] = marketdf['volume'] * marketdf['close']
    
    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)
    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
    newsdf['coverage'] = newsdf['sentimentWordCount'] / newsdf['wordCount']

    # filter post 2010 data, huge crisis is not representative
    marketdf = marketdf.loc[marketdf['time'] > 20100000]
    
    # columns that i tought are not necessary
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'assetCodes','urgency','wordCount','sentimentWordCount']
    newsdf.drop(droplist, axis=1, inplace=True)
    marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)
    
    # combine multiple news reports for same assets on same day
    newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    
    # join news reports to market data, note many assets will have many days without news data
    return pd.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'], copy=False) #, right_on=['time', 'assetCodes'])

cdf = prepare_data(marketdf, newsdf)    
del marketdf, newsdf  # save the precious memory


print('building training set...')
targetcols = ['returnsOpenNextMktres10']
traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe'] + targetcols]

dates = cdf['time'].unique()
train = range(len(dates))[:int(0.85*len(dates))]
val = range(len(dates))[int(0.85*len(dates)):]

# we be classifyin
cdf[targetcols[0]] = (cdf[targetcols[0]] > 0).astype(int)

# train data
Xt = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[train])].values
Yt = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[train])].values

# validation data
Xv = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[val])].values
Yv = cdf[targetcols].fillna(0).loc[cdf['time'].isin(dates[val])].values

print(Xt.shape, Xv.shape)




## LightGBM

import lightgbm as lgb

# money
params = {"objective" : 'binary',
          "metric" : "binary_logloss",
          "num_leaves" : 130,
          "max_depth": -1,
          "learning_rate" : 0.01,
          "bagging_fraction" : 0.75,  
          "feature_fraction" : 0.9, 
          "bagging_freq" : 2,        
          "bagging_seed" : 2018,
         
          "verbosity" : -1 }


lgtrain, lgval = lgb.Dataset(Xt, Yt[:,0]), lgb.Dataset(Xv, Yv[:,0])
lgbmodel = lgb.train(params, lgtrain, 2000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, verbose_eval=200)





print("generating predictions...")
preddays = env.get_prediction_days()
for marketdf, newsdf, predtemplatedf in preddays:
    cdf = prepare_data(marketdf, newsdf)
    Xp = cdf[traincols].fillna(0).values
    preds = lgbmodel.predict(Xp, num_iteration=lgbmodel.best_iteration) * 2 - 1
    predsdf = pd.DataFrame({'ast':cdf['assetCode'],'conf':preds})
    predtemplatedf['confidenceValue'][predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values
    env.predict(predtemplatedf)

env.write_submission_file()