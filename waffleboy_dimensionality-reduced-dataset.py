# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

alg = xgb.XGBClassifier(learning_rate =0.02,n_estimators=700,max_depth=6,min_child_weight=1,
                         gamma=0,subsample=0.9,colsample_bytree=0.85,objective= 'binary:logistic',nthread=7,
                         scale_pos_weight=1,seed=27)
target='TARGET'
IDcol='ID'
                         
predictors= [x for x in train.columns if x not in [target, IDcol]]
cv_folds=5
early_stopping_rounds=25

#get parameters
xgb_param = alg.get_xgb_params()
#make dmatrix
xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
#do cross validation 
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds, metrics=['auc'],
     early_stopping_rounds=early_stopping_rounds)
alg.set_params(n_estimators=cvresult.shape[0]) #set n_estimators to best round

#Fit the algorithm on the data
alg.fit(train[predictors], train['TARGET'],eval_metric='auc')

#Predict training set:
train_predictions = alg.predict(train[predictors])
train_predprob = alg.predict_proba(train[predictors])[:,1]

#Print model report:
print ("\nModel Report")
print ("Accuracy : %.4g" % metrics.accuracy_score(train['TARGET'].values, train_predictions))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(train['TARGET'], train_predprob))

feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#feat_imp.plot(kind='bar', title='Feature Importances')  ##If needed
#plt.ylabel('Feature Importance Score') 

#get 64 most important features
feat_imp=list(feat_imp.index[:64])
feat_imp.append('TARGET')

newTrain = train[feat_imp] #done!
pd.to_csv(newTrain,'Satander64Features.csv')

#test your new dataset
def testFeatureReduction(train,newtrain):
    if sum(newTrain['TARGET']) == sum(train['TARGET']):
        print('Number of ones in target column preserved')
    else:
        print('Some information lost - not all ones are present')
    print('Columns reduced from '+str(len(train.columns)) +' to ' +str(len(newtrain.columns)))

testFeatureReduction(train,newTrain)
