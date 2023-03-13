import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from tqdm import tqdm_notebook

import warnings

import multiprocessing

from scipy.optimize import minimize  

warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
oof = np.zeros(len(train))

preds = np.zeros(len(test))



for i in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = QuadraticDiscriminantAnalysis(0.5)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')
test['target'] = preds

test.loc[test['target'] > 0.99, 'target'] = 1

test.loc[test['target'] < 0.01, 'target'] = 0
usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

new_train = pd.concat([train, usefull_test]).reset_index(drop=True)

new_train.loc[oof > 0.99, 'target'] = 1

new_train.loc[oof < 0.01, 'target'] = 0
oof2 = np.zeros(len(train))

preds2 = np.zeros(len(test))

for i in tqdm_notebook(range(512)):



    train2 = new_train[new_train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train[train['wheezy-copper-turtle-magic']==i].index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=2).fit_transform(data[cols]))

    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=11, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):

        oof_test_index = [t for t in test_index if t < len(idx1)]

        

        clf = QuadraticDiscriminantAnalysis(0.5)

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        if len(oof_test_index) > 0:

            oof2[idx1[oof_test_index]] = clf.predict_proba(train3[oof_test_index,:])[:,1]

        preds2[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        

auc = roc_auc_score(train['target'], oof2)

print(f'AUC: {auc:.5}')
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds2

sub.to_csv('submission.csv',index=False)