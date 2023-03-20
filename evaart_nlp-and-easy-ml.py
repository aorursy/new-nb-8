#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn
import xgboost 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




df_train = pd.read_csv(os.path.join('../input/train', 'train.csv'))
df_test = pd.read_csv(os.path.join('../input/test', 'test.csv'))
df_breed_labels = pd.read_csv(os.path.join('../input', 'breed_labels.csv'))
df_color_labels = pd.read_csv(os.path.join('../input', 'color_labels.csv'))




df_train.shape, df_test.shape




df_breed_labels.head()




df_train.isnull().sum()




df_train.head()




adoption_0 = (df_train['AdoptionSpeed'] == 0).sum()
adoption_1 = (df_train['AdoptionSpeed'] == 1).sum()
adoption_2 = (df_train['AdoptionSpeed'] == 2).sum()
adoption_3 = (df_train['AdoptionSpeed'] == 3).sum()
adoption_4 = (df_train['AdoptionSpeed'] == 4).sum()
adoption_0, adoption_1, adoption_2, adoption_3, adoption_4




df_nlp = df_train[['Description', 'AdoptionSpeed']]
df_nlp.shape




df_nlp.head()




index_delete = np.array(df_nlp[df_nlp['Description'].isnull() == True].index)
index_delete




df_nlp = df_nlp.drop(index_delete)
df_nlp.shape




import random
index_adoption_0 = np.array(df_nlp[df_nlp['AdoptionSpeed'] == 0].index)
index_adoption_1 = np.array(df_nlp[df_nlp['AdoptionSpeed'] == 1].index)
index_adoption_2= np.array(df_nlp[df_nlp['AdoptionSpeed'] == 2].index)
index_adoption_3 = np.array(df_nlp[df_nlp['AdoptionSpeed'] == 3].index)
index_adoption_4 = np.array(df_nlp[df_nlp['AdoptionSpeed'] == 4].index)
index_adoption_1_reduc = [index_adoption_1[i] for i in range(int(len(index_adoption_0)))]
index_adoption_2_reduc = [index_adoption_2[i] for i in range(int(len(index_adoption_0)))]
index_adoption_3_reduc = [index_adoption_3[i] for i in range(int(len(index_adoption_0)))]
index_adoption_4_reduc = [index_adoption_4[i] for i in range(int(len(index_adoption_0)))]




X = pd.concat([df_nlp['Description'].reindex(index_adoption_0), df_nlp['Description'].reindex(index_adoption_1_reduc),
               df_nlp['Description'].reindex(index_adoption_2_reduc), df_nlp['Description'].reindex(index_adoption_3_reduc),
               df_nlp['Description'].reindex(index_adoption_4_reduc)])

y = pd.concat([df_nlp['AdoptionSpeed'].reindex(index_adoption_0), df_nlp['AdoptionSpeed'].reindex(index_adoption_1_reduc),
               df_nlp['AdoptionSpeed'].reindex(index_adoption_2_reduc), df_nlp['AdoptionSpeed'].reindex(index_adoption_3_reduc),
               df_nlp['AdoptionSpeed'].reindex(index_adoption_4_reduc)])
X.shape, y.shape




from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape




import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer




def tokenize(data):
    tokenized_docs = [word_tokenize(doc) for doc in data]
    alpha_tokens = [[t.lower() for t in doc if t.isalpha() == True] for doc in tokenized_docs]
    lemmatizer = WordNetLemmatizer()
    lem_tokens = [[lemmatizer.lemmatize(alpha) for alpha in doc] for doc in alpha_tokens]
    X_stem_as_string = [" ".join(x_t) for x_t in lem_tokens]
    return X_stem_as_string




X_train_tk = tokenize(X_train)
X_valid_tk = tokenize(X_valid)




from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline




vct = CountVectorizer(stop_words='english', lowercase=False)
svd = TruncatedSVD(n_components=200, random_state=42)
tfvec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=False)

preprocessing_pipe = Pipeline([
    ('vectorizer', tfvec),
    ('svd', svd)   
])




lsa_train = preprocessing_pipe.fit_transform(X_train_tk)
lsa_train.shape




sns.scatterplot(lsa_train[:, 0], lsa_train[:, 1], hue=y_train);




components = pd.DataFrame(data=svd.components_, columns=preprocessing_pipe.named_steps['vectorizer'].get_feature_names())




fig, axes = plt.subplots(10, 2, figsize=(18, 30))
for i, ax in enumerate(axes.flat):
    components.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB




rf = RandomForestClassifier()
mb = MultinomialNB()
pipe = Pipeline([
    ('vectorizer', tfvec),
    ('rf', mb)
])




pipe.fit(X_train_tk, y_train)
y_pred = pipe.predict(X_valid_tk)




print(classification_report(y_valid, y_pred))




df_train = df_train.drop(['Name', 'Description', 'RescuerID', 'PetID'], axis=1)
df_test = df_test.drop(['Name', 'Description'], axis=1)
X_test = df_test.drop(['RescuerID', 'PetID'], axis=1)




cor_mat = df_train[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig = plt.gcf()
fig.set_size_inches(80,15)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True);




var = 'Type'
data = pd.concat([df_train['AdoptionSpeed'], df_train[var]], axis=1)
plt.xlabel('Type')
plt.ylabel('AdoptionSpeed')
sns.boxplot(x=var, y="AdoptionSpeed", data=data);




X = df_train.drop('AdoptionSpeed', axis=1)
y = df_train['AdoptionSpeed']




from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split




## Data spliting 
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, shuffle=y, random_state=42)
(X_train.shape, y_train.shape), (X_valid.shape, y_valid.shape)




## Data well-balanced
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)
X_valid_sm, y_valid_sm = smote.fit_sample(X_valid, y_valid)




from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score




rf = RandomForestClassifier(max_depth=6, n_estimators=24, max_features=19, random_state=1)
rf.fit(X_train_sm, y_train_sm)
y_predic = rf.predict(X_train_sm)
acc_score = accuracy_score(y_train_sm, y_predic)
acc_score_valid = accuracy_score(y_valid_sm, rf.predict(X_valid_sm))
acc_score, acc_score_valid




xgb = XGBClassifier(max_depth=3, n_estimators=200, learning_rate=0.19, random_state=42)
xgb.fit(X_train_sm, y_train_sm)
y_predic = xgb.predict(X_train_sm)
acc_score = accuracy_score(y_train_sm, y_predic)
acc_score_valid = accuracy_score(y_valid_sm, xgb.predict(X_valid_sm))
acc_score, acc_score_valid




y_pred_true = xgb.predict(X_test.as_matrix())




model_submission  = pd.DataFrame(y_pred_true).apply(np.round)
submission = pd.DataFrame(data={"PetID" : df_test["PetID"], 
                                   "AdoptionSpeed" : model_submission[0]})
submission.AdoptionSpeed = submission.AdoptionSpeed.astype(int)
submission.to_csv("submission.csv", index=False)
















