# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import gensim

print(os.listdir("../input/embeddings/GoogleNews-vectors-negative300/"))



# Any results you write to the current directory are saved as output.
url = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"

embeddings = gensim.models.KeyedVectors.load_word2vec_format(url,binary =True)
url = 'https://raw.githubusercontent.com/vivektop/Unstructrured-Data-Anaysis/master/yelp_labelled.csv'

yelp = pd.read_csv(url,sep='\t',header=None)

yelp = yelp.rename(columns={0:'Review', 1: 'Sentiment'})

yelp.shape
import nltk

docs_vectors = pd.DataFrame() # Final DF

stopwords = nltk.corpus.stopwords.words('english')

for doc in yelp['Review'].str.lower().str.replace('[^a-z ]',''): # For every document

    temp = pd.DataFrame() # temp df for every word

    for word in doc.split(' '):

        if word not in stopwords: #one word at a time

            try:

                word_vec = embeddings[word] #If available in embeddings append it to temp

                temp = temp.append(pd.Series(word_vec),ignore_index = True) # convert to seriesto append easier

            except: # if not then pass

                pass

    doc_vector = temp.mean() # Find the column sum 

    docs_vectors = docs_vectors.append(doc_vector,ignore_index = True)
docs_vectors['sentiment'] = yelp['Sentiment']
from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier

train_x,test_x,train_y,test_y = train_test_split(docs_vectors.drop('sentiment',axis = 1),

                                                 docs_vectors['sentiment'],

                                                 test_size = 0.2,

                                                random_state = 100)

train_x.shape,test_x.shape,train_y.shape,test_y.shape
model = AdaBoostClassifier(n_estimators=800,random_state=1)

model.fit(train_x,train_y)

test_pred = model.predict(test_x)

from sklearn.metrics import accuracy_score

accuracy_score(test_y,test_pred)