# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/train.csv')

data.shape
#data[data['target']==1].head()

data['target'].value_counts()/data.shape[0]*100
from wordcloud import WordCloud

import matplotlib.pyplot as plt

insincere_rows=data[data['target']==1]

wc=WordCloud(background_color='white').generate(' '.join(insincere_rows['question_text']))

plt.imshow(wc)
from sklearn.model_selection import train_test_split

train,validate=train_test_split(data,test_size=0.3,random_state=1)

train.shape,validate.shape

train.head()
import nltk

def clean_sentence(doc,stopwords,stemmer):

    words=doc.split(' ')

    words_clean=[stemmer.stem(word) for word in words if word not in stopwords]

    return ' '.join(words_clean)

def clean_documents(docs_raw):

    stopwords=nltk.corpus.stopwords.words('english')

    stemmer=nltk.stem.PorterStemmer()

    docs=docs_raw.str.lower().str.replace('[^a-z ]','' )

    docs_clean=docs.apply(lambda doc:clean_sentence(doc,stopwords,stemmer))

    return docs_clean

train_docs_clean=clean_documents(train['question_text'])

train_docs_clean.head()
from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(min_df=10).fit(train_docs_clean)

dtm=vectorizer.transform(train_docs_clean)
dtm
from sklearn.tree import DecisionTreeClassifier

model_df=DecisionTreeClassifier(max_depth=10).fit(dtm,train['target'])
validate_docs_clean=clean_documents(validate['question_text'])

dtm_validate=vectorizer.transform(validate_docs_clean)

dtm_validate
validate_pred=model_df.predict(dtm_validate)

from sklearn.metrics import f1_score

f1_score(validate['target'],validate_pred)
from sklearn.naive_bayes import MultinomialNB

model_nb=MultinomialNB().fit(dtm,train['target'])

validate_pred=model_nb.predict(dtm_validate)

f1_score(validate['target'],validate_pred)
test=pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')
test.head()
docs_clean=clean_documents(test['question_text'])

dtm_test=vectorizer.transform(docs_clean)

dtm_test
test_pred=model_nb.predict(dtm_test)
sample_submission=pd.read_csv("/kaggle/input/quora-insincere-questions-classification/sample_submission.csv")

sample_submission
submission=pd.DataFrame({'qid':test['qid'],

                        'prediction':test_pred})

submission[['qid','prediction']].to_csv('submission.csv',index=False)
