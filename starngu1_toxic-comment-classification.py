# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv')

subm = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
train.head()
train['comment_text'][0]
train['comment_text'][2]
lens = train.comment_text.str.len()

lens.mean(), lens.std(), lens.max()
lens.hist()
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train['none'] = 1-train[label_cols].max(axis=1)

train.describe()
len(train),len(test)
COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)

test[COMMENT].fillna("unknown", inplace=True)
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

trn_term_doc = vec.fit_transform(train[COMMENT])

test_term_doc = vec.transform(test[COMMENT])
trn_term_doc, test_term_doc
def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc

test_x = test_term_doc
def get_mdl(y):

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    m = LogisticRegression(C=4, dual=True, max_iter=10000)

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r
preds = np.zeros((len(test), len(label_cols)))



for i, j in enumerate(label_cols):

    print('fit', j)

    m,r = get_mdl(train[j])

    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.to_csv('submission.csv', index=False)