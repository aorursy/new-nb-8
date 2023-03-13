# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
test = pd.read_csv('../input/test.csv')

test.head()
from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit
def encode(train,test):

    le = LabelEncoder().fit(train.species)

    labels = le.transform(train.species)

    classes = list(le.classes_)

    test_ids = test.id

    

    train = train.drop(['id','species'],axis = 1)

    test = test.drop(['id'],axis = 1)

    return train,labels,test,test_ids,classes



train,labels,test,test_ids,classes = encode(train,test)

train.head(1)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)



for train_index,test_index in sss:

    X_train,X_test = train.values[train_index],train.values[test_index]

    y_train,y_test = labels[train_index],labels[test_index]
X_train.shape
X_test.shape
0.2*990