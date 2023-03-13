# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



def warn(*args, **kwargs): pass

import warnings

warnings.warn = warn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.
def encode(train,test):

    le = LabelEncoder().fit(train.species) # fit label encoder

    labels = le.transform(train.species)   # Transform labels to normalized encoding

    classes = list(le.classes_) #save column names for submission

    test_ids = test.id  # save test ids for submission

    

    train = train.drop(['species', 'id'], axis=1) #해당 컬럼들을 data에서 제거.

    test = test.drop(['id'], axis=1)

    

    return train,labels,test,test_ids,classes



train, labels, test, test_ids, classes = encode(train, test)

train.head(1)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=23)



for train_index, test_index in sss.split(train, labels):

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis







log_cols = ["Classifier", "Accuracy","Log Loss"]

log = pd.DataFrame(columns = log_cols)



clf = KNeighborsClassifier(3)

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

print(train_predictions)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

print(train_predictions[:6])

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = SVC(kernel='rbf', C=0.025, probability=True)



clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = NuSVC(probability=True)

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = RandomForestClassifier()

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = AdaBoostClassifier()

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = GradientBoostingClassifier()

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = GaussianNB()

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = LinearDiscriminantAnalysis()

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
clf = QuadraticDiscriminantAnalysis()

clf.fit(X_train, y_train)

name = clf.__class__.__name__



print(name)

train_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, train_predictions)

print("Accuracy: {:.4%}".format(acc))



train_predictions = clf.predict_proba(X_test)

ll = log_loss(y_test, train_predictions) #로그손실의 평균 계산

print("Log Loss:{}".format(ll))
classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()