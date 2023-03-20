#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # visualization tools


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




#Loading in dataset
data = pd.read_csv('../input/train.csv')

# Distribution of labels
data['target_encode'] = data['target'].replace([0,1],['sincere','insincere'])
target_plot = sns.countplot(x='target_encode', data=data)
target_plot
print('Number of sincere questions:',len(data[data['target_encode']=='sincere']))
print('Number of sincere questions:',len(data[data['target_encode']=='insincere']))




import string
from nltk.corpus import stopwords

#Tokenization function
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Vectorization & TF-IDF Transformation tools
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer




# Naives Bayes model
from sklearn.naive_bayes import MultinomialNB

# Create a data pipeline for training and testing our models
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])




from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data['question_text'], data['target'], test_size=0.2)

print("Number of Training Instances:",len(X_train))
print("Number of Testing Instances:",len(X_test))




training_data = pd.DataFrame({'questions_text':X_train,'target':Y_train})




# Downsampling the dataset so the ratio of sincere:insincere apprximmately 1:1
data_sincere = training_data[training_data['target']==0]
data_insincere = training_data[training_data['target']==1]
data_sincere_downsample = data_sincere.sample(n=98000)

data_downsample = pd.concat([data_insincere, data_sincere_downsample], axis=0)

print(data_downsample.target.value_counts())




# Train the model
pipeline.fit(data_downsample['questions_text'],data_downsample['target'])




# Predict on the testing set and evaluate the model
predictions = pipeline.predict(X_test)

# Classication report
from sklearn.metrics import classification_report
print(classification_report(predictions,Y_test))




# Read in test data and check the dataframes
test_data = pd.read_csv('../input/test.csv')

# Make a prediction and output to csv file
test_data_predictions = pipeline.predict(test_data['question_text'])

test_data['prediction'] = test_data_predictions
test_data.to_csv('submission.csv', columns=['qid','prediction'], index=False)

