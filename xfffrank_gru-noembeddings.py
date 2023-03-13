import os

import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

import math

from sklearn.model_selection import train_test_split

from sklearn import metrics



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, CuDNNLSTM

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Train shape : ",train_df.shape)

print("Test shape : ",test_df.shape)
## split to train and val

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)



## some config values 

embed_size = 300 # how big is each word vector

max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 100 # max number of words in a question to use



## fill up the missing values

train_X = train_df["question_text"].fillna("_na_").values

val_X = val_df["question_text"].fillna("_na_").values

test_X = test_df["question_text"].fillna("_na_").values



## Tokenize the sentences

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(train_X))

train_X = tokenizer.texts_to_sequences(train_X)

val_X = tokenizer.texts_to_sequences(val_X)

test_X = tokenizer.texts_to_sequences(test_X)



## Pad the sentences 

train_X = pad_sequences(train_X, maxlen=maxlen)

val_X = pad_sequences(val_X, maxlen=maxlen)

test_X = pad_sequences(test_X, maxlen=maxlen)



## Get the target values

train_y = train_df['target'].values

val_y = val_df['target'].values
print(len(train_df))

print(len(val_df))
inp = Input(shape=(maxlen,))

x = Embedding(max_features, embed_size)(inp)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)

x = Dense(16, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print(model.summary())
## Train the model 

model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y))
pred_noemb_val_y = model.predict([val_X], batch_size=1024, verbose=1)

best_score = 0

best_threshold = None

for thresh in np.arange(0.1, 0.501, 0.01):

    thresh = np.round(thresh, 2)

    score = metrics.f1_score(val_y, (pred_noemb_val_y>thresh))

    print("F1 score at threshold {0} is {1}".format(thresh, score))

    if score > best_score:

        best_score = score

        best_threshold = thresh

    

print('Best score: {0}, best threshold: {1}'.format(best_score, best_threshold))
pred_noemb_test_y = model.predict([test_X], batch_size=1024, verbose=1)
pred_test_y = (pred_noemb_test_y > best_threshold).astype(int)

out_df = pd.DataFrame({"qid":test_df["qid"].values})

out_df['prediction'] = pred_test_y

out_df.to_csv("submission.csv", index=False)