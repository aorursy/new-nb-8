# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
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
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import ModelCheckpoint, History

## some config values 
EMBED_SIZE = 300 #how big is each word vector
MAX_FEATURES= 50000 # how many unique words to use (i.e num rows in embedding vector)
MAX_LEN = 100 # max number of words in a question to use

def load_preprocess_data(test_size=0.1,  max_features = MAX_FEATURES, maxlen = MAX_LEN ):
    
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
#     print("Train shape : ",train_df.shape)
#     print("Test shape : ",test_df.shape)

    ## split to train and val
    train_df_1, val_df = train_test_split(train_df, test_size=test_size, random_state=2018)
    
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
    
    return train_X , train_y, val_X, val_y, test_X, tokenizer.index_word
train_X , train_y, val_X, val_y, test_X, word_index = load_preprocess_data()
def build_model( max_features = MAX_FEATURES, maxlen = MAX_LEN, embed_size = EMBED_SIZE,):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
test_model = build_model()
# weight_path="{}_weights.best.hdf5".format('model')
# checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
#                              save_best_only=True, mode='max', save_weights_only = True)
# callbacks_list = [checkpoint]
def get_embedding_matrix(max_features = MAX_FEATURES):
    
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = get_embedding_matrix()
## Train the model 
history = History()
#history_log = [test_model.fit(train_X, train_y, batch_size=512, epochs=2, callbacks=callbacks_list, validation_data=(val_X, val_y))]
history_log = [test_model.fit(train_X, train_y, batch_size=512, epochs=1, callbacks=[history], validation_data=(val_X, val_y),weights=[embedding_matrix])]

# test_model.load_weights(weight_path)
# test_model.save("../input/best_model.h5")
pred_noemb_val_y = test_model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_noemb_val_y>thresh).astype(int))))
pred_noemb_test_y = test_model.predict([test_X], batch_size=1024, verbose=1)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve
y_pred_keras = test_model.predict([val_X], batch_size=1024, verbose=1).ravel()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(val_y, y_pred_keras)
from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()



