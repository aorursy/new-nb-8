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
import re

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import string

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords



from collections import defaultdict



from transformers import BertTokenizer



import tensorflow as tf
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

#from kaggle_datasets import KaggleDatasets

import transformers

from tqdm.notebook import tqdm

from tokenizers import BertWordPieceTokenizer

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional

from tensorflow.keras.models import Model

from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.layers import concatenate

from tensorflow.compat.v1.keras.layers import CuDNNLSTM

import gc

import os

import numpy as np

import pandas as pd

from transformers import BertTokenizer
data_directory = '/kaggle/input/tweet-sentiment-extraction/'
train_df = pd.read_csv(data_directory + 'train.csv')

test_df = pd.read_csv(data_directory + 'test.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')



# Save the loaded tokenizer locally

save_path = '/kaggle/working/bert_base_cased/'

if not os.path.exists(save_path):

    os.makedirs(save_path)

tokenizer.save_pretrained(save_path)



# Reload it with the huggingface tokenizers library

fast_tokenizer = BertWordPieceTokenizer(save_path + 'vocab.txt', lowercase=False)

fast_tokenizer
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=128):

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
x_train = fast_encode(train_df.text.astype(str), fast_tokenizer, maxlen=128)

x_test = fast_encode(test_df.text.astype(str),fast_tokenizer,maxlen=128)
transformer_layer = transformers.TFBertForQuestionAnswering.from_pretrained('bert-base-cased')
def create_targets(df):

    df['t_text'] = df['text'].apply(lambda x: tokenizer.tokenize(str(x)))

    df['t_selected_text'] = df['selected_text'].apply(lambda x: tokenizer.tokenize(str(x)))

    def func(row):

        x,y = row['t_text'],row['t_selected_text'][:]

        for offset in range(len(x)):

            d = dict(zip(x[offset:],y))

            #when k = v that means we found the offset

            check = [k==v for k,v in d.items()]

            if all(check)== True:

                break 

        return [0]*offset + [1]*len(y) + [0]* (len(x)-offset-len(y))

    df['targets'] = df.apply(func,axis=1)

    return df



train_df = create_targets(train_df)



print('MAX_SEQ_LENGTH_TEXT', max(train_df['t_text'].apply(len)))

print('MAX_TARGET_LENGTH',max(train_df['targets'].apply(len)))

MAX_TARGET_LEN=108
train_df['targets'] = train_df['targets'].apply(lambda x :x + [0] * (MAX_TARGET_LEN-len(x)))

targets=np.asarray(train_df['targets'].values.tolist())
lb=LabelEncoder()

sent_train=lb.fit_transform(train_df['sentiment'])

sent_test=lb.fit_transform(test_df['sentiment'])
def new_model(transformer_layer):

    

    inp = Input(shape=(128, ))

    inp2= Input(shape=(1,))

    

    embedding_matrix=transformer_layer.weights[0].numpy()



    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)



    x = LSTM(150, return_sequences=True,name='lstm_layer',)(x)

    x = LSTM(100, return_sequences=False,name='lstm_layer-2',)(x)

    

    y =Dense(10,activation='relu')(inp2)

    x= concatenate([x,y])

    

    x = Dense(MAX_TARGET_LEN,activation='sigmoid')(x)



    model = Model(inputs=[inp,inp2], outputs=x)



    model.compile(loss='binary_crossentropy',

                      optimizer='adam')



    

    return model
model=new_model(transformer_layer)

history=model.fit([x_train,sent_train],targets,epochs=3)
predictions=model.predict([x_test,sent_test])
def convert_output(sub,predictions):

    preds=[]

    for i,row in enumerate(sub['text']):



        text,target=row.lower(),predictions[i].tolist()

        target=np.round(target).tolist()

        try:

            start,end=target.index(1),target[::-1].index(1)

            text_list=tokenizer.tokenize(text)

            text_list=text_list+((108-len(text_list))*['pad'])

            start_w,end_w=text_list[start],text_list[-end]

            start=text.find(start_w.replace("#",'',1))    ## remove # to match substring

            end=text.find(end_w.replace("#",''),start)

            #pred=' '.join([x for x in text_list[start:-end]])

            pred=text[start:end]

        except:

            pred=text

        

        preds.append(pred)

        

    return preds
prediction_text=convert_output(test_df,predictions)
len(prediction_text)
output_directory = '/kaggle/input'

sub=pd.read_csv(data_directory + "sample_submission.csv")

sub['selected_text']=prediction_text

sub.to_csv('/kaggle/working/submission.csv',index=False)

sub.head()