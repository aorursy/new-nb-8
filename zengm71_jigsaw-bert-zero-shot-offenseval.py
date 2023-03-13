import os



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from tqdm.notebook import tqdm

from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    tokenizer.enable_truncation(max_length=maxlen)

    tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.encode_batch(text_chunk)

        all_ids.extend([enc.ids for enc in encs])

    

    return np.array(all_ids)
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
AUTO = tf.data.experimental.AUTOTUNE



# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')



# Configuration

EPOCHS = 3

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 128
# First load the real tokenizer

tokenizer = transformers.AutoTokenizer.from_pretrained("jplu/tf-xlm-roberta-large")

# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')

# Reload it with the huggingface tokenizers library

# fast_tokenizer = SentencePieceBPETokenizer('vocab.txt')

# fast_tokenizer
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

train1.head()
olid = pd.read_csv('/kaggle/input/olid2019/olid-training-v1.0.tsv', sep='\t')

olid = olid.rename(columns={"id": "id", "tweet": "comment_text", 'subtask_a': 'toxic'})

olid.toxic = (olid.toxic == 'OFF').astype(int)

olid['comment_text'] = olid['comment_text'].str.replace('@USER', '')

olid.head()
from sklearn.model_selection import train_test_split

olid_train, olid_test = train_test_split(olid, test_size=3240, random_state = 2020)

olid_train_1k = olid_train[0:1000]

olid_train_2k = olid_train[0:2000]

olid_train_5k = olid_train[0:5000]

train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=150000, random_state=0)

])
train.toxic.value_counts()
def fast_encode_xlm(texts, tokenizer, chunk_size=256, maxlen=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.batch_encode_plus(text_chunk, pad_to_max_length = True, max_length = maxlen)

        all_ids.extend(np.array(encs.input_ids))

    

    return np.array(all_ids)
x_train = fast_encode_xlm(train.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_valid = fast_encode_xlm(valid.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_test = fast_encode_xlm(test.content.astype(str), tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values
# olid_encode = fast_encode_xlm(olid.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_encode = fast_encode_xlm(olid_train.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_1k_encode = fast_encode_xlm(olid_train_1k.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_2k_encode = fast_encode_xlm(olid_train_2k.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_5k_encode = fast_encode_xlm(olid_train_5k.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_test_encode = fast_encode_xlm(olid_test.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)



y_olid_train = olid_train.toxic.values

y_olid_train_1k = olid_train_1k.toxic.values

y_olid_train_2k = olid_train_2k.toxic.values

y_olid_train_5k = olid_train_5k.toxic.values

y_olid_test = olid_test.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
train_dataset
olid_test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(olid_test_encode)

    .batch(BATCH_SIZE)

)

olid_test_dataset_wlabel = (

    tf.data.Dataset

    .from_tensor_slices((olid_test_encode, y_olid_test))

    .batch(BATCH_SIZE)

)

olid_train_1k_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_1k_encode, y_olid_train_1k))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

olid_train_2k_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_2k_encode, y_olid_train_2k))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

olid_train_5k_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_5k_encode, y_olid_train_5k))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

olid_train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_encode, y_olid_train))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

def build_model(transformer, max_len=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    

    return model

with strategy.scope():

    transformer_layer = (

        transformers.TFAutoModelWithLMHead

        .from_pretrained('jplu/tf-xlm-roberta-large')

    )

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()

n_steps = x_train.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=EPOCHS

)
# type(model)

# model.save_weights('model_0k.h5') 
sub['toxic'] = model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False)

sub.describe()
from sklearn.metrics import roc_auc_score
olid_test.toxic_predict = model.predict(olid_test_dataset, verbose=1)

olid_test.to_csv('olid_test_0shot.csv', index=False)



roc_auc_score(y_true = olid_test.toxic, y_score = olid_test.toxic_predict)
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, log_loss, f1_score

import seaborn as sns

import numpy as np

from matplotlib import pyplot as plt



def Find_Optimal_Cutoff(target, predicted):

    """ Find the optimal probability cutoff point for a classification model related to event rate

    Parameters

    ----------

    target : Matrix with dependent or target data, where rows are observations



    predicted : Matrix with predicted data, where rows are observations



    Returns

    -------     

    list type, with optimal cutoff value



    """

    fpr, tpr, threshold = roc_curve(target, predicted)

    i = np.arange(len(tpr)) 

    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})

    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]



    return list(roc_t['threshold']) 



threshold = Find_Optimal_Cutoff(olid_test.toxic, olid_test.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in olid_test.toxic_predict]
f1_score(y_true = olid_test.toxic, y_pred = olid_test.toxic_predict_binary)

def plot_matrix(target, predicted_binary, name):

    matrix = confusion_matrix(target, predicted_binary)

    TN, FP, FN, TP = matrix.ravel()

    if (TP + FP > 0) and (TP + FN > 0):

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)

        F =  2 * (precision*recall) / (precision + recall)

    else:

        F = 0

    cm_df = pd.DataFrame(matrix,

                         index = ['Nagative', 'Positive'], 

                         columns = ['Nagative', 'Positive'])

    subtitle = 'Precision ' + str(round(precision, 2)) + ' Recall ' + str(round(recall, 2))

    fig, ax = plt.subplots(figsize=(8,6))

    ax = sns.heatmap(cm_df, annot=True, fmt="d")

    bottom, top = ax.get_ylim()

    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.title('Confusion Matrix - ' + name + "\n" + subtitle)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

plot_matrix(olid_test.toxic, olid_test.toxic_predict_binary, name = 'Zero-Shot')

model_1k = model

model_1k.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

n_steps = olid_train_1k.shape[0] // BATCH_SIZE

train_history_1k = model_1k.fit(

    olid_train_1k_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_test_dataset_wlabel,

    epochs=10

)



olid_test1k = olid_test

olid_test1k.toxic_predict = model_1k.predict(olid_test_dataset, verbose=1)

olid_test1k.to_csv('olid_test_1k.csv', index=False)



print('1k roc is ' + str(roc_auc_score(y_true = olid_test1k.toxic, y_score = olid_test1k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test1k.toxic, olid_test1k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test1k.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in olid_test1k.toxic_predict]



print('1k f1-score is ' + str(f1_score(y_true = olid_test1k.toxic, y_pred = olid_test1k.toxic_predict_binary)))



del model_1k
model_2k = model

model_2k.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])



n_steps = olid_train_2k.shape[0] // BATCH_SIZE

train_history_2k = model_2k.fit(

    olid_train_2k_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_test_dataset_wlabel,

    epochs=10

)



olid_test2k = olid_test

olid_test2k.toxic_predict = model_2k.predict(olid_test_dataset, verbose=1)

olid_test2k.to_csv('olid_test_2k.csv', index=False)



print('2k roc is ' + str(roc_auc_score(y_true = olid_test2k.toxic, y_score = olid_test2k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test2k.toxic, olid_test2k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test2k.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in olid_test2k.toxic_predict]



print('2k f1-score is ' + str(f1_score(y_true = olid_test2k.toxic, y_pred = olid_test2k.toxic_predict_binary)))



del model_2k
model_5k = model

model_5k.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])



n_steps = olid_train_5k.shape[0] // BATCH_SIZE

train_history_5 = model_5k.fit(

    olid_train_5k_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_test_dataset_wlabel,

    epochs=10

)



olid_test5k = olid_test

olid_test5k.toxic_predict = model_5k.predict(olid_test_dataset, verbose=1)

olid_test5k.to_csv('olid_test_5k.csv', index=False)



print('5k roc is ' + str(roc_auc_score(y_true = olid_test5k.toxic, y_score = olid_test5k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test5k.toxic, olid_test5k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test5k.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in olid_test5k.toxic_predict]



print('5k f1-score is ' + str(f1_score(y_true = olid_test5k.toxic, y_pred = olid_test5k.toxic_predict_binary)))



del model_5k
model_10k = model

model_10k.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])



n_steps = olid_train.shape[0] // BATCH_SIZE

train_history_10 = model_10k.fit(

    olid_train_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_test_dataset_wlabel,

    epochs=10

)



olid_test10k = olid_test

olid_test10k.toxic_predict = model_10k.predict(olid_test_dataset, verbose=1)

olid_test10k.to_csv('olid_test_10k.csv', index=False)



print('10k roc is ' + str(roc_auc_score(y_true = olid_test10k.toxic, y_score = olid_test10k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test10k.toxic, olid_test10k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test10k.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in olid_test10k.toxic_predict]



print('10k f1-score is ' + str(f1_score(y_true = olid_test10k.toxic, y_pred = olid_test10k.toxic_predict_binary)))



del model_10k