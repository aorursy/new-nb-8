import pandas as pd

import os

import numpy as np

import pandas as pd

import zipfile

from matplotlib import pyplot as plt


import sys

import datetime
#downloading weights and cofiguration file for the model

repo = 'model_repo'

with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:

    zip_ref.extractall(repo)



# Available pretrained model checkpoints:

#   uncased_L-12_H-768_A-12: uncased BERT base model

#   uncased_L-24_H-1024_A-16: uncased BERT large model

#   cased_L-12_H-768_A-12: cased BERT large model

#We will use the most basic of all of them

BERT_MODEL = 'uncased_L-12_H-768_A-12'

BERT_PRETRAINED_DIR = f'{repo}/uncased_L-12_H-768_A-12'

OUTPUT_DIR = f'{repo}/outputs'

print(f'***** Model output directory: {OUTPUT_DIR} *****')

print(f'***** BERT pretrained directory: {BERT_PRETRAINED_DIR} *****')




def _row_to_y(row):

    if row.loc['A-coref']:

        return '0'

    if row.loc['B-coref']:

        return '1'

    return '2'
from sklearn.model_selection import train_test_split



train_df =  pd.read_csv('gap-test.tsv', sep='\t')

dev_df = pd.read_csv('gap-development.tsv', sep='\t')

test_df = pd.read_csv('gap-validation.tsv', sep='\t')



train_df = pd.concat([train_df, test_df]) 

test_df = dev_df



train_lines, train_labels = train_df.Text.values, train_df.apply(_row_to_y, axis=1)

test_lines, test_labels = test_df.Text.values, test_df.apply(_row_to_y, axis=1)
import modeling

import optimization

import run_classifier

import tokenization

import tensorflow as tf





def create_examples(lines, set_type, labels=None):

#Generate data for the BERT model

    guid = f'{set_type}'

    examples = []

    if guid == 'train':

        for line, label in zip(lines, labels):

            text_a = line

            label = str(label)

            examples.append(

              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    else:

        for line in lines:

            text_a = line

            label = '0'

            examples.append(

              run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    return examples



# Model Hyper Parameters

TRAIN_BATCH_SIZE = 32

EVAL_BATCH_SIZE = 8

LEARNING_RATE = 2e-5

NUM_TRAIN_EPOCHS = 4.0

WARMUP_PROPORTION = 0.1

MAX_SEQ_LENGTH = 128

# Model configs

SAVE_CHECKPOINTS_STEPS = 1000 #if you wish to finetune a model on a larger dataset, use larger interval

# each checpoint weights about 1,5gb

ITERATIONS_PER_LOOP = 1000

NUM_TPU_CORES = 8

VOCAB_FILE = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')

CONFIG_FILE = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')

INIT_CHECKPOINT = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')

DO_LOWER_CASE = BERT_MODEL.startswith('uncased')



label_list = ['0', '1', '2']

tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

train_examples = create_examples(train_lines, 'train', labels=train_labels)



tpu_cluster_resolver = None #Since training will happen on GPU, we won't need a cluster resolver

#TPUEstimator also supports training on CPU and GPU. You don't need to define a separate tf.estimator.Estimator.

run_config = tf.contrib.tpu.RunConfig(

    cluster=tpu_cluster_resolver,

    model_dir=OUTPUT_DIR,

    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,

    tpu_config=tf.contrib.tpu.TPUConfig(

        iterations_per_loop=ITERATIONS_PER_LOOP,

        num_shards=NUM_TPU_CORES,

        per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2))



num_train_steps = int(

    len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)

num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)



model_fn = run_classifier.model_fn_builder(

    bert_config=modeling.BertConfig.from_json_file(CONFIG_FILE),

    num_labels=len(label_list),

    init_checkpoint=INIT_CHECKPOINT,

    learning_rate=LEARNING_RATE,

    num_train_steps=num_train_steps,

    num_warmup_steps=num_warmup_steps,

    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available  

    use_one_hot_embeddings=True)



estimator = tf.contrib.tpu.TPUEstimator(

    use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 

    model_fn=model_fn,

    config=run_config,

    train_batch_size=TRAIN_BATCH_SIZE,

    eval_batch_size=EVAL_BATCH_SIZE)
"""

Note: You might see a message 'Running train on CPU'. 

This really just means that it's running on something other than a Cloud TPU, which includes a GPU.

"""



# Train the model.

print('Please wait...')

train_features = run_classifier.convert_examples_to_features(

    train_examples, label_list, MAX_SEQ_LENGTH, tokenizer)

print('***** Started training at {} *****'.format(datetime.datetime.now()))

print('  Num examples = {}'.format(len(train_examples)))

print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))

tf.logging.info("  Num steps = %d", num_train_steps)

train_input_fn = run_classifier.input_fn_builder(

    features=train_features,

    seq_length=MAX_SEQ_LENGTH,

    is_training=True,

    drop_remainder=True)

estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

print('***** Finished training at {} *****'.format(datetime.datetime.now()))