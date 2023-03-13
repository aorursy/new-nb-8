import numpy as np 

import pandas as pd 

import tensorflow as tf

import sys

import collections

sys.path.extend(['../input/bert-joint-baseline/'])



import bert_utils

import modeling 



import tokenization

import json



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

on_kaggle_server = os.path.exists('/kaggle')

nq_test_file = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl' 

nq_train_file = '../input/tensorflow2-question-answering/simplified-nq-train.jsonl'

public_dataset = os.path.getsize(nq_test_file)<20_000_000

private_dataset = os.path.getsize(nq_test_file)>=20_000_000
if True:

    import importlib

    importlib.reload(bert_utils)
with open('../input/bert-joint-baseline/bert_config.json','r') as f:

    config = json.load(f)

print(json.dumps(config,indent=4))

class TDense(tf.keras.layers.Layer):

    def __init__(self,

                 output_size,

                 kernel_initializer=None,

                 bias_initializer="zeros",

                **kwargs):

        super().__init__(**kwargs)

        self.output_size = output_size

        self.kernel_initializer = kernel_initializer

        self.bias_initializer = bias_initializer

    def build(self,input_shape):

        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

        if not (dtype.is_floating or dtype.is_complex):

          raise TypeError("Unable to build `TDense` layer with "

                          "non-floating point (and non-complex) "

                          "dtype %s" % (dtype,))

        input_shape = tf.TensorShape(input_shape)

        if tf.compat.dimension_value(input_shape[-1]) is None:

          raise ValueError("The last dimension of the inputs to "

                           "`TDense` should be defined. "

                           "Found `None`.")

        last_dim = tf.compat.dimension_value(input_shape[-1])

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3, axes={-1: last_dim})

        self.kernel = self.add_weight(

            "kernel",

            shape=[self.output_size,last_dim],

            initializer=self.kernel_initializer,

            dtype=self.dtype,

            trainable=True)

        self.bias = self.add_weight(

            "bias",

            shape=[self.output_size],

            initializer=self.bias_initializer,

            dtype=self.dtype,

            trainable=True)

        super(TDense, self).build(input_shape)

    def call(self,x):

        return tf.matmul(x,self.kernel,transpose_b=True)+self.bias

    

def mk_model(config):

    seq_len = config['max_position_embeddings']

    unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')

    input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_ids')

    input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')

    segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')

    BERT = modeling.BertModel(config=config,name='bert')

    pooled_output, sequence_output = BERT(input_word_ids=input_ids,

                                          input_mask=input_mask,

                                          input_type_ids=segment_ids)

    

    logits = TDense(2,name='logits')(sequence_output)

    start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')

    start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')

    end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')

    

    ans_type      = TDense(5,name='ans_type')(pooled_output)

    return tf.keras.Model([input_ for input_ in [unique_id,input_ids,input_mask,segment_ids] 

                           if input_ is not None],

                          [unique_id,start_logits,end_logits,ans_type],

                          name='bert-baseline')    
small_config = config.copy()

small_config['vocab_size']=16

small_config['hidden_size']=64

small_config['max_position_embeddings'] = 32

small_config['num_hidden_layers'] = 4

small_config['num_attention_heads'] = 4

small_config['intermediate_size'] = 256

small_config
model= mk_model(config)
model.summary()
cpkt = tf.train.Checkpoint(model=model)

cpkt.restore('../input/bert-joint-baseline/model_cpkt-1').assert_consumed()
class DummyObject:

    def __init__(self,**kwargs):

        self.__dict__.update(kwargs)



FLAGS=DummyObject(skip_nested_contexts=True,

                 max_position=50,

                 max_contexts=48,

                 max_query_length=64,

                 max_seq_length=512,

                 doc_stride=128,

                 include_unknowns=-1.0,

                 n_best_size=20,

                 max_answer_length=30)
import tqdm

eval_records = "../input/bert-joint-baseline/nq-test.tfrecords"

#nq_test_file = '../input/tensorflow2-question-answering/simplified-nq-test.jsonl'

if on_kaggle_server and private_dataset:

    eval_records='nq-test.tfrecords'

if not os.path.exists(eval_records):

    # tf2baseline.FLAGS.max_seq_length = 512

    eval_writer = bert_utils.FeatureWriter(

        filename=os.path.join(eval_records),

        is_training=False)



    tokenizer = tokenization.FullTokenizer(vocab_file='../input/bert-joint-baseline/vocab-nq.txt', 

                                           do_lower_case=True)



    features = []

    convert = bert_utils.ConvertExamples2Features(tokenizer=tokenizer,

                                                   is_training=False,

                                                   output_fn=eval_writer.process_feature,

                                                   collect_stat=False)



    n_examples = 0

    tqdm_notebook= tqdm.tqdm_notebook if not on_kaggle_server else None

    for examples in bert_utils.nq_examples_iter(input_file=nq_test_file, 

                                           is_training=False,

                                           tqdm=tqdm_notebook):

        for example in examples:

            n_examples += convert(example)



    eval_writer.close()

    print('number of test examples: %d, written to file: %d' % (n_examples,eval_writer.num_features))
seq_length = FLAGS.max_seq_length #config['max_position_embeddings']

name_to_features = {

      "unique_id": tf.io.FixedLenFeature([], tf.int64),

      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),

      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),

      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),

  }



def _decode_record(record, name_to_features=name_to_features):

    """Decodes a record to a TensorFlow example."""

    example = tf.io.parse_single_example(serialized=record, features=name_to_features)



    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.

    # So cast all int64 to int32.

    for name in list(example.keys()):

        t = example[name]

        if name != 'unique_id': #t.dtype == tf.int64:

            t = tf.cast(t, dtype=tf.int32)

        example[name] = t



    return example



def _decode_tokens(record):

    return tf.io.parse_single_example(serialized=record, 

                                      features={

                                          "unique_id": tf.io.FixedLenFeature([], tf.int64),

                                          "token_map" :  tf.io.FixedLenFeature([seq_length], tf.int64)

                                      })

      

raw_ds = tf.data.TFRecordDataset(eval_records)

token_map_ds = raw_ds.map(_decode_tokens)

decoded_ds = raw_ds.map(_decode_record)

ds = decoded_ds.batch(batch_size=16,drop_remainder=False)
result=model.predict_generator(ds,verbose=1)
np.savez_compressed('bert-joint-baseline-output.npz',

                    **dict(zip(['uniqe_id','start_logits','end_logits','answer_type_logits'],

                               result)))
Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])
class ScoreSummary(object):

  def __init__(self):

    self.predicted_label = None

    self.short_span_score = None

    self.cls_token_score = None

    self.answer_type_logits = None
class EvalExample(object):

  """Eval data available for a single example."""

  def __init__(self, example_id, candidates):

    self.example_id = example_id

    self.candidates = candidates

    self.results = {}

    self.features = {}
def get_best_indexes(logits, n_best_size):

  """Get the n-best logits from a list."""

  index_and_score = sorted(

      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)

  best_indexes = []

  for i in range(len(index_and_score)):

    if i >= n_best_size:

      break

    best_indexes.append(index_and_score[i][0])

  return best_indexes



def top_k_indices(logits,n_best_size,token_map):

    indices = np.argsort(logits[1:])+1

    indices = indices[token_map[indices]!=-1]

    return indices[-n_best_size:]
def compute_predictions(example):

  """Converts an example into an NQEval object for evaluation."""

  predictions = []

  n_best_size = FLAGS.n_best_size

  max_answer_length = FLAGS.max_answer_length

  i = 0

  for unique_id, result in example.results.items():

    if unique_id not in example.features:

      raise ValueError("No feature found with unique_id:", unique_id)

    token_map = np.array(example.features[unique_id]["token_map"]) #.int64_list.value

    start_indexes = top_k_indices(result.start_logits,n_best_size,token_map)

    if len(start_indexes)==0:

        continue

    end_indexes   = top_k_indices(result.end_logits,n_best_size,token_map)

    if len(end_indexes)==0:

        continue

    indexes = np.array(list(np.broadcast(start_indexes[None],end_indexes[:,None])))  

    indexes = indexes[(indexes[:,0]<indexes[:,1])*(indexes[:,1]-indexes[:,0]<max_answer_length)]

    for start_index,end_index in indexes:

        summary = ScoreSummary()

        summary.short_span_score = (

            result.start_logits[start_index] +

            result.end_logits[end_index])

        summary.cls_token_score = (

            result.start_logits[0] + result.end_logits[0])

        summary.answer_type_logits = result.answer_type_logits-result.answer_type_logits.mean()

        start_span = token_map[start_index]

        end_span = token_map[end_index] + 1



        # Span logits minus the cls logits seems to be close to the best.

        score = summary.short_span_score - summary.cls_token_score

        predictions.append((score, i, summary, start_span, end_span))

        i += 1 # to break ties



  # Default empty prediction.

  score = -10000.0

  short_span = Span(-1, -1)

  long_span  = Span(-1, -1)

  summary    = ScoreSummary()



  if predictions:

    score, _, summary, start_span, end_span = sorted(predictions, reverse=True)[0]

    short_span = Span(start_span, end_span)

    for c in example.candidates:

      start = short_span.start_token_idx

      end = short_span.end_token_idx

      ## print(c['top_level'],c['start_token'],start,c['end_token'],end)

      if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:

        long_span = Span(c["start_token"], c["end_token"])

        break



  summary.predicted_label = {

      "example_id": int(example.example_id),

      "long_answer": {

          "start_token": int(long_span.start_token_idx),

          "end_token": int(long_span.end_token_idx),

          "start_byte": -1,

          "end_byte": -1

      },

      "long_answer_score": float(score),

      "short_answers": [{

          "start_token": int(short_span.start_token_idx),

          "end_token": int(short_span.end_token_idx),

          "start_byte": -1,

          "end_byte": -1

      }],

      "short_answer_score": float(score),

      "yes_no_answer": "NONE",

      "answer_type_logits": summary.answer_type_logits.tolist(),

      # here:

      "answer_type": int(np.argmax(summary.answer_type_logits))

  }



  return summary
def compute_pred_dict(candidates_dict, dev_features, raw_results,tqdm=None):

    """Computes official answer key from raw logits."""

    raw_results_by_id = [(int(res.unique_id),1, res) for res in raw_results]



    examples_by_id = [(int(k),0,v) for k, v in candidates_dict.items()]

  

    features_by_id = [(int(d['unique_id']),2,d) for d in dev_features] 

  

    # Join examples with features and raw results.

    examples = []

    print('merging examples...')

    merged = sorted(examples_by_id + raw_results_by_id + features_by_id)

    print('done.')

    for idx, type_, datum in merged:

        if type_==0: #isinstance(datum, list):

            examples.append(EvalExample(idx, datum))

        elif type_==2: #"token_map" in datum:

            examples[-1].features[idx] = datum

        else:

            examples[-1].results[idx] = datum



    # Construct prediction objects.

    print('Computing predictions...')

   

    nq_pred_dict = {}

    #summary_dict = {}

    if tqdm is not None:

        examples = tqdm(examples)

    for e in examples:

        summary = compute_predictions(e)

        #summary_dict[e.example_id] = summary

        nq_pred_dict[e.example_id] = summary.predicted_label



    return nq_pred_dict

def read_candidates_from_one_split(input_path):

  """Read candidates from a single jsonl file."""

  candidates_dict = {}

  print("Reading examples from: %s" % input_path)

  if input_path.endswith(".gz"):

    with gzip.GzipFile(fileobj=tf.io.gfile.GFile(input_path, "rb")) as input_file:

      for index, line in enumerate(input_file):

        e = json.loads(line)

        candidates_dict[e["example_id"]] = e["long_answer_candidates"]

        

  else:

    with tf.io.gfile.GFile(input_path, "r") as input_file:

      for index, line in enumerate(input_file):

        e = json.loads(line)

        candidates_dict[e["example_id"]] = e["long_answer_candidates"]

        # candidates_dict['question'] = e['question_text']

  return candidates_dict
def read_candidates(input_pattern):

  """Read candidates with real multiple processes."""

  input_paths = tf.io.gfile.glob(input_pattern)

  final_dict = {}

  for input_path in input_paths:

    final_dict.update(read_candidates_from_one_split(input_path))

  return final_dict
all_results = [bert_utils.RawResult(*x) for x in zip(*result)]

    

print ("Going to candidates file")



candidates_dict = read_candidates('../input/tensorflow2-question-answering/simplified-nq-test.jsonl')



print ("setting up eval features")



eval_features = list(token_map_ds)



print ("compute_pred_dict")



tqdm_notebook= tqdm.tqdm_notebook

nq_pred_dict = compute_pred_dict(candidates_dict, 

                                       eval_features,

                                       all_results,

                                      tqdm=tqdm_notebook)



predictions_json = {"predictions": list(nq_pred_dict.values())}



print ("writing json")



with tf.io.gfile.GFile('predictions.json', "w") as f:

    json.dump(predictions_json, f, indent=4)
def create_short_answer(entry):

    answer = []    

    if entry['answer_type'] == 0:

        return ""

    

    elif entry['answer_type'] == 1:

        return 'YES'

    

    elif entry['answer_type'] == 2:

        return 'NO'

        

    elif entry["short_answer_score"] < 1.5:

        return ""

    

    else:

        for short_answer in entry["short_answers"]:

            if short_answer["start_token"] > -1:

                answer.append(str(short_answer["start_token"]) + ":" + str(short_answer["end_token"]))

    

        return " ".join(answer)



def create_long_answer(entry):

    

    answer = []

    

    if entry['answer_type'] == 0:

        return ''

    

    elif entry["long_answer_score"] < 1.5:

        return ""



    elif entry["long_answer"]["start_token"] > -1:

        answer.append(str(entry["long_answer"]["start_token"]) + ":" + str(entry["long_answer"]["end_token"]))

        return " ".join(answer)
test_answers_df = pd.read_json("../working/predictions.json")

for var_name in ['long_answer_score','short_answer_score','answer_type']:

    test_answers_df[var_name] = test_answers_df['predictions'].apply(lambda q: q[var_name])

test_answers_df["long_answer"] = test_answers_df["predictions"].apply(create_long_answer)

test_answers_df["short_answer"] = test_answers_df["predictions"].apply(create_short_answer)

test_answers_df["example_id"] = test_answers_df["predictions"].apply(lambda q: str(q["example_id"]))



long_answers = dict(zip(test_answers_df["example_id"], test_answers_df["long_answer"]))

short_answers = dict(zip(test_answers_df["example_id"], test_answers_df["short_answer"]))



test_answers_df.head()
sample_submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")



long_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_long")].apply(lambda q: long_answers[q["example_id"].replace("_long", "")], axis=1)

short_prediction_strings = sample_submission[sample_submission["example_id"].str.contains("_short")].apply(lambda q: short_answers[q["example_id"].replace("_short", "")], axis=1)



sample_submission.loc[sample_submission["example_id"].str.contains("_long"), "PredictionString"] = long_prediction_strings

sample_submission.loc[sample_submission["example_id"].str.contains("_short"), "PredictionString"] = short_prediction_strings

sample_submission.to_csv('submission.csv', index=False)
sample_submission.head()
yes_answers = sample_submission[sample_submission['PredictionString'] == 'YES']

yes_answers
no_answers = sample_submission[sample_submission['PredictionString'] == 'NO']

no_answers
blank_answers = sample_submission[sample_submission['PredictionString'] == '']

blank_answers.head()
blank_answers.count()