#!/usr/bin/env python
# coding: utf-8



import os
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from tqdm import tqdm_notebook as tqdm
from torch import nn
import copy
import transformers
print(transformers.__version__)
from transformers.modeling_t5 import *
import json
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import *
import torch
# import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
from torch.optim import Adagrad, Adamax
from transformers.modeling_utils import *
from scipy.special import softmax
import argparse
import itertools
from collections import OrderedDict
from fuzzywuzzy import fuzz
import operator
from transformers.tokenization_bert import BasicTokenizer
from tqdm import tqdm_notebook as tqdm
import regex as re




get_ipython().system('ls -halt ../input/bart-drop-head')




from transformers import *
from transformers.modeling_bart import PretrainedBartModel
import torch
from torch import nn
import torch.nn.functional as F

import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

class PoolerStartLogits(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, p_mask=None):
        x = self.dense(hidden_states).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x


class PoolerEndLogits(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_0 = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps if hasattr(config,"layer_norm_eps") else 1e-5)
        self.dense_1 = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states, start_states=None, start_positions=None, p_mask=None):
        assert (
            start_states is not None or start_positions is not None
        ), "One of start_states, start_positions should be not None"
        if start_positions is not None:
            slen, hsz = hidden_states.shape[-2:]
            start_positions = start_positions[:, None, None].expand(-1, -1, hsz)  # shape (bsz, 1, hsz)
            start_states = hidden_states.gather(-2, start_positions)  # shape (bsz, 1, hsz)
            start_states = start_states.expand(-1, slen, -1)  # shape (bsz, slen, hsz)

        x = self.dense_0(torch.cat([hidden_states, start_states], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        x = self.dense_1(x).squeeze(-1)

        if p_mask is not None:
            if next(self.parameters()).dtype == torch.float16:
                x = x * (1 - p_mask) - 65500 * p_mask
            else:
                x = x * (1 - p_mask) - 1e30 * p_mask

        return x

class RobertaForSentimentExtraction(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
#         self.transformer = self.roberta
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.init_weights()

    def forward(
        self, beam_size=1, drop_head=True,
        input_ids=None,attention_mask=None,token_type_ids=None,input_mask=None,position_ids=None,
        head_mask=None,inputs_embeds=None,start_positions=None,end_positions=None,p_mask=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=None,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        if drop_head:
            hidden_states = outputs[2][-3]
        else:
            hidden_states = outputs[0]
#        hidden_states = torch.cat((outputs[2][-1],outputs[2][-2], outputs[2][-3], outputs[2][-4]),-1)
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        outputs = outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = total_loss

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)
#            start_log_probs = F.sigmoid(start_logits)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, beam_size, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)
#            end_log_probs = F.sigmoid(end_logits)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, beam_size, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, beam_size * beam_size)
            end_top_index = end_top_index.view(-1, beam_size * beam_size)

            outputs = start_top_log_probs, start_top_index, end_top_log_probs, end_top_index

        return outputs

class BartForSentimentExtraction(PretrainedBartModel):

    def __init__(self, config):
        super().__init__(config)
        self.start_logits = PoolerStartLogits(config)
        self.end_logits = PoolerEndLogits(config)
        self.model = BartModel(config)
#         self.transformer = self.model
        self.init_weights()

    def forward(
        self, beam_size=1, drop_head=True,
        input_ids=None,attention_mask=None,token_type_ids=None,input_mask=None,position_ids=None,
        head_mask=None,inputs_embeds=None,start_positions=None,end_positions=None,p_mask=None
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        
        if drop_head:
            hidden_states = outputs[1][-3]
        else:
            hidden_states = outputs[0]
#        hidden_states = torch.cat((outputs[2][-1],outputs[2][-2], outputs[2][-3], outputs[2][-4]),-1)
        start_logits = self.start_logits(hidden_states, p_mask=p_mask)

        outputs = outputs[1:]  # Keep mems, hidden states, attentions if there are in it

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, let's remove the dimension added by batch splitting
            for x in (start_positions, end_positions):
                if x is not None and x.dim() > 1:
                    x.squeeze_(-1)

            # during training, compute the end logits based on the ground truth of the start position
            end_logits = self.end_logits(hidden_states, start_positions=start_positions, p_mask=p_mask)

            loss_fct = CrossEntropyLoss()

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = total_loss

        else:
            # during inference, compute the end logits based on beam search
            bsz, slen, hsz = hidden_states.size()
            start_log_probs = F.softmax(start_logits, dim=-1)  # shape (bsz, slen)

            start_top_log_probs, start_top_index = torch.topk(
                start_log_probs, beam_size, dim=-1
            )  # shape (bsz, start_n_top)
            start_top_index_exp = start_top_index.unsqueeze(-1).expand(-1, -1, hsz)  # shape (bsz, start_n_top, hsz)
            start_states = torch.gather(hidden_states, -2, start_top_index_exp)  # shape (bsz, start_n_top, hsz)
            start_states = start_states.unsqueeze(1).expand(-1, slen, -1, -1)  # shape (bsz, slen, start_n_top, hsz)

            hidden_states_expanded = hidden_states.unsqueeze(2).expand_as(
                start_states
            )  # shape (bsz, slen, start_n_top, hsz)
            p_mask = p_mask.unsqueeze(-1) if p_mask is not None else None
            end_logits = self.end_logits(hidden_states_expanded, start_states=start_states, p_mask=p_mask)
            end_log_probs = F.softmax(end_logits, dim=1)  # shape (bsz, slen, start_n_top)

            end_top_log_probs, end_top_index = torch.topk(
                end_log_probs, beam_size, dim=1
            )  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.view(-1, beam_size * beam_size)
            end_top_index = end_top_index.view(-1, beam_size * beam_size)

            outputs = start_top_log_probs, start_top_index, end_top_log_probs, end_top_index

        return outputs




batch_size = 64
beam_size = 3
max_sequence_length = 128




def find_best_combinations(start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, valid_start= 0, valid_end=512):
    best = (valid_start, valid_end - 1)
    best_score = -9999
    for i in range(len(start_top_log_probs)):
        for j in range(end_top_log_probs.shape[0]):
            if valid_start <= start_top_index[i] < valid_end and valid_start <= end_top_index[j,i] < valid_end and start_top_index[i] < end_top_index[j,i]:
                score = start_top_log_probs[i] * end_top_log_probs[j,i]
                if score > best_score:
                    best = (start_top_index[i],end_top_index[j,i])
                    best_score = score
    return best

def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    try:
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        return 0

def fuzzy_match(x,y,weights=None):
    l1 = len(x.split())
    matches = dict()
    x_ = x.split()
    if type(y) is str:
        y = [y]
    for curr_length in range(l1 + 1):
        for i in range(l1 + 1 - curr_length):
            sub_x = ' '.join(x_[i:i+curr_length])
            if sub_x not in matches:
                matches[sub_x] = np.average([fuzz.ratio(sub_x,y_) for y_ in y],weights=weights)
    if len(matches) == 0:
        return None, x
    return matches, sorted(matches.items(), key=operator.itemgetter(1))[-1][0]
    
def ensemble_v0(context, predictions,weights=None):
    context = context.split()
    scores = dict()
    for i,j in itertools.combinations(range(len(context) + 1),r=2):
        curr_context = ' '.join(context[i:j])
        scores[curr_context] = np.average([jaccard(curr_context, p) for p in predictions],weights=weghts)
    best_score = np.max([val for val in scores.values()])
    has_tie = np.sum([val == best_score for val in scores.values()]) > 1
    if not has_tie:
        for key, val in scores.items():
            if val == best_score:
                return key
    else:
        keys = [key for key, val in scores.items() if val == best_score]
#         return keys[np.argmax([jaccard(key,predictions[0]) for key in keys])]
        return keys[np.argmax([len(key.split()) for key in keys])]

def ensemble(context, predictions,slow_mode=False,weights=None):
    starts = []
    ends = []
    for p in predictions:
        if p in context:
            start = context.index(p)
            starts.append(start)
            ends.append(start+len(p))
    if len(starts) == 0:
        print(context)
        return ensemble_v0(context, predictions)
    scores = dict()
    context = context[np.min(starts):np.max(ends)]
    for i,j in itertools.combinations(range(len(context) + 1),r=2):
        if slow_mode or len(context.split()) == 1 or             (not ((i > 0 and context[i-1].isalnum() and context[i].isalnum()) or (j < len(context) and j > 1 and context[j-1].isalnum() and context[j].isalnum()))):
            curr_context = context[i:j].strip()
            scores[curr_context] = np.average([jaccard(curr_context, p) for p in predictions],weights=weights)
    for pred in predictions:
        if pred not in scores:
            scores[pred] = np.average([jaccard(pred, p) for p in predictions],weights=weights)
    best_score = np.max([val for val in scores.values()])
    has_tie = np.sum([val == best_score for val in scores.values()]) > 1
    if not has_tie:
        for key, val in scores.items():
            if val == best_score:
                return key
    else:
        keys = [key for key, val in scores.items() if val == best_score]
#         return keys[np.argmax([jaccard(key,predictions[0]) for key in keys])]
        return keys[np.argmax([len(key.split()) for key in keys])]

basic_tokenizer = BasicTokenizer(do_lower_case=False)
def fix_spaces(t):
    for i,item in enumerate(t):
        re_res = re.search('\s+$', item)
        if bool(re_res) & (i < len(t)-1):
            sp = re_res.span()
            t[i+1] = t[i][sp[0]:] + t[i+1]
            t[i] = t[i][:sp[0]]
    return t
def roberta_tokenize_v2(tokenizer, line):
    tokenized_line = []
    line2 = basic_tokenizer._run_split_on_punc(line)
    line2 = fix_spaces(line2)
    for item in line2:
        sub_word_tokens = tokenizer.tokenize(item)
        tokenized_line += sub_word_tokens
    return tokenized_line


def convert_lines(tokenizer, df, max_sequence_length = 512):
    pad_token_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    outputs = np.zeros((len(df), max_sequence_length))
    type_outputs = np.zeros((len(df), max_sequence_length))
    position_outputs = np.zeros((len(df), 2))
    extracted = []
    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        input_ids_0 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.sentiment)) 
        input_ids_1 = tokenizer.convert_tokens_to_ids(roberta_tokenize_v2(tokenizer, row.text)) 
        input_ids = [tokenizer.cls_token_id, ]+ input_ids_0 +  [tokenizer.sep_token_id,] +input_ids_1 + [tokenizer.sep_token_id, ]
        token_type_ids = [0,]*(len(input_ids_0) + 1) + [1,]*(len(input_ids_1) + 2)
        if len(input_ids) > max_sequence_length: 
#            input_ids = input_ids[:max_sequence_length//2] + input_ids[-max_sequence_length//2:] 
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = tokenizer.sep_token_id
            token_type_ids = token_type_ids[:max_sequence_length]
        else:
            input_ids = input_ids + [pad_token_idx, ]*(max_sequence_length - len(input_ids))
            token_type_ids = token_type_ids + [pad_token_idx, ]*(max_sequence_length - len(token_type_ids))
        assert len(input_ids) == len(token_type_ids)
        outputs[idx,:max_sequence_length] = np.array(input_ids)
        type_outputs[idx,:] = token_type_ids
    return outputs, type_outputs




# test_df = pd.read_csv("./data/test_holdout.csv")
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["sep_text"] = test_df.text.apply(lambda x: " ".join(x.split()).lower())
def get_predictions(x_test, x_type_test, model, is_xlnet=False,drop_head=True):
    all_start_top_log_probs = None
    all_start_top_index = None
    all_end_top_log_probs = None
    all_end_top_index = None
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(x_test,dtype=torch.long), torch.tensor(x_type_test,dtype=torch.long))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader),total=len(test_loader),leave=False)
        for i, items in pbar:
            x_batch, x_type_batch = items
            attention_mask = x_batch != tokenizer.pad_token_id
            p_mask = torch.zeros(x_batch.shape,dtype=torch.float32)
            p_mask[x_batch == tokenizer.pad_token_id] = 1.0
            p_mask[x_batch == tokenizer.cls_token_id] = 1.0
            if is_xlnet:
                attention_mask = attention_mask.float()
                p_mask[:,:2] = 1.0
            else:
                p_mask[:,:3] = 1.0
            start_top_log_probs, start_top_index, end_top_log_probs, end_top_index = model(input_ids=x_batch.cuda(), attention_mask=attention_mask.cuda(),                                                 token_type_ids=x_type_batch.cuda(), beam_size=beam_size, p_mask=p_mask.cuda(), drop_head=drop_head)
            start_top_log_probs = start_top_log_probs.detach().cpu().numpy()
            start_top_index = start_top_index.detach().cpu().numpy()
            end_top_log_probs = end_top_log_probs.detach().cpu().numpy()
            end_top_index = end_top_index.detach().cpu().numpy()

            all_start_top_log_probs = start_top_log_probs if all_start_top_log_probs is None else np.concatenate([all_start_top_log_probs, start_top_log_probs])
            all_start_top_index = start_top_index if all_start_top_index is None else np.concatenate([all_start_top_index, start_top_index])
            all_end_top_log_probs = end_top_log_probs if all_end_top_log_probs is None else np.concatenate([all_end_top_log_probs, end_top_log_probs])
            all_end_top_index = end_top_index if all_end_top_index is None else np.concatenate([all_end_top_index, end_top_index])

    return all_start_top_log_probs,all_start_top_index,all_end_top_log_probs,all_end_top_index

def load_and_fix_state(model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_selected_texts(x_test, tokenizer, offset=3, is_xlnet=False):
    selected_texts = []
    for i_, x in tqdm(enumerate(x_test),total=len(x_test)):
        real_length = np.sum(x != tokenizer.pad_token_id)
        if is_xlnet:
            real_length -= 1
        best_start, best_end = find_best_combinations(all_start_top_log_probs[i_], all_start_top_index[i_],                                                         all_end_top_log_probs[i_].reshape(beam_size,beam_size), all_end_top_index[i_].reshape(beam_size,beam_size),                                                         valid_start = offset, valid_end = real_length)
        selected_text = tokenizer.decode([w for w in x[best_start:best_end] if w != tokenizer.pad_token_id],clean_up_tokenization_spaces=False)
        selected_texts.append(selected_text if selected_text in test_df.loc[i_].text 
                              else fuzzy_match(test_df.loc[i_].text, selected_text)[-1])
    return selected_texts

def check_corrs(model_name):
    corrs = np.zeros((5,5))
    for i in range(5):
        for j in range(5):
            if corrs[j,i] == 0:
                corrs[i,j] = np.mean([jaccard(x,y) for x,y in zip(all_preds[f"{model_name}_{i}"], all_preds[f"{model_name}_{j}"])])
                corrs[j,i] = corrs[i,j]
    return corrs




all_preds = dict()

get_ipython().system('mkdir configs')
get_ipython().system('cp ../input/roberta-large-quest/roberta-large-vocab.json ./configs/vocab.json')
get_ipython().system('cp ../input/bart-large/config.json ./configs/config.json')
get_ipython().system('cp ../input/roberta-large-quest/roberta-large-merges.txt ./configs/merges.txt')

tokenizer = BartTokenizer.from_pretrained('./configs', do_lower_case=False)
config = BartConfig.from_pretrained('./configs',do_lower_case=False, output_hidden_states=True)
model = BartForSentimentExtraction(config)
model.cuda()
X_test, X_type_test = convert_lines(tokenizer, test_df, max_sequence_length= max_sequence_length)
dirs = ["bart-drop-head","bart-drop-head-2","tweet-all"]

for data_dir in dirs:
    for fn in os.listdir(f"../input/{data_dir}"):
        if "bart_" in fn:
            drop_head = data_dir != "tweet-all"
            print(fn,drop_head)
            model.load_state_dict(torch.load(f"../input/{data_dir}/{fn}"), strict=True)
            model.eval()
            all_start_top_log_probs,all_start_top_index,all_end_top_log_probs,all_end_top_index = get_predictions(X_test, X_type_test, model)
            selected_texts = get_selected_texts(X_test, tokenizer)
            print(selected_texts[:10])
            all_preds[f"{fn}"] = [x if type(x) is str else "" for x in selected_texts]
get_ipython().system('rm -rf configs')

get_ipython().system('mkdir configs')
get_ipython().system('cp ../input/roberta-base-quest/roberta-base-vocab.json ./configs/vocab.json')
get_ipython().system('cp ../input/roberta-base-quest/roberta-base-config.json ./configs/config.json')
get_ipython().system('cp ../input/roberta-base-quest/roberta-base-merges.txt ./configs/merges.txt')
tokenizer = RobertaTokenizer.from_pretrained(f'./configs/', do_lower_case=False)
config = RobertaConfig.from_pretrained('./configs',do_lower_case=False, output_hidden_states=True)
model = RobertaForSentimentExtraction(config)
model.cuda()
X_test, X_type_test = convert_lines(tokenizer, test_df, max_sequence_length= max_sequence_length)
dirs = ["roberta-base-drop-head","tweet-all"]
for data_dir in dirs:
    for fn in os.listdir(f"../input/{data_dir}"):
        if "roberta_" in fn:
            drop_head = data_dir != "tweet-all"
            print(fn,drop_head)
            model.load_state_dict(torch.load(f"../input/{data_dir}/{fn}"), strict=True)
            model.eval()
            all_start_top_log_probs,all_start_top_index,all_end_top_log_probs,all_end_top_index = get_predictions(X_test, X_type_test, model)
            selected_texts = get_selected_texts(X_test, tokenizer)
            print(selected_texts[:10])
            all_preds[f"{fn}"] = [x if type(x) is str else "" for x in selected_texts]
get_ipython().system('rm -rf configs')

get_ipython().system('mkdir configs')
get_ipython().system('cp ../input/roberta-large-quest/roberta-large-vocab.json ./configs/vocab.json')
get_ipython().system('cp ../input/roberta-large-quest/roberta-large-config.json ./configs/config.json')
get_ipython().system('cp ../input/roberta-large-quest/roberta-large-merges.txt ./configs/merges.txt')
tokenizer = RobertaTokenizer.from_pretrained(f'./configs/', do_lower_case=False)
config = RobertaConfig.from_pretrained('./configs',do_lower_case=False, output_hidden_states=True)
model = RobertaForSentimentExtraction(config)
model.cuda()
X_test, X_type_test = convert_lines(tokenizer, test_df, max_sequence_length= max_sequence_length)
dirs = ["roberta-large-drop-head","roberta-large-drop-head-2","tweet-all"]
for data_dir in dirs:
    for fn in os.listdir(f"../input/{data_dir}"):
        if "roberta-large_" in fn:
            drop_head = data_dir != "tweet-all"
            print(fn,drop_head)
            model.load_state_dict(torch.load(f"../input/{data_dir}/{fn}"), strict=True)
            model.eval()
            all_start_top_log_probs,all_start_top_index,all_end_top_log_probs,all_end_top_index = get_predictions(X_test, X_type_test, model)
            selected_texts = get_selected_texts(X_test, tokenizer)
            print(selected_texts[:10])
            all_preds[f"{fn}"] = [x if type(x) is str else "" for x in selected_texts]
get_ipython().system('rm -rf configs')




# model_list = [key for key in all_preds.keys()]
# print(model_list)
# all_vals = [all_preds[model_name] for model_name in model_list]  #+ dieter_preds
# print(len(all_vals))
# ensembled = []
# sep_texts = test_df.text.values
# # sep_texts = test_df.text.apply(lambda x: " ".join(x.split()).lower())
# for i in tqdm(range(len(test_df))):
#     predictions = [val[i] for val in all_vals]
#     slow_mode = False
#     ensembled.append(ensemble(sep_texts[i], predictions, slow_mode=slow_mode))
    
# sub_df = pd.read_csv("../input/valid-sub/submission.csv")
# print(np.mean([jaccard(x,y) for x,y in zip(sub_df.selected_text, ensembled)]))




import numpy as np 
import pandas as pd 
import torch
from torch import nn
from tqdm import tqdm
import os
import gc

def get_features(line, tokenizer, sentiment,  span=None):
    
    MAX_LEN = 114
    MAX_CHAR = 146
    pad_token_id = 1
    sep_token_id = 2
    cls_token_id = 0
    
    encoding = tokenizer.encode(line)
    offsets = np.array(encoding.offsets)
    token_lenghts = list(np.diff(offsets, axis =1)[:,0])
    
    #handle unnatural long chars
    while sum(token_lenghts)>(MAX_CHAR-4):
        token_lenghts = token_lenghts[:-1]
    
    rep_vec = [1,1,1] + token_lenghts + [1]
    sentiment_id = tokenizer.encode(sentiment).ids
    

    input_ids = [cls_token_id] + sentiment_id + [sep_token_id] + encoding.ids[:MAX_LEN-4] + [sep_token_id]
    attention_mask = [1] * len(input_ids)
    
    #perform padding
    attention_padding = [0] * (MAX_LEN - len(input_ids))
    padding = [pad_token_id] * (MAX_LEN - len(input_ids))
    rep_vec_padding = [0] * (MAX_LEN - len(rep_vec) - 1)
    rep_vec_padding += [MAX_CHAR - sum(rep_vec_padding) - sum(rep_vec)]

    
    input_ids.extend(padding)
    attention_mask.extend(attention_padding)
    rep_vec.extend(rep_vec_padding)
    
    assert len(rep_vec) == MAX_LEN
    assert sum(rep_vec) == MAX_CHAR
    assert len(input_ids) == MAX_LEN
    assert len(attention_mask) == MAX_LEN

    
    
    features = {'input_ids': np.array(input_ids,dtype=int),
                      'attention_mask': np.array(attention_mask,dtype=int),
                'rep_vec':np.array(rep_vec,dtype=int),
                      #'token_type_ids': np.array(token_type_ids,dtype=int),
                      #'token_idx2word_idx':np.array(token_idx2word_idx,dtype=int)
                      }
    
    if span is not None:
        features['target'] = np.array(span,dtype=int) + (3,3)

    else:
        features['target'] = np.array((-1,-1),dtype=int)
        
    return features

from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

class TEDataset(Dataset):
    
    def __init__(self,text,tokenizer,sentiments, labels=None, verbose=False):
        
        if labels is None:
            self.features = [get_features(text[i], tokenizer,sentiments[i], span=None) for i in tqdm(range(len(text)),disable= 1-verbose)]
        
        else:
            self.features = [get_features(text[i], tokenizer,sentiments[i], span=labels[i]) for i in tqdm(range(len(text)),disable= 1-verbose)]
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):

        return self.features[index]
    
def test_collate(batch):

    input_dict = {}
    target_dict = {}
    
    for key in ['input_ids','attention_mask','rep_vec']:#,'token_idx2word_idx','token_type_ids']:
        input_dict[key] = torch.from_numpy(np.stack([b[key] for b in batch])).long()
    for key in ['target']:#,'token_idx2word_idx','token_type_ids']:
        input_dict[key] = [None,None]
    
    for key in ['target']:
        target_dict[key] = torch.from_numpy(np.stack([b[key] for b in batch])).long()
    
    return input_dict, target_dict
import tokenizers

tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file='/kaggle/input/roberta-base-quest/roberta-base-vocab.json', 
            merges_file='/kaggle/input/roberta-base-quest/roberta-base-merges.txt', 
            lowercase=True,
            add_prefix_space=False)
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test['text'] = test['text'].astype(str)#.apply(split_join)
test['text'] = test['text'].apply(lambda x: x.replace('ï¿½','<i>'))
test_text = test['text'].values
test_sentiments = test['sentiment'].values
te_ds = TEDataset(test_text,tokenizer,test_sentiments)
te_dl = DataLoader(te_ds,sampler=SequentialSampler(te_ds), collate_fn=test_collate, batch_size=256)
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_roberta import RobertaModel
from transformers.configuration_bart import BartConfig
from transformers.modeling_bart import BartModel



class TERobertaBase(nn.Module):

    def __init__(self):
        super(TERobertaBase, self).__init__()

        config = RobertaConfig.from_json_file('/kaggle/input/roberta-base-quest/roberta-base-config.json')
        self.transformer = RobertaModel(config)
        self.dropout = torch.nn.Dropout(0)
        hdz = self.transformer.config.hidden_size
        
        self.rnn_start0 = nn.GRU(hdz, hdz, num_layers=4, batch_first=True, bidirectional=True)
        self.act_start0 = nn.ReLU()
        self.rnn_end0 = nn.GRU(hdz, hdz, num_layers=4, batch_first=True, bidirectional=True)
        self.act_end0 = nn.ReLU()
        
        self.head_start = nn.Sequential(nn.Linear(self.transformer.config.hidden_size,1))
        self.head_end = nn.Sequential(nn.Linear(self.transformer.config.hidden_size,1))
        self.beam_size = 1

    def forward(self, input_dict):
        beam_size = self.beam_size
        input_ids= input_dict['input_ids']
        #token_type_ids = input_dict['token_type_ids']

        attention_mask = input_dict['attention_mask']
        # targets = input_dict['target']
        rep_vec = input_dict['rep_vec']
        outputs = self.transformer(input_ids,
                            attention_mask=attention_mask,
                            #token_type_ids=token_type_ids
                                   )

        hidden_states = outputs[0]
        
        bs = hidden_states.shape[0]
        hidden_states = torch.stack([torch.repeat_interleave(hidden_states[k],rep_vec[k], axis = 0) for k in range(bs)])
        hs_start, _ = self.rnn_start0(hidden_states)
        hs_start = hs_start.reshape(hs_start.shape[0],hs_start.shape[1],hs_start.shape[2]//2,2).sum(-1)
        hs_start = self.act_start0(hs_start)
        
        hs_end, _ = self.rnn_end0(hidden_states)
        hs_end = hs_end.reshape(hs_end.shape[0],hs_end.shape[1],hs_end.shape[2]//2,2).sum(-1)
        hs_end = self.act_end0(hs_end)
        

        start_logits = self.head_start(hs_start).squeeze(-1)
        end_logits = self.head_start(hs_end).squeeze(-1)

        output_dict = {'start_logits':start_logits,
                   'end_logits':end_logits
                   }
     
        return output_dict

    
class TERobertaLarge(nn.Module):

    def __init__(self):
        super(TERobertaLarge, self).__init__()

        config = RobertaConfig.from_json_file('/kaggle/input/roberta-large-quest/roberta-large-config.json')
        self.transformer = RobertaModel(config)
        self.dropout = torch.nn.Dropout(0)
        hdz = self.transformer.config.hidden_size
        
        self.rnn_start0 = nn.GRU(hdz, hdz, num_layers=2, batch_first=True, bidirectional=True)
        self.act_start0 = nn.ReLU()
        self.rnn_end0 = nn.GRU(hdz, hdz, num_layers=2, batch_first=True, bidirectional=True)
        self.act_end0 = nn.ReLU()
        
        self.head_start = nn.Sequential(nn.Linear(self.transformer.config.hidden_size,1))
        self.head_end = nn.Sequential(nn.Linear(self.transformer.config.hidden_size,1))
        self.beam_size = 1

    def forward(self, input_dict):
        beam_size = self.beam_size
        input_ids= input_dict['input_ids']
        #token_type_ids = input_dict['token_type_ids']

        attention_mask = input_dict['attention_mask']
        #targets = input_dict['target']
        rep_vec = input_dict['rep_vec']
        outputs = self.transformer(input_ids,
                            attention_mask=attention_mask,
                            #token_type_ids=token_type_ids
                                   )

        hidden_states = outputs[0]
        
        bs = hidden_states.shape[0]
        hidden_states = torch.stack([torch.repeat_interleave(hidden_states[k],rep_vec[k], axis = 0) for k in range(bs)])
        hs_start, _ = self.rnn_start0(hidden_states)
        hs_start = hs_start.reshape(hs_start.shape[0],hs_start.shape[1],hs_start.shape[2]//2,2).mean(-1)
        hs_start = self.act_start0(hs_start)
        
        hs_end, _ = self.rnn_end0(hidden_states)
        hs_end = hs_end.reshape(hs_end.shape[0],hs_end.shape[1],hs_end.shape[2]//2,2).mean(-1)
        hs_end = self.act_end0(hs_end)
        
        start_logits = self.head_start(hs_start).squeeze(-1)
        end_logits = self.head_start(hs_end).squeeze(-1)

        output_dict = {'start_logits':start_logits,
                   'end_logits':end_logits
                   }

        return output_dict
    
    
class TEBartLarge(nn.Module):

    def __init__(self):
        super(TEBartLarge, self).__init__()

        config = BartConfig.from_json_file('/kaggle/input/bart-large/config.json')
        self.transformer = BartModel(config)
        self.dropout = torch.nn.Dropout(0)
        hdz = self.transformer.config.hidden_size
        
        self.rnn_start0 = nn.GRU(hdz, hdz, num_layers=3, batch_first=True, bidirectional=True)
        self.act_start0 = nn.ReLU()
        self.rnn_end0 = nn.GRU(hdz, hdz, num_layers=3, batch_first=True, bidirectional=True)
        self.act_end0 = nn.ReLU()
        
        self.head_start = nn.Sequential(nn.Linear(self.transformer.config.hidden_size,1))
        self.head_end = nn.Sequential(nn.Linear(self.transformer.config.hidden_size,1))

        self.beam_size = 1

    def forward(self, input_dict):
        beam_size = self.beam_size
        input_ids= input_dict['input_ids']
        #token_type_ids = input_dict['token_type_ids']

        attention_mask = input_dict['attention_mask']
        #targets = input_dict['target']
        rep_vec = input_dict['rep_vec']
        outputs = self.transformer(input_ids,
                            attention_mask=attention_mask,
                            #token_type_ids=token_type_ids
                                   )

        hidden_states = outputs[0]
        
        bs = hidden_states.shape[0]
        hidden_states = torch.stack([torch.repeat_interleave(hidden_states[k],rep_vec[k], axis = 0) for k in range(bs)])
        hs_start, _ = self.rnn_start0(hidden_states)
        hs_start = hs_start.reshape(hs_start.shape[0],hs_start.shape[1],hs_start.shape[2]//2,2).sum(-1)
        hs_start = self.act_start0(hs_start)
        
        hs_end, _ = self.rnn_end0(hidden_states)
        hs_end = hs_end.reshape(hs_end.shape[0],hs_end.shape[1],hs_end.shape[2]//2,2).sum(-1)
        hs_end = self.act_end0(hs_end)
        

        start_logits = self.head_start(hs_start).squeeze(-1)

        end_logits = self.head_start(hs_end).squeeze(-1)

        output_dict = {'start_logits':start_logits,
                   'end_logits':end_logits
                   }
     
            

        return output_dict
def dict_unravel(batch):
        input_dict, label_dict = batch
        input_dict2 = {k: input_dict[k].cuda() for k in input_dict if not k == 'target'}
#         input_dict2['target'] = input_dict['target']
        label_dict = {k: label_dict[k].cuda() for k in label_dict}

        return input_dict2, label_dict


def get_logits(dl, model):
    with torch.no_grad():

        all_start_logits = []
        all_end_logits = []
        for batch in tqdm(dl):
            input_dict, target_dict = dict_unravel(batch)
            outs = model.forward(input_dict)
            start_logits = outs['start_logits']
            end_logits = outs['end_logits']
            all_start_logits += [start_logits.detach().cpu().numpy()]
            all_end_logits += [end_logits.detach().cpu().numpy()]
           
        all_start_logits = np.concatenate(all_start_logits)
        all_end_logits = np.concatenate(all_end_logits)

    return all_start_logits, all_end_logits
weights_path = '/kaggle/input/roberta-base-38-full-fp16-weights/'
model = TERobertaBase().cuda().eval()

test_start_logits = []
test_end_logits = []
for FOLD in [0,1,2,3,4]:
    model.load_state_dict(torch.load(weights_path + f'fold{FOLD}.pt'))
    test_start_logits_, test_end_logits_ = get_logits(te_dl, model)
    test_start_logits += [test_start_logits_]
    test_end_logits += [test_end_logits_]

weights_path = '/kaggle/input/roberta-base-38-pseudo-full-fp16-weights/'
for FOLD in [0,1,2,3,4]:
    model.load_state_dict(torch.load(weights_path + f'fold{FOLD}.pt'))
    test_start_logits_, test_end_logits_ = get_logits(te_dl, model)
    test_start_logits += [test_start_logits_]
    test_end_logits += [test_end_logits_]
    
del model
gc.collect()

weights_path = '/kaggle/input/roberta-large-3-full-fp16-weights/'
model = TERobertaLarge().cuda().eval()

for FOLD in [0,1,2,3,4]:
    model.load_state_dict(torch.load(weights_path + f'fold{FOLD}.pt'))
    test_start_logits_, test_end_logits_ = get_logits(te_dl, model)
    test_start_logits += [test_start_logits_]
    test_end_logits += [test_end_logits_]
    
weights_path = '/kaggle/input/roberta-large-3-full-pseudo-weights/'
for FOLD in [0,1,2,3,4]:
    model.load_state_dict(torch.load(weights_path + f'fold{FOLD}.pt'))
    test_start_logits_, test_end_logits_ = get_logits(te_dl, model)
    test_start_logits += [test_start_logits_]
    test_end_logits += [test_end_logits_]

del model
gc.collect()

weights_path = '/kaggle/input/bart-large-2-full-fp16-weights/'
model = TEBartLarge().cuda().eval()

for FOLD in [0,1,2,3,4]:
    model.load_state_dict(torch.load(weights_path + f'fold{FOLD}.pt'))
    test_start_logits_, test_end_logits_ = get_logits(te_dl, model)
    test_start_logits += [test_start_logits_]
    test_end_logits += [test_end_logits_]
    
weights_path = '/kaggle/input/bart-large-2-full-pseudo-weights/'
for FOLD in [0,1,2,3,4]:
    model.load_state_dict(torch.load(weights_path + f'fold{FOLD}.pt'))
    test_start_logits_, test_end_logits_ = get_logits(te_dl, model)
    test_start_logits += [test_start_logits_]
    test_end_logits += [test_end_logits_]
    
del model
gc.collect()

def find_best_combinations(start_logits,end_logits, valid_start= 3, valid_end=512):
    
    best = (valid_start, valid_end-1)
    best_score = -9999
    best_ = np.argmax(start_logits), np.argmax(end_logits)
    if (valid_start <= best_[0] < valid_end) and (valid_start <= best_[1] < valid_end) and (best_[0] < best_[1]):
        return best_
    for i in range(valid_start, valid_end):
        for j in range(i + 1, valid_end):
            score = start_logits[i] + end_logits[j]
            if score > best_score:
                best = (i,j)
                best_score = score
    return best

def get_preds(all_start_logits, all_end_logits, valid_start= 3, valid_end=512 ):
    
    all_start_logits = all_start_logits
    
    start_preds = []
    end_preds = []
    for start_logits,end_logits in zip(all_start_logits, all_end_logits):
        start, end = find_best_combinations(start_logits,end_logits, valid_start, valid_end)
        start_preds += [start]
        end_preds += [end]
    
    start_preds = np.array(start_preds)
    end_preds = np.array(end_preds)
    return start_preds, end_preds

start_logits = np.mean(test_start_logits, axis = 0)
end_logits = np.mean(test_end_logits, axis = 0)
test_start_preds, test_end_preds = get_preds(start_logits, end_logits, valid_start= 3, valid_end=146 )
test_pred_text = []
for i in range(len(test_text)):
    test_pred_text += [test_text[i][test_start_preds[i]-3:test_end_preds[i]-3]]




all_vals2 = []
for i in tqdm(range(len(test_start_logits))):
    test_start_preds, test_end_preds = get_preds(test_start_logits[i], test_end_logits[i], valid_start= 3, valid_end=146 )
    curr_val = []
    for i in range(len(test_text)):
        curr_val += [test_text[i][test_start_preds[i]-3:test_end_preds[i]-3]]
    all_vals2.append(curr_val)




all_vals = [val for val in all_preds.values()] + all_vals2
print(len(all_vals))
ensembled = []
sep_texts = test_df.text.values
for i in tqdm(range(len(test_df))):
    if "  " in sep_texts[i]:
        ensembled.append(test_pred_text[i])
    else:
        predictions = [val[i] for val in all_vals]
        ensembled.append(ensemble(sep_texts[i], predictions, slow_mode=True))
#     predictions = [val[i] for val in all_vals]
#     ensembled.append(ensemble(sep_texts[i], predictions, slow_mode=True))




if len(all_vals) == 60:
    test_df["selected_text"] = ensembled
    test_df[["textID","selected_text"]].to_csv("submission.csv",index=False)




get_ipython().system('head -n10 submission.csv')






