
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import random

from sklearn.metrics import roc_auc_score



import torch

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

import transformers

from transformers import BertForSequenceClassification, BertPreTrainedModel, BertConfig, BertModel

import torch.nn as nn

import torch.nn.functional as F



import torch_xla

import torch_xla.core.xla_model as xm



import os



bert = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if ("seqlen128" in filename):

            bert.append(filename)

        print(os.path.join(dirname, filename))
class config:

    EPOCHS = 1

    BATCH_SIZE = 32

    VAL_BATCH_SIZE = 128

    TEST_BATCH_SIZE = 128

    LR = 3e-5
valid = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation-processed-seqlen128.csv")

train = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train-processed-seqlen128.csv")

train_bias = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train-processed-seqlen128.csv")

test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test-processed-seqlen128.csv")

submit = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")

evalid = train[['id', 'comment_text', 'input_word_ids', 'input_mask','all_segment_id', 'toxic']].iloc[20001:25001]

train = train[['id', 'comment_text', 'input_word_ids', 'input_mask','all_segment_id', 'toxic']].iloc[:20000]

train_bias = train_bias[['id', 'comment_text', 'input_word_ids', 'input_mask','all_segment_id', 'toxic']].iloc[:20000]
evalid = evalid.reset_index()
# train = pd.concat([train[train["toxic"] == 1], train[train["toxic"] == 0].sample(n=21384, random_state=29)]).sample(frac=1).reset_index(drop=True)
train.info()
train_distribution = train["toxic"].value_counts().values

valid_distribution = valid["toxic"].value_counts().values



non_toxic = [train_distribution[0] / sum(train_distribution) * 100, valid_distribution[0] / sum(valid_distribution) * 100]

toxic = [train_distribution[1] / sum(train_distribution) * 100, valid_distribution[1] / sum(valid_distribution) * 100]



plt.figure(figsize=(9,6))

plt.bar([0, 1], non_toxic, alpha=.4, color="r", width=0.35, label="non-toxic")

plt.bar([0.4, 1.4], toxic, alpha=.4, width=0.35, label="toxic")

plt.xlabel("Dataset")

plt.ylabel("Percentage")

plt.xticks([0.2, 1.2], ["train", "valid"])

plt.legend(loc="upper right")



plt.show()
print(f"train: \nnon-toxic rate: {train_distribution[0] / sum(train_distribution) * 100: .2f} %\ntoxic rate: {train_distribution[1] / sum(train_distribution) * 100: .2f} %")

print(f"valid: \nnon-toxic rate: {valid_distribution[0] / sum(valid_distribution) * 100: .2f} %\ntoxic rate: {valid_distribution[1] / sum(valid_distribution) * 100: .2f} %")
lang = valid["lang"].value_counts()



plt.figure(figsize=(9, 6))

plt.xlabel("Lang")

plt.ylabel("Num")

plt.xticks([0.2, 0.6, 1], ["tr", "es", "it"])

plt.bar([0.2, 0.6, 1], lang, color="purple", width=0.28, alpha=.4)

plt.show()
class TweetDataset(Dataset):

    def __init__(self, mode, df):

        self.mode = mode

        self.df = df

    

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, idx):

        token, segment, mask = self.df.loc[idx, ["input_word_ids", "all_segment_id", "input_mask"]].values

        if self.mode=="train" or self.mode == "valid":

            label_tensor = torch.tensor(self.df.loc[idx, "toxic"])

        else:

            label_tensor = torch.tensor(-1)

        tokens_tensor = torch.tensor([int(i) for i in token[1:-1].split(",")])

        segments_tensor = torch.tensor([int(i) for i in segment[1:-1].split(",")])

        masks_tensor = torch.tensor([int(i) for i in mask[1:-1].split(",")])

           

        return tokens_tensor, segments_tensor, masks_tensor, label_tensor
lang = {'Spanish': 'es', 'Italian': 'it', 'Turkish': 'tr'}



validsets = {}

for i, k in lang.items():

    validsets[i] = TweetDataset("valid", valid[valid["lang"] == k].reset_index(drop=True))

trainset = TweetDataset("train", train)

trainbias = TweetDataset("train", train_bias)

validset = TweetDataset("valid", valid)

testset = TweetDataset("test", test)

evalidset = TweetDataset("valid", evalid)



validloaders = {}

for i, k in validsets.items():

    validloaders[i] = DataLoader(k, batch_size=config.VAL_BATCH_SIZE, num_workers=4, shuffle=False)

trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False)

trainbiasloader = DataLoader(trainbias, batch_size=config.BATCH_SIZE, num_workers=4, shuffle=False)

validloader = DataLoader(validset, batch_size=config.VAL_BATCH_SIZE, num_workers=4, shuffle=False)

testloader = DataLoader(testset, batch_size=config.TEST_BATCH_SIZE, num_workers=4, shuffle=False)

evalidloader = DataLoader(evalidset, batch_size=config.VAL_BATCH_SIZE, num_workers=4, shuffle=False)
class Model(nn.Module):

    

    def __init__(self, labels=1):

        

        super().__init__()

        

        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")

        self.num_features = self.bert.pooler.dense.out_features

        self.labels = labels

        

        self.drop = nn.Dropout(0.3)

        self.fc1 = nn.Linear(self.num_features * 2, self.num_features)

        self.logit = nn.Linear(self.num_features, self.labels)

        

    def forward(self, tokens_tensors, segments_tensors, masks_tensors):



        hidden_states, cls = self.bert(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)

        avgpool = torch.mean(hidden_states, 1)

        maxpool, _ = torch.max(hidden_states, 1)

        cat = torch.cat((avgpool, maxpool), 1)

        x = self.drop(cat)

        x = torch.tanh(self.fc1(x))

        output = self.logit(x)



        return output
model = Model()
model
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = xm.xla_device()

model.to(device)

print(f"Now we use {device}\n")
def training(model, warmup_prop=0.1):



    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    num_warmup_steps = int(warmup_prop * config.EPOCHS * len(trainloader))

    num_training_steps = config.EPOCHS * len(trainloader)

    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    loss_fun = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)    



    for epoch in range(config.EPOCHS):

        model.train()

        

        optimizer.zero_grad()

        avg_loss = 0

        

        for data in tqdm(trainloader):             

            tokens_tensor, segments_tensor, masks_tensor, labels_tensor = [k.to(device) for k in data if k is not None]

            output = model(tokens_tensor, segments_tensor, masks_tensor)

            loss = loss_fun(output.view(-1).float(), labels_tensor.float().to(device))

            loss.backward()

            avg_loss += loss.item() / len(trainloader)



            xm.optimizer_step(optimizer, barrier=True)

            scheduler.step()

            model.zero_grad()

            optimizer.zero_grad()

                

        model.eval()

        preds = []

        truths = []

        avg_val_loss = 0.



        with torch.no_grad():

            for data in validloader:

                tokens_tensor, segments_tensor, masks_tensor, labels_tensor = [k.to(device) for k in data if k is not None]

                output = model(tokens_tensor, segments_tensor, masks_tensor)

                loss = loss_fun(output.detach().view(-1).float(), labels_tensor.float().to(device))

                avg_val_loss += loss.item() / len(validloader)

                

                probs = torch.sigmoid(output).detach().cpu().numpy()

                preds += list(probs.flatten())

                truths += list(labels_tensor.detach().cpu().numpy().flatten())

            score = roc_auc_score(truths, preds)

        

        lr = scheduler.get_last_lr()[0]

        print(f'[Epoch {epoch + 1}] lr={lr:.1e} loss={avg_loss:.4f} val_loss={avg_val_loss:.4f} val_auc={score:.4f}')
threshold = lambda x: 1 if x>=0.5 else 0



def predict(model, dataloader, df, isAccuracy=True):

 

    model.eval().to(device)

    preds = np.empty((0, 1))

    accuracy = None



    with torch.no_grad():

        for data in tqdm(dataloader):

            tokens_tensor, segments_tensor, masks_tensor, labels_tensor = [k.to(device) for k in data if k is not None]

            probs = torch.sigmoid(model(tokens_tensor, segments_tensor, masks_tensor)).detach().cpu().numpy()

            preds = np.concatenate([preds, probs])

            

    preds = preds.reshape(len(preds))        

    predicts = np.array([threshold(i) for i in preds])

    if isAccuracy:

        accuracy = (df["toxic"].values == predicts).sum() / len(df)



    return preds, predicts, accuracy 
# before training model accuracy

pre, pre_class, accuracy = predict(model, trainloader, train)

auc = roc_auc_score(train["toxic"].values, pre_class)

print("Train: ")

print(f"Model before fine-tune accuracy: {accuracy * 100:.3f}%\nModel before fine-tune AUC: {auc:.3f}")



for key, value in validloaders.items():

    pre, pre_class, accuracy = predict(model, value, valid[valid["lang"] == lang[key]].reset_index(drop=True))

    auc = roc_auc_score(valid[valid["lang"] == lang[key]].reset_index(drop=True)["toxic"].values, pre_class)

    print(f"{key} Valid: ")

    print(f"Model before fine-tune accuracy: {accuracy * 100:.2f}%\nModel before fine-tune AUC: {auc:.3f}")



pre, pre_class, accuracy = predict(model, validloader, valid)

auc = roc_auc_score(valid["toxic"].values, pre_class)

print(f"Combined Valid: ")

print(f"Model before fine-tune accuracy: {accuracy * 100:.2f}%\nModel before fine-tune AUC: {auc:.3f}")



training(model)
train_bias["toxic"] = train_bias["toxic"].apply(lambda x: (1 if x >= .5 else 0))
# After training model accuracy

pre, pre_class, accuracy = predict(model, trainloader, train)

auc = roc_auc_score(train["toxic"].values, pre)

print("Train: ")

print(f"Model after fine-tune accuracy: {accuracy * 100:.3f}%\nModel after fine-tune AUC: {auc:.3f}")

  

for key, value in validloaders.items():

    pre, pre_class, accuracy = predict(model, value, valid[valid["lang"] == lang[key]].reset_index(drop=True))

    auc = roc_auc_score(valid[valid["lang"] == lang[key]].reset_index(drop=True)["toxic"].values, pre)

    print(f"{key} Valid: ")

    print(f"Model after fine-tune accuracy: {accuracy * 100:.2f}%\nModel after fine-tune AUC: {auc:.3f}")



pre, pre_class, accuracy = predict(model, validloader, valid)

auc = roc_auc_score(valid["toxic"].values, pre)

print(f"Combined Valid: ")

print(f"Model after fine-tune accuracy: {accuracy * 100:.2f}%\nModel after fine-tune AUC: {auc:.3f}")
pre, pre_class, accuracy = predict(model, evalidloader, evalid)

auc = roc_auc_score(evalid["toxic"].values, pre)

print("Evalid: ")

print(f"Model after fine-tune accuracy: {accuracy * 100:.3f}%\nModel after fine-tune AUC: {auc:.3f}")
pre, pre_class, accuracy = predict(model, testloader, test, False)

submit['toxic'] = pre

submit.to_csv('submission.csv', index=False)

submit.head()
import torch



a = torch.tensor([[[1, 2, 3], [1, 4, 5]], [[4, 5, 6], [7, 4, 5]], [[4, 5, 6], [7, 4, 5]]]).type(torch.float32)

torch.max(a, 1)