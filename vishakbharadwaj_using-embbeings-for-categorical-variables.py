import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from datetime import datetime
train = pd.read_csv('../input/shelter-animal-outcomes/train.csv.gz')
print("Shape:", train.shape)
train.head()
test = pd.read_csv('../input/shelter-animal-outcomes/test.csv.gz')
print("Shape:", test.shape)
test.head()
sample = pd.read_csv('../input/shelter-animal-outcomes/sample_submission.csv.gz')
sample.head()
Counter(train['OutcomeType'])
Counter(train['Name']).most_common(5)
train_X = train.drop(columns= ['OutcomeType', 'OutcomeSubtype', 'AnimalID'])
Y = train['OutcomeType']
test_X = test
stacked_df = train_X.append(test_X.drop(columns=['ID']))
# stacked_df['DateTime'] = pd.to_datetime(stacked_df['DateTime'])
# stacked_df['year'] = stacked_df['DateTime'].dt.year
# stacked_df['month'] = stacked_df['DateTime'].dt.month
stacked_df = stacked_df.drop(columns=['DateTime'])
stacked_df.head()
for col in stacked_df.columns:
    if stacked_df[col].isnull().sum() > 10000:
        print("dropping", col, stacked_df[col].isnull().sum())
        stacked_df = stacked_df.drop(columns = [col])
stacked_df.head()
for col in stacked_df.columns:
    if stacked_df.dtypes[col] == "object":
        stacked_df[col] = stacked_df[col].fillna("NA")
    else:
        stacked_df[col] = stacked_df[col].fillna(0)
    stacked_df[col] = LabelEncoder().fit_transform(stacked_df[col])
stacked_df.head()
# making all variables categorical
for col in stacked_df.columns:
    stacked_df[col] = stacked_df[col].astype('category')
X = stacked_df[0:26729]
test_processed = stacked_df[26729:]

#check if shape[0] matches original
print("train shape: ", X.shape, "orignal: ", train.shape)
print("test shape: ", test_processed.shape, "original: ", test.shape)
Y = LabelEncoder().fit_transform(Y)

#sanity check to see numbers match and matching with previous counter to create target dictionary
print(Counter(train['OutcomeType']))
print(Counter(Y))
target_dict = {
    'Return_to_owner' : 3,
    'Euthanasia': 2,
    'Adoption': 0,
    'Transfer': 4,
    'Died': 1
}
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.10, random_state=0)
X_train.head()
#categorical embedding for columns having more than two values
embedded_cols = {n: len(col.cat.categories) for n,col in X.items() if len(col.cat.categories) > 2}
embedded_cols
embedded_col_names = embedded_cols.keys()
len(X.columns) - len(embedded_cols) #number of numerical columns
embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]
embedding_sizes
class ShelterOutcomeDataset(Dataset):
    def __init__(self, X, Y, embedded_col_names):
        X = X.copy()
        self.X1 = X.loc[:,embedded_col_names].copy().values.astype(np.int64) #categorical columns
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32) #numerical columns
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]
#creating train and valid datasets
train_ds = ShelterOutcomeDataset(X_train, y_train, embedded_col_names)
valid_ds = ShelterOutcomeDataset(X_val, y_val, embedded_col_names)
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
device = get_default_device()
device
class ShelterOutcomeModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories,size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings) #length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 5)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)
        

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x
model = ShelterOutcomeModel(embedding_sizes, 1)
to_device(model, device)
def get_optimizer(model, lr = 0.001, wd = 0.0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch_optim.Adam(parameters, lr=lr, weight_decay=wd)
    return optim
def train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0]
        output = model(x1, x2)
        loss = F.cross_entropy(output, y)   
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch*(loss.item())
    return sum_loss/total
def val_loss(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = F.cross_entropy(out, y)
        sum_loss += current_batch_size*(loss.item())
        total += current_batch_size
        pred = torch.max(out, 1)[1]
        correct += (pred == y).float().sum().item()
    print("valid loss %.3f and accuracy %.3f" % (sum_loss/total, correct/total))
    return sum_loss/total, correct/total
def train_loop(model, epochs, lr=0.01, wd=0.0):
    optim = get_optimizer(model, lr = lr, wd = wd)
    for i in range(epochs): 
        loss = train_model(model, optim, train_dl)
        print("training loss: ", loss)
        val_loss(model, valid_dl)
batch_size = 1000
train_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size,shuffle=True)
train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)
train_loop(model, epochs=8, lr=0.05, wd=0.00001)