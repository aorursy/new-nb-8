from fastai.conv_learner import *
from fastai.dataset import *
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import math
MODEL_PATH = 'Resnet18_v1'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
SAMPLE_SUB = '../input/sample_submission.csv'

arch = resnet34
nw = 4
df = pd.read_csv(LABELS).set_index('Image')
new_whale_df = df[df.Id == "new_whale"] # only new_whale dataset
train_df = df[~(df.Id == "new_whale")] # no new_whale dataset, used for training
unique_labels = np.unique(train_df.Id.values)

labels_dict = dict()
labels_list = []
for i in range(len(unique_labels)):
    labels_dict[unique_labels[i]] = i
    labels_list.append(unique_labels[i])
print("Number of classes: {}".format(len(unique_labels)))
train_names = train_df.index.values
train_df.Id = train_df.Id.apply(lambda x: labels_dict[x])
train_labels = np.asarray(train_df.Id.values)
test_names = [f for f in os.listdir(TEST)]
labels_count = train_df.Id.value_counts()
_, _,_ = plt.hist(labels_count,bins=100)
labels_count
dup = []
for idx,row in train_df.iterrows():
    if labels_count[row['Id']] < 5:
        dup.extend([idx]*math.ceil((5 - labels_count[row['Id']])/labels_count[row['Id']]))
train_names = np.concatenate([train_names, dup])
train_names = train_names[np.random.RandomState(seed=42).permutation(train_names.shape[0])]
len(train_names)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42069)
for train_idx, val_idx in sss.split(train_names, np.zeros(train_names.shape)):
    tr_n, val_n = train_names[train_idx], train_names[val_idx]
print(len(tr_n), len(val_n))
avg_width = 0
avg_height = 0
for fn in os.listdir(TRAIN)[:1000]:
    img = cv2.imread(os.path.join(TRAIN,fn))
    avg_width += img.shape[1]
    avg_height += img.shape[0]
avg_width //= 1000
avg_height //= 1000
print(avg_width, avg_height)
class HWIDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.train_df = train_df
        super().__init__(fnames, transform, path)

    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        # We crop the center of the original image for faster training time
        img = cv2.resize(img, (self.sz, self.sz))
        return img

    def get_y(self, i):
        if (self.path == TEST): return 0
        return self.train_df.loc[self.fnames[i]]['Id']


    def get_c(self):
        return len(unique_labels)

class RandomLighting(Transform):
    def __init__(self, b, c, tfm_y=TfmType.NO):
        super().__init__(tfm_y)
        self.b, self.c = b, c

    def set_state(self):
        self.store.b_rand = rand0(self.b)
        self.store.c_rand = rand0(self.c)

    def do_transform(self, x, is_y):
        if is_y and self.tfm_y != TfmType.PIXEL: return x  # add this line to fix the bug
        b = self.store.b_rand
        c = self.store.c_rand
        c = -1 / (c - 1) if c < 0 else c + 1
        x = lighting(x, b, c)
        return x
    
def get_data(sz, bs, test_names=test_names, test_dir=TEST):
    aug_tfms = [RandomRotateZoom(deg=20, zoom=2, stretch=1),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO),
                RandomBlur(blur_strengths=3,tfm_y=TfmType.NO),
                RandomFlip(tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO,
                           aug_tfms=aug_tfms)
    ds = ImageData.get_ds(HWIDataset, (tr_n[:-(len(tr_n) % bs)], TRAIN),
                          (val_n, TRAIN), tfms, test=(test_names, test_dir))
    md = ImageData("./", ds, bs, num_workers=nw, classes=None)
    return md

# sz = (avg_width//2, avg_height//2)
batch_size = 64
md = get_data(384, batch_size)
learn = ConvLearner.pretrained(arch, md) 
learn.opt_fn = optim.Adam
# learn.lr_find()
# learn.sched.plot()
lr = 5e-3
learn.fit(lr, 2, cycle_len=3)
learn.unfreeze()
lrs = np.array([lr/10, lr/20, lr/40])
learn.fit(lrs, 4, cycle_len=4, use_clr=(20, 16))
learn.save(MODEL_PATH)
preds_v,y_v = learn.TTA(is_test=False,n_aug=2)
preds_v = np.stack(preds_v, axis=-1)
preds_v = np.exp(preds_v)
preds_v = preds_v.mean(axis=-1)
y_v += 1
preds_v_max = np.max(preds_v,axis=1)
_,_,_ = plt.hist(preds_v_max)
TEST=TRAIN # sorry
total_new_whale = len(new_whale_df.index.values)
md = get_data(384, batch_size, test_names=new_whale_df.index.values[:int(total_new_whale*0.2)], test_dir=TRAIN)
learn.set_data(md)
preds_w,y_w = learn.TTA(is_test=True,n_aug=2)
preds_w = np.stack(preds_w, axis=-1)
preds_w = np.exp(preds_w)
preds_w = preds_w.mean(axis=-1)
preds_w_max = np.max(preds_w,axis=1)
_,_,_ = plt.hist(preds_w_max)
y = np.concatenate([y_v,y_w])
preds = np.concatenate([preds_v, preds_w],axis=0)
preds = np.concatenate([np.zeros((preds.shape[0],1)), preds],axis=1)
def map5(X, y):
    score = 0
    for i in range(X.shape[0]):
        pred = X[i].argsort()[-5:][::-1]
        for j in range(pred.shape[0]):
            if pred[j] == y[i]:
                score += (5 - j)/5
                break
    return score/X.shape[0]

best_th = 0
best_score = 0
for th in np.arange(0.1, 0.801, 0.01):
    preds[:,0] = th
    score = map5(preds, y)
    if score > best_score:
        best_score = score
        best_th = th
    print("Threshold = {:.3f}, MAP5 = {:.3f}".format(th,score))

TEST = '../input/test/'
md = get_data(384, batch_size, test_names=test_names, test_dir=TEST)
learn.set_data(md)
preds_t,y_t = learn.TTA(is_test=True,n_aug=8)
preds_t = np.stack(preds_t, axis=-1)
preds_t = np.exp(preds_t)
preds_t = preds_t.mean(axis=-1)
preds_t = np.concatenate([np.zeros((preds_t.shape[0],1))+best_th, preds_t],axis=1)
sample_df = pd.read_csv(SAMPLE_SUB)
sample_list = list(sample_df.Image)
labels_list = ["new_whale"]+labels_list
pred_list = [[labels_list[i] for i in p.argsort()[-5:][::-1]] for p in preds_t]
pred_dic = dict((key, value) for (key, value) in zip(learn.data.test_ds.fnames,pred_list))
pred_list_cor = [' '.join(pred_dic[id]) for id in sample_list]
df = pd.DataFrame({'Image':sample_list,'Id': pred_list_cor})
df.to_csv('submission.csv'.format(MODEL_PATH), header=True, index=False)
df.head()