import os

import numpy as np

import pandas as pd

import torch

import cv2

import sys

import collections

import segmentation_models_pytorch as smp

import albumentations as albu



from torch.utils.data import DataLoader

from glob import glob

from os import path

from PIL import Image

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt



IMG_WIDTH = 256

IMG_HEIGHT = 256

NUM_CLASSES = 46



BATCH_SIZE = 32

N_WORKERS = 2



root_path_train = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train'

df_path_train = '/kaggle/input/imaterialist-fashion-2019-FGVC6/train.csv'
def create_one_represent_class(df_param):

    v_c_df = df_param['CategoryId'].value_counts().reset_index()

    one_represent = v_c_df.loc[v_c_df['CategoryId'] == 1, 'index'].tolist()

    df_param.loc[df_param['CategoryId'].isin(one_represent), 'CategoryId'] = 'one_represent'

    return df_param



def custom_train_test_split(df_param):

    

    df_param['CategoryId'] = df_param.ClassId.apply(lambda x: str(x).split("_")[0])

    

    img_categ = train_df.groupby('ImageId')['CategoryId'].apply(list).reset_index()

    img_categ['CategoryId'] = img_categ['CategoryId'].apply(lambda x: ' '.join(sorted(x)))

    

    img_categ = create_one_represent_class(img_categ)

    

    img_train, img_val  = train_test_split(img_categ, test_size=0.2, random_state=42, stratify=img_categ['CategoryId'])

    

    df_param = df_param.drop(columns='CategoryId')

    

    df_train = df_param[df_param['ImageId'].isin(img_train['ImageId'])].reset_index(drop=True)

    df_val = df_param[df_param['ImageId'].isin(img_val['ImageId'])].reset_index(drop=True)

    

    return df_train, df_val
train_df = pd.read_csv(df_path_train)

train_df.head()
train_df, val_df = custom_train_test_split(train_df)
def rle_decode(mask_rle, shape):

    '''

    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array

    shape: (height,width) of array to return

    Returns numpy array according to the shape, 1 - mask, 0 - background

    '''

    shape = (shape[1], shape[0])

    s = mask_rle.split()

    # gets starts & lengths 1d arrays

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]

    starts -= 1

    # gets ends 1d array

    ends = starts + lengths

    # creates blank mask image 1d array

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    # sets mark pixles

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    # reshape as a 2d mask image

    return img.reshape(shape).T  # Needed to align to RLE direction



    

class UnetDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df, height, width, augmentation=None, preprocessing=None):

        

        self.preprocessing = preprocessing

        self.augmentation = augmentation

        

        self.image_dir = image_dir

        self.df = df

        

        self.height = height

        self.width = width

        

        self.image_info = collections.defaultdict(dict)

        

        self.df['CategoryId'] = self.df.ClassId.apply(lambda x: str(x).split("_")[0])

        self.num_classes = self.df['CategoryId'].nunique()

        

        temp_df = self.df.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()

        size_df = self.df.groupby('ImageId')['Height', 'Width'].mean().reset_index()

        temp_df = temp_df.merge(size_df, on='ImageId', how='left')

        

        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):

            image_id = row['ImageId']

            image_path = os.path.join(self.image_dir, image_id)

            self.image_info[index]["image_id"] = image_id

            self.image_info[index]["image_path"] = image_path

            self.image_info[index]["width"] = self.width

            self.image_info[index]["height"] = self.height

            self.image_info[index]["labels"] = row["CategoryId"]

            self.image_info[index]["orig_height"] = row["Height"]

            self.image_info[index]["orig_width"] = row["Width"]

            self.image_info[index]["annotations"] = row["EncodedPixels"]



    def __getitem__(self, idx):

        

        img_path = self.image_info[idx]["image_path"]

        

        img = cv2.imread(img_path)

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.width, self.height))



        info = self.image_info[idx]

        

        mask = np.zeros((self.width, self.height, self.num_classes))

        labels = np.zeros(self.num_classes)

        

        for annotation, label in zip(info['annotations'], info['labels']):

            cur_mask = rle_decode(annotation, (info['orig_height'], info['orig_width']))

            mask[:, :, int(label)] += cv2.resize(cur_mask, (self.width, self.height))

            labels[int(label)] = 1

            

        mask = (mask > 0.5).astype(np.float32)

        

        # apply augmentations

        if self.augmentation is not None:

            sample = self.augmentation(image=img, mask=mask)

            img, mask = sample['image'], sample['mask']

        

        # apply preprocessing

        if self.preprocessing is not None:

            sample = self.preprocessing(image=img, mask=mask)

            img, mask = sample['image'], sample['mask']

            

        return img, mask, labels



    def __len__(self):

        return len(self.image_info)



    

def collate_function(batch):

    image_array = torch.zeros((len(batch), batch[0][0].shape[0], batch[0][0].shape[1], batch[0][0].shape[2]))

    mask_array = torch.zeros((len(batch), batch[0][1].shape[0], batch[0][1].shape[1], batch[0][1].shape[2]))

    label_array = torch.zeros((len(batch), batch[0][2].shape[0]))

    

    for i in range(len(batch)):

        image_array[i,:,:,:] = torch.Tensor(batch[i][0])

        mask_array[i,:,:,:] = torch.Tensor(batch[i][1])

        label_array[i,:] = torch.Tensor(batch[i][2])

        

    return image_array, (mask_array, label_array)
def get_training_augmentation():

    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.VerticalFlip(p=0.5),

    ]

    return albu.Compose(train_transform)





def to_tensor(x, **kwargs):

    return x.transpose(2, 0, 1).astype('float32')



def get_preprocessing(preprocessing_fn):

    """Construct preprocessing transform

    

    Args:

        preprocessing_fn (callbale): data normalization function 

            (can be specific for each pretrained neural network)

    Return:

        transform: albumentations.Compose

    

    """

    

    _transform = [

        albu.Lambda(image=preprocessing_fn),

        albu.Lambda(image=to_tensor, mask=to_tensor),

    ]

    return albu.Compose(_transform)

class FirstHeadDiceSecondHeadBCE(smp.utils.base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):

        super().__init__(**kwargs)

        self.dice_loss = smp.utils.losses.DiceLoss(eps=1., beta=1., activation=None, ignore_channels=None, **kwargs)

        self.bce = smp.utils.losses.BCEWithLogitsLoss()

        

    def forward(self, y_pr, y_gt):

        return self.dice_loss(y_pr[0], y_gt[0]) + self.bce(y_pr[1], y_gt[1])
class MyEpoch(smp.utils.train.Epoch):

    def run(self, dataloader):



        self.on_epoch_start()



        logs = {}

        loss_meter = smp.utils.meter.AverageValueMeter()

        metrics_meters = {metric.__name__: smp.utils.meter.AverageValueMeter() for metric in self.metrics}



        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:

            for x, y in iterator:

                x, y = x.to(self.device), (y[0].to(self.device), y[1].to(self.device))

                loss, y_pred = self.batch_update(x, y)



                # update loss logs

                loss_value = loss.cpu().detach().numpy()

                loss_meter.add(loss_value)

                loss_logs = {self.loss.__name__: loss_meter.mean}

                logs.update(loss_logs)



                # update metrics logs

                for metric_fn in self.metrics:

                    metric_value = metric_fn(y_pred[0], y[0]).cpu().detach().numpy()

                    metrics_meters[metric_fn.__name__].add(metric_value)

                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}

                logs.update(metrics_logs)



                if self.verbose:

                    s = self._format_logs(logs)

                    iterator.set_postfix_str(s)



        return logs

    

class TrainEpoch(MyEpoch):



    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):

        super().__init__(

            model=model,

            loss=loss,

            metrics=metrics,

            stage_name='train',

            device=device,

            verbose=verbose,

        )

        self.optimizer = optimizer



    def on_epoch_start(self):

        self.model.train()



    def batch_update(self, x, y):

        self.optimizer.zero_grad()

        prediction = self.model.forward(x)

        loss = self.loss(prediction, y)

        loss.backward()

        self.optimizer.step()

        return loss, prediction

    

class ValidEpoch(MyEpoch):



    def __init__(self, model, loss, metrics, device='cpu', verbose=True):

        super().__init__(

            model=model,

            loss=loss,

            metrics=metrics,

            stage_name='valid',

            device=device,

            verbose=verbose,

        )



    def on_epoch_start(self):

        self.model.eval()



    def batch_update(self, x, y):

        with torch.no_grad():

            prediction = self.model.forward(x)

            loss = self.loss(prediction, y)

        return loss, prediction
ENCODER = 'mobilenet_v2'

ENCODER_WEIGHTS = 'imagenet'

DEVICE = 'cuda'



ACTIVATION = 'sigmoid'



aux_params=dict(

    pooling='avg',             

    dropout=0.2,               

    activation=None,      

    classes=NUM_CLASSES,                 

)
model = smp.Unet(

    encoder_name=ENCODER, 

    encoder_weights=ENCODER_WEIGHTS, 

    classes=NUM_CLASSES, 

    activation=ACTIVATION,

    aux_params=aux_params

)



preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = UnetDataset(

    root_path_train,

    train_df,

    IMG_HEIGHT,

    IMG_WIDTH, 

    preprocessing=get_preprocessing(preprocessing_fn),

    augmentation=get_training_augmentation()

)



valid_dataset = UnetDataset(

    root_path_train,

    val_df,

    IMG_HEIGHT,

    IMG_WIDTH, 

    preprocessing=get_preprocessing(preprocessing_fn),

)





train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS, collate_fn=collate_function)

valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, collate_fn=collate_function)
loss = FirstHeadDiceSecondHeadBCE()

metrics = [

    smp.utils.metrics.IoU(threshold=0.5),

]



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# create epoch runners 

# it is a simple loop of iterating over dataloader`s samples

train_epoch = TrainEpoch(

    model, 

    loss=loss, 

    metrics=metrics, 

    optimizer=optimizer,

    device=DEVICE,

    verbose=True,

)



valid_epoch = ValidEpoch(

    model, 

    loss=loss, 

    metrics=metrics, 

    device=DEVICE,

    verbose=True,

)
torch.save(model.state_dict(), 'best_model.pth')

max_score = 0



for i in range(0, 2):

    

    print('\nEpoch: {}'.format(i))

    train_logs = train_epoch.run(train_loader)

    valid_logs = valid_epoch.run(valid_loader)

    

    # do something (save model, change lr, etc.)

    if max_score < valid_logs['iou_score']:

        max_score = valid_logs['iou_score']

        torch.save(model.state_dict(), 'best_model.pth')

        print('Model saved!')