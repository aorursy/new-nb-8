#!/usr/bin/env python
# coding: utf-8



import numpy as np, pandas as pd, os, gc
import matplotlib.pyplot as plt, time
from PIL import Image 
import warnings
import random
warnings.filterwarnings("ignore")

path = '../input/severstal-steel-defect-detection/'
train = pd.read_csv(path + 'train.csv')

# RESTRUCTURE TRAIN DATAFRAME
train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')
train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})
train2['e1'] = train['EncodedPixels'][::4].values
train2['e2'] = train['EncodedPixels'][1::4].values
train2['e3'] = train['EncodedPixels'][2::4].values
train2['e4'] = train['EncodedPixels'][3::4].values
train2.reset_index(inplace=True,drop=True)
train2.fillna('',inplace=True); 
train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values

indexes = list(range(len(train2)))
random.shuffle(indexes)
train_ratio = 0.95
partio = int(len(train2) * train_ratio)
train_indexes = indexes[:partio]
val_indexes = indexes[partio:]
train_df = train2.iloc[train_indexes, :]
val_df = train2.iloc[val_indexes, :]




from albumentations import (
    Compose, HorizontalFlip, ShiftScaleRotate, PadIfNeeded, RandomCrop,
    RGBShift, RandomBrightness, RandomContrast, VerticalFlip, 
)
crop_size = [256, 416]
train_augmentator = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightness(limit=(-0.25, 0.25), p=0.75),
        RandomContrast(limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1)




import mxnet as mx
from mxnet.gluon import data, HybridBlock, nn
import pandas as pd
import cv2
import os
import numpy as np
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.model_zoo import vision
from mxnet.lr_scheduler import CosineScheduler
from mxnet.gluon import loss, Trainer
from mxnet import autograd
import random
from PIL import Image, ImageOps, ImageFilter
from mxnet import nd as F, lr_scheduler as lrs
from mxnet.gluon.contrib.estimator import Estimator
import gluoncv.model_zoo  as gm

def scale_func(image_shape):
    return random.uniform(0.5, 1.2)


class SteelDataset(data.Dataset):
    def __init__(self, df, img_dir, debug=False):
        
        self.train_df = df
        self.root_dir = img_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)
                )
            ]
        )
        
        self.debug = debug
        
    def __getitem__(self, i):
        if self.debug:
            curr_df = self.train_df.head(20)
        masks = np.zeros((256, 1600), np.uint8)
        img_names = []
        item = self.train_df.iloc[i, :]
        img_name = item['ImageId']
        for j in range(4):
            curr_item = item["e{}".format(j+1)]
            if len(curr_item) > 0:
                rle_pixels = curr_item
                label = rle_pixels.split(" ")
                positions = list(map(int, label[0::2]))
                length = list(map(int, label[1::2]))
                mask = np.zeros(256 * 1600, dtype=np.uint8)
                for pos, le in zip(positions, length):
                    mask[pos - 1:(pos + le - 1)] = j+1
                count = np.sum(np.where(mask==(j+1), 1, 0))
                if count < 8:
                    mask = np.where(mask==(j+1), -1, 0)
                    
                masks[ :, :] += mask.reshape(256, 1600, order='F')
                
        oimg = cv2.imread(os.path.join(self.root_dir, img_name))[:, :, ::-1]
        oimg, masks = self.rescale_sample(oimg, masks)
        aug_out = train_augmentator(image=oimg, mask=masks)
        oimg = aug_out['image']
        masks = aug_out['mask']
        img = F.array(oimg)
        img = self.transform(img)
        
        if self.debug:
            return img, F.array(masks[::4, ::4]), oimg, masks, curr_df
        else:
            return img, F.array(masks)
        
    def __len__(self):
        return len(self.train_df)


    def rescale_sample(self, image, mask):

        scale = scale_func(image.shape)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        new_size = (image.shape[1], image.shape[0])

        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        return image, mask




# for test
import matplotlib.pyplot as plt
csv_file = 'train.csv'
img_dir = '../input/severstal-steel-defect-detection/train_images/'
steel_dataset = SteelDataset(train2, img_dir, debug=True)
print(len(steel_dataset))
_, mm, im, mask, curr_df = steel_dataset[11]
plt.figure(figsize=(20, 20))
plt.subplot(2, 1, 1)
plt.imshow(im)
plt.subplot(2, 1, 2)
plt.imshow(mask[::4, ::4])
mm.flatten().shape




from gluoncv.model_zoo.resnetv1b import resnet50_v1s, resnet101_v1s, resnet152_v1s
import mxnet as mx

class ResNetBackbone(mx.gluon.HybridBlock):
    def __init__(self, backbone='resnet101', pretrained_base=True,dilated=True, **kwargs):
        super(ResNetBackbone, self).__init__()

        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = resnet50_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            elif backbone == 'resnet152':
                pretrained = resnet152_v1s(pretrained=pretrained_base, dilated=dilated, **kwargs)
            else:
                raise RuntimeError(f'unknown backbone: {backbone}')

            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4




import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

class ResNetFPN(mx.gluon.HybridBlock):
    def __init__(self, backbone= 'resnet101', backbone_lr_mult=0.1, **kwargs):
        super(ResNetFPN, self).__init__()

        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, dilated=False, **kwargs)

            self.head = _FPNHead(output_channels=256, **kwargs)

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, dilated=False, **self._kwargs)
        backbone_params = self.backbone.collect_params()
        pretrained_weights = pretrained.collect_params()
        for k, v in pretrained_weights.items():
            param_name = backbone_params.prefix + k[len(pretrained_weights.prefix):]
            backbone_params[param_name].set_data(v.data())

        self.backbone.collect_params().setattr('lr_mult', self.backbone_lr_mult)

    def hybrid_forward(self,F, x):
        c1, c2, c3, c4 = self.backbone(x)
        p1, p2, p3, p4 = self.head(c1, c2, c3, c4)

        return p1, p2, p3, p4

class ResNetUnet(mx.gluon.HybridBlock):
    def __init__(self, backbone= 'resnet101', backbone_lr_mult=0.1, **kwargs):
        super(ResNetUnet, self).__init__()

        self.backbone_name = backbone
        self.backbone_lr_mult = backbone_lr_mult
        self._kwargs = kwargs

        with self.name_scope():
            self.backbone = ResNetBackbone(backbone=self.backbone_name, pretrained_base=False, dilated=False, **kwargs)

            self.head = _UnetHead(**kwargs)

    def load_pretrained_weights(self):
        pretrained = ResNetBackbone(backbone=self.backbone_name, pretrained_base=True, dilated=False, **self._kwargs)
        backbone_params = self.backbone.collect_params()
        pretrained_weights = pretrained.collect_params()
        for k, v in pretrained_weights.items():
            param_name = backbone_params.prefix + k[len(pretrained_weights.prefix):]
            backbone_params[param_name].set_data(v.data())

        self.backbone.collect_params().setattr('lr_mult', self.backbone_lr_mult)

    def hybrid_forward(self,F, x):
        c1, c2, c3, c4 = self.backbone(x)
        out = self.head(c1, c2, c3, c4)

        return out

class _DecoderBlock(HybridBlock):
    def __init__(self, output_channels, norm_layer=nn.BatchNorm):
        super(_DecoderBlock, self).__init__()

        with self.name_scope():
            self.block = nn.HybridSequential()
            self.block.add(ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer))
            self.block.add(ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer))

    def hybrid_forward(self, F, x, y=None):
        if y is not None:
            x = F.contrib.BilinearResize2D(x, scale_height=2, scale_width=2)
            x = F.concat(x, y, dim=1)
        out = self.block(x)
        return out


class _UnetHead(HybridBlock):
    def __init__(self, num_classes, output_channels=[256, 128, 64, 32], scale=4, norm_layer=nn.BatchNorm):
        super(_UnetHead, self).__init__()
        
        self.scale = scale
        with self.name_scope():
            self.block4 = _DecoderBlock(output_channels[0], norm_layer=norm_layer)
            self.block3 = _DecoderBlock(output_channels[1], norm_layer=norm_layer)
            self.block2 = _DecoderBlock(output_channels[2], norm_layer=norm_layer)
            self.block1 = _DecoderBlock(output_channels[3], norm_layer=norm_layer)
            self.postprocess_block = nn.Conv2D(num_classes, kernel_size=1)

    def hybrid_forward(self, F, c1, c2, c3, c4):

        p4 = self.block4(c4)
        p3 = self.block3(p4, c3)
        p2 = self.block2(p3, c2)
        p1 = self.block1(p2, c1)
        if self.scale > 1:
            p1 = F.contrib.BilinearResize2D(p1, scale_height=self.scale, scale_width=self.scale)
        out = self.postprocess_block(p1)

        return out


class _FPNHead(HybridBlock):
    def __init__(self, output_channels=256, norm_layer=nn.BatchNorm):
        super(_FPNHead, self).__init__()
        self._hdsize = {}

        with self.name_scope():
            self.block4 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block3 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block2 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)
            self.block1 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)

    def hybrid_forward(self, F, c1, c2, c3, c4):
        p4 = self.block4(c4)
        p3 = self._resize_as(F, 'id_1', p4, c3) + self.block3(c3)
        p2 = self._resize_as(F, 'id_2', p3, c2) + self.block2(c2)
        p1 = self._resize_as(F, 'id_3', p2, c1) + self.block1(c1)

        return p1, p2, p3, p4

    def _resize_as(self, F, name, x, y):
        h_key = name + '_h'
        w_key = name + '_w'

        if hasattr(y, 'shape'):
            _, _, h, w = y.shape
            _, _, h2, w2 = x.shape

            if h == h2 and w == w2:
                h = 0
                w = 0

            self._hdsize[h_key] = h
            self._hdsize[w_key] = w
        else:
            h, w = self._hdsize[h_key], self._hdsize[w_key]

        if h == 0 and w == 0:
            return x
        else:
            return F.contrib.BilinearResize2D(x, height=h, width=w)


class SemanticFPNHead(HybridBlock):
    def __init__(self, num_classes, output_channels=128, norm_layer=nn.BatchNorm):
        super(SemanticFPNHead, self).__init__()
        self._hdsize = {}

        with self.name_scope():
            self.block4_1 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block4_2 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block4_3 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)

            self.block3_1 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block3_2 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)

            self.block2 = ConvBlock(output_channels, kernel_size=3, padding=1, norm_layer=norm_layer)
            self.block1 = ConvBlock(output_channels, kernel_size=1, norm_layer=norm_layer)

            self.postprocess_block = nn.Conv2D(num_classes, kernel_size=1)

    def hybrid_forward(self, F, c1, c2, c3, c4):
        out4 = self._resize_as(F, 'id_1', self.block4_1(c4), c3)
        out4 = self._resize_as(F, 'id_2', self.block4_2(out4), c2)
        out4 = self._resize_as(F, 'id_3', self.block4_3(out4), c1)

        out3 = self._resize_as(F, 'id_4', self.block3_1(c3), c2)
        out3 = self._resize_as(F, 'id_5', self.block3_2(out3), c1)

        out2 = self._resize_as(F, 'id_6', self.block2(c2), c1)

        out1 = self.block1(c1)

        out = out1 + out2 + out3 + out4

        out = self.postprocess_block(out)
        out = F.contrib.BilinearResize2D(out,scale_height=4,scale_width=4)
        return out

    def _resize_as(self, F,name, x, y):
        h_key = name + '_h'
        w_key = name + '_w'

        if hasattr(y, 'shape'):
            _, _, h, w = y.shape
            _, _, h2, w2 = x.shape

            if h == h2 and w == w2:
                h = 0
                w = 0

            self._hdsize[h_key]=h
            self._hdsize[w_key]=w
        else:
            h, w = self._hdsize[h_key], self._hdsize[w_key]

        if h == 0 and w == 0:
            return x
        else:
            return F.contrib.BilinearResize2D(x,height=h,width=w)


class ConvBlock(HybridBlock):
    def __init__(self, output_channels, kernel_size, padding=0, activation='relu', norm_layer=nn.BatchNorm):
        super().__init__()
        self.body = nn.HybridSequential()
        self.body.add(
            nn.Conv2D(output_channels, kernel_size=kernel_size, padding=padding, activation=activation),
            norm_layer(in_channels=output_channels)
        )

    def hybrid_forward(self, F, x):
        return self.body(x)




class SteelFPN(HybridBlock):
    
    def __init__(self, n_classes=5, ctx=mx.cpu()):
        super().__init__()
        with self.name_scope():
            self.feature_extractor = ResNetFPN()
            self.segment_head = SemanticFPNHead(num_classes=n_classes)
    def hybrid_forward(self, F, x):
        fpn_feature = self.feature_extractor(x)
        segment_out = self.segment_head(*fpn_feature)
        return segment_out




# unet = ResNetUnet(output_channels=[256, 128, 64, 32], num_classes=5)
# unet.collect_params().initialize()
# unet.load_pretrained_weights()
# a = mx.nd.normal(shape=(1, 3, 512, 512))
# out = unet(a)
# print(out.shape)




import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like

class NormalizedFocalLossSoftmax(Loss):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, detach_delimeter=True, gamma=2, eps=1e-10, **kwargs):
        super(NormalizedFocalLossSoftmax, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._eps = eps
        self._gamma = gamma
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label):
        label = F.expand_dims(label, axis=1)
        softmaxout = F.softmax(pred, axis=1)

        t = label != self._ignore_label
        pt = F.pick(softmaxout, label, axis=1, keepdims=True)
        pt = F.where(t, pt, F.ones_like(pt))
        beta = (1 - pt) ** self._gamma

        t_sum = F.cast(F.sum(t, axis=(-2, -1), keepdims=True), 'float32')
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = F.broadcast_mul(beta, mult)
        self._k_sum = 0.9 * self._k_sum + 0.1 * mult.asnumpy().mean()

        loss = -beta * F.log(F.minimum(pt + self._eps, 1))

        if self._size_average:
            bsum = F.sum(t_sum, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (bsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class NormalizedFocalLossSigmoid(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, size_average=True, detach_delimeter=True,
                 eps=1e-12, scale=1.0,
                 ignore_label=-1, **kwargs):
        super(NormalizedFocalLossSigmoid, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._ignore_label = ignore_label

        self._scale = scale
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average
        self._detach_delimeter = detach_delimeter
        self._k_sum = 0

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        one_hot = label > 0
        t = F.ones_like(one_hot)

        if not self._from_logits:
            pred = F.sigmoid(pred)

        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        pt = F.where(one_hot, pred, 1 - pred)
        pt = F.where(label != self._ignore_label, pt, F.ones_like(pt))

        beta = (1 - pt) ** self._gamma

        t_sum = F.sum(t, axis=(-2, -1), keepdims=True)
        beta_sum = F.sum(beta, axis=(-2, -1), keepdims=True)
        mult = t_sum / (beta_sum + self._eps)
        if self._detach_delimeter:
            mult = mult.detach()
        beta = F.broadcast_mul(beta, mult)

        ignore_area = F.sum(label == -1, axis=0, exclude=True).asnumpy()
        sample_mult = F.mean(mult, axis=0, exclude=True).asnumpy()
        if np.any(ignore_area == 0):
            self._k_sum = 0.9 * self._k_sum + 0.1 * sample_mult[ignore_area == 0].mean()

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != self._ignore_label

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            bsum = F.sum(sample_weight, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (bsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss

    def log_states(self, sw, name, global_step):
        sw.add_scalar(tag=name + '_k', value=self._k_sum, global_step=global_step)


class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2,
                 from_logits=False, batch_axis=0,
                 weight=None, num_class=None,
                 eps=1e-9, size_average=True, scale=1.0, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

        self._scale = scale
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        if not self._from_logits:
            pred = F.sigmoid(pred)

        one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)

        t = label != -1
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        beta = (1 - pt) ** self._gamma

        loss = -alpha * beta * F.log(F.minimum(pt + self._eps, 1))
        sample_weight = label != -1

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            tsum = F.sum(label == 1, axis=self._batch_axis, exclude=True)
            loss = F.sum(loss, axis=self._batch_axis, exclude=True) / (tsum + self._eps)
        else:
            loss = F.sum(loss, axis=self._batch_axis, exclude=True)

        return self._scale * loss


class SoftmaxCrossEntropyLoss(Loss):
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, grad_scale=1.0, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average
        self._grad_scale = grad_scale

    def hybrid_forward(self, F, pred, label):
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null',
            grad_scale=self._grad_scale,
        )
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SigmoidBinaryCrossEntropyLoss(Loss):
    def __init__(self, from_sigmoid=False, weight=None, batch_axis=0, ignore_label=-1, **kwargs):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._from_sigmoid = from_sigmoid
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, pred, label):
        label = _reshape_like(F, label, pred)
        sample_weight = label != self._ignore_label
        label = F.where(sample_weight, label, F.zeros_like(label))

        if not self._from_sigmoid:
            loss = F.relu(pred) - pred * label +                 F.Activation(-F.abs(pred), act_type='softrelu')
        else:
            eps = 1e-12
            loss = -(F.log(pred + eps) * label
                     + F.log(1. - pred + eps) * (1. - label))

        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)




def compute_iou(label, pred):
    union = np.logical_or(label, pred)
    intersection = np.logical_and(label, pred)
    iou = intersection / (union + 1e-5)
    return np.mean(iou)

def iou_metric(labels, preds):
    
#     labels = F.array(labels)
#     preds = F.array(preds)
    labels = labels.asnumpy()
    preds = F.argmax(F.softmax(preds, axis=1), axis=1).asnumpy()
    ious = []
    for i in range(5):
        curr_pred = np.where(preds==i, 1, 0)
        curr_labels = np.where(labels==i, 1, 0)
        curr_iou = compute_iou(curr_labels, curr_pred)
        ious.append(curr_iou)
    mean_iou = np.mean(ious)
    ious.append(mean_iou)
#     print("IOU_INFO:: bg:{}, 1:{}, 2:{}, 3:{}, 4:{}, mean_iou:{}".format(*ious))
    cls = ['bg', '1', '2', '3', '4', 'mean_iou']
    return {k:v for k, v in zip(cls, ious)}




def training(epoch, data, net, loss, trainer, ctx):
    train_loss = 0.0
    train_iou = 0.0
    bg_iou, iou1, iou2, iou3, iou4 = [0.0] * 5
    hybridize = False
    tbar = tqdm(data)
    for i, batch_data in enumerate(tbar):
        image, mask = batch_data
        image = image.as_in_context(ctx)
        mask = mask.as_in_context(ctx)
        with autograd.record():
            outputs = net(image)
            losses = loss(outputs, mask)
            ious = iou_metric(mask, outputs)
        losses.backward()
        global_step = epoch * len(data) + i
        trainer.step(len(batch_data))

        batch_loss = sum(loss.asnumpy().mean() for loss in losses) / len(losses)
        train_loss += batch_loss
        train_iou += ious['mean_iou']
        bg_iou += ious['bg']
        iou1 += ious['1']
        iou2 += ious['2']
        iou3 += ious['3']
        iou4 += ious['4']
        if i % 20:
            tbar.set_description(f'Epoch {epoch}, training loss {train_loss/(i+1):.6f}, training_ious:{train_iou/(i+1):.6f}, bg_ious:{bg_iou/(i+1):.6f},class1_ious:{iou1/(i+1):.6f},class2_ious:{iou2/(i+1):.6f}, class3_ious:{iou3/(i+1):.6f}, class4_ious:{iou4/(i+1):.6f}')




def evaluation(data, net, ctx):
    val_iou = 0.0
    bg_iou, iou1, iou2, iou3, iou4 = [0.0] * 5
    hybridize = False
    tbar = tqdm(data)
    for i, batch_data in enumerate(tbar):
        image, mask = batch_data
        image = image.as_in_context(ctx)
        mask = mask.as_in_context(ctx)
        outputs = net(image)
        ious = iou_metric(mask, outputs)
        val_iou += ious['mean_iou']
        bg_iou += ious['bg']
        iou1 += ious['1']
        iou2 += ious['2']
        iou3 += ious['3']
        iou4 += ious['4']
        if i % 20:
            tbar.set_description(f'val_ious:{val_iou/(i+1):.6f}, bg_ious:{bg_iou/(i+1):.6f},class1_ious:{iou1/(i+1):.6f},class2_ious:{iou2/(i+1):.6f}, class3_ious:{iou3/(i+1):.6f}, class4_ious:{iou4/(i+1):.6f}')
    return val_iou * 1.0 /(i+1)




import os
from tqdm import tqdm
def train_from_manual(train_df, val_df, img_dir, batch_size, epoches, lr=0.001, ctx=mx.cpu()):
    # TODO: finish trainer .etc, add ctx
    steel_dataset = SteelDataset(train_df, img_dir)
    steel_data = data.DataLoader(steel_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    
    val_steel_dataset = SteelDataset(val_df, img_dir)
    val_steel_data = data.DataLoader(val_steel_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    normal_focal_loss = NormalizedFocalLossSoftmax(ignore_label=-1, gamma=1)
#     normal_focal_loss = SoftmaxCrossEntropyLoss()
#     unet = SteelUnet(n_classes=5, ctx=ctx)
#     unet.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2), ctx=ctx)
#     unet.feature_extractor.load_pretrained_weights()
    
    unet = ResNetUnet(output_channels=[256, 128, 64, 32], num_classes=5)
    unet.initialize(mx.init.Xavier(rnd_type='gaussian', magnitude=2), ctx=ctx)
    unet.load_pretrained_weights()
    for k, v in unet.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    lr_sche = lrs.FactorScheduler(step=5, base_lr=lr, factor=0.7,  warmup_steps=2, warmup_begin_lr=0.00002)
    trainer = Trainer(unet.collect_params(), 'adam', 
                        {'learning_rate': lr,
                         'wd':1e-5,
#                          'lr_scheduler': lr_sche
                        })
    for epoch in range(epoches):
        max_iou = -1
        if epoch in [10, 15, 20, 25, 30]:
            lr = lr * 0.7
            trainer.set_learning_rate(lr=lr)
        training(epoch, steel_data, unet, normal_focal_loss, trainer, ctx)
        if epoch % 2 == 0:
            val_iou = evaluation(val_steel_data, unet, ctx)
            unet.save_parameters('unet_{}_{}.params'.format(epoch, max_iou))




batch_size = 12
csv_file = '../input/severstal-steel-defect-detection/train.csv'
img_dir = '../input/severstal-steel-defect-detection/train_images/'

epoches = 15
train_from_manual(train_df, val_df, img_dir, batch_size, epoches, ctx=mx.gpu())






