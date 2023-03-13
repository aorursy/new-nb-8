import collections

import copy

import datetime

import errno

import gc

import glob

import json

import logging

import os

import pickle

import random

import time

import warnings

from collections import defaultdict, deque

from functools import partial

from pprint import pprint



import albumentations as albu

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

import torch.distributed as dist

import torchvision

from joblib import Parallel, delayed

from matplotlib import patches

from PIL import Image, ImageFile

from skimage.measure import label, regionprops

from sklearn.model_selection import KFold, train_test_split

from torch import nn

from torchvision import transforms as T

from torchvision.models import resnet

from torchvision.models.detection import FasterRCNN, MaskRCNN

from torchvision.models.detection.backbone_utils import (BackboneWithFPN,

                                                         resnet_fpn_backbone)

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection.transform import GeneralizedRCNNTransform

from torchvision.ops import misc as misc_nn_ops

from tqdm import tqdm




warnings.filterwarnings("ignore")
class RunLogger(object):

    def __init__(self, fname):

        self.fname = fname



    def initialize(self):

        import importlib



        importlib.reload(logging)

        logging.basicConfig(

            filename=self.fname,

            format="%(asctime)s: %(message)s",

            datefmt="%m/%d/%Y %I:%M:%S %p",

            level=logging.INFO,

        )

        return



    def log(self, msg):

        logging.info(msg)

        return





class SmoothedValue(object):

    """Track a series of values and provide access to smoothed values over a

    window or the global series average.

    """



    def __init__(self, window_size=20, fmt=None):

        if fmt is None:

            fmt = "{median:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)

        self.total = 0.0

        self.count = 0

        self.fmt = fmt



    def update(self, value, n=1):

        self.deque.append(value)

        self.count += n

        self.total += value * n



    def synchronize_between_processes(self):

        """

        Warning: does not synchronize the deque!

        """

        if not is_dist_avail_and_initialized():

            return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")

        dist.barrier()

        dist.all_reduce(t)

        t = t.tolist()

        self.count = int(t[0])

        self.total = t[1]



    @property

    def median(self):

        d = torch.tensor(list(self.deque))

        return d.median().item()



    @property

    def avg(self):

        d = torch.tensor(list(self.deque), dtype=torch.float32)

        return d.mean().item()



    @property

    def global_avg(self):

        return self.total / self.count



    @property

    def max(self):

        return max(self.deque)



    @property

    def value(self):

        return self.deque[-1]



    def __str__(self):

        return self.fmt.format(

            median=self.median,

            avg=self.avg,

            global_avg=self.global_avg,

            max=self.max,

            value=self.value,

        )





def all_gather(data):

    """

    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Args:

        data: any picklable object

    Returns:

        list[data]: list of data gathered from each rank

    """

    world_size = get_world_size()

    if world_size == 1:

        return [data]



    # serialized to a Tensor

    buffer = pickle.dumps(data)

    storage = torch.ByteStorage.from_buffer(buffer)

    tensor = torch.ByteTensor(storage).to("cuda")



    # obtain Tensor size of each rank

    local_size = torch.tensor([tensor.numel()], device="cuda")

    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]

    dist.all_gather(size_list, local_size)

    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)



    # receiving Tensor from all ranks

    # we pad the tensor because torch all_gather does not support

    # gathering tensors of different shapes

    tensor_list = []

    for _ in size_list:

        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))

    if local_size != max_size:

        padding = torch.empty(

            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"

        )

        tensor = torch.cat((tensor, padding), dim=0)

    dist.all_gather(tensor_list, tensor)



    data_list = []

    for size, tensor in zip(size_list, tensor_list):

        buffer = tensor.cpu().numpy().tobytes()[:size]

        data_list.append(pickle.loads(buffer))



    return data_list





def reduce_dict(input_dict, average=True):

    """

    Args:

        input_dict (dict): all the values will be reduced

        average (bool): whether to do average or sum

    Reduce the values in the dictionary from all processes so that all processes

    have the averaged results. Returns a dict with the same fields as

    input_dict, after reduction.

    """

    world_size = get_world_size()

    if world_size < 2:

        return input_dict

    with torch.no_grad():

        names = []

        values = []

        # sort the keys so that they are consistent across processes

        for k in sorted(input_dict.keys()):

            names.append(k)

            values.append(input_dict[k])

        values = torch.stack(values, dim=0)

        dist.all_reduce(values)

        if average:

            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict





class MetricLogger(object):

    def __init__(self, delimiter="\t", logger=None):

        self.meters = defaultdict(SmoothedValue)

        self.delimiter = delimiter

        self.logger = logger



    def update(self, **kwargs):

        for k, v in kwargs.items():

            if isinstance(v, torch.Tensor):

                v = v.item()

            assert isinstance(v, (float, int))

            self.meters[k].update(v)



    def __getattr__(self, attr):

        if attr in self.meters:

            return self.meters[attr]

        if attr in self.__dict__:

            return self.__dict__[attr]

        raise AttributeError(

            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)

        )



    def __str__(self):

        loss_str = []

        for name, meter in self.meters.items():

            loss_str.append("{}: {}".format(name, str(meter)))

        return self.delimiter.join(loss_str)



    def synchronize_between_processes(self):

        for meter in self.meters.values():

            meter.synchronize_between_processes()



    def add_meter(self, name, meter):

        self.meters[name] = meter



    def log_every(self, iterable, print_freq, header=None):

        i = 0

        if not header:

            header = ""

        start_time = time.time()

        end = time.time()

        iter_time = SmoothedValue(fmt="{avg:.4f}")

        data_time = SmoothedValue(fmt="{avg:.4f}")

        space_fmt = ":" + str(len(str(len(iterable)))) + "d"

        log_msg = self.delimiter.join(

            [

                header,

                "[{0" + space_fmt + "}/{1}]",

                "eta: {eta}",

                "{meters}",

                "time: {time}",

                "data: {data}",

                "max mem: {memory:.0f}",

            ]

        )

        MB = 1024.0 * 1024.0

        for obj in iterable:

            data_time.update(time.time() - end)

            yield obj

            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:

                eta_seconds = iter_time.global_avg * (len(iterable) - i)

                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                print(

                    log_msg.format(

                        i,

                        len(iterable),

                        eta=eta_string,

                        meters=str(self),

                        time=str(iter_time),

                        data=str(data_time),

                        memory=torch.cuda.max_memory_allocated() / MB,

                    )

                )

                if self.logger is not None:

                    self.logger.log(

                        log_msg.format(

                            i,

                            len(iterable),

                            eta=eta_string,

                            meters=str(self),

                            time=str(iter_time),

                            data=str(data_time),

                            memory=torch.cuda.max_memory_allocated() / MB,

                        )

                    )

            i += 1

            end = time.time()

        total_time = time.time() - start_time

        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        print(

            "{} Total time: {} ({:.4f} s / it)".format(

                header, total_time_str, total_time / len(iterable)

            )

        )

        if self.logger is not None:

            self.logger.log(

                "{} Total time: {} ({:.4f} s / it)".format(

                    header, total_time_str, total_time / len(iterable)

                )

            )





def collate_fn(batch):

    return tuple(zip(*batch))





def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):

        if x >= warmup_iters:

            return 1

        alpha = float(x) / warmup_iters

        return warmup_factor * (1 - alpha) + alpha



    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)





def mkdir(path):

    try:

        os.makedirs(path)

    except OSError as e:

        if e.errno != errno.EEXIST:

            raise





def setup_for_distributed(is_master):

    """

    This function disables printing when not in master process

    """

    import builtins as __builtin__



    builtin_print = __builtin__.print



    def print(*args, **kwargs):

        force = kwargs.pop("force", False)

        if is_master or force:

            builtin_print(*args, **kwargs)



    __builtin__.print = print





def is_dist_avail_and_initialized():

    if not dist.is_available():

        return False

    if not dist.is_initialized():

        return False

    return True





def get_world_size():

    if not is_dist_avail_and_initialized():

        return 1

    return dist.get_world_size()





def get_rank():

    if not is_dist_avail_and_initialized():

        return 0

    return dist.get_rank()





def is_main_process():

    return get_rank() == 0





def save_on_master(*args, **kwargs):

    if is_main_process():

        torch.save(*args, **kwargs)





def init_distributed_mode(args):

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:

        args.rank = int(os.environ["RANK"])

        args.world_size = int(os.environ["WORLD_SIZE"])

        args.gpu = int(os.environ["LOCAL_RANK"])

    elif "SLURM_PROCID" in os.environ:

        args.rank = int(os.environ["SLURM_PROCID"])

        args.gpu = args.rank % torch.cuda.device_count()

    else:

        print("Not using distributed mode")

        args.distributed = False

        return



    args.distributed = True



    torch.cuda.set_device(args.gpu)

    args.dist_backend = "nccl"

    print(

        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True

    )

    torch.distributed.init_process_group(

        backend=args.dist_backend,

        init_method=args.dist_url,

        world_size=args.world_size,

        rank=args.rank,

    )

    torch.distributed.barrier()

    setup_for_distributed(args.rank == 0)
# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode

def mask2rle(img):

    """

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    """

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return " ".join(str(x) for x in runs)





# https://www.kaggle.com/titericz/building-and-visualizing-masks

def rle2mask(rle, imgshape):

    width = imgshape[0]

    height = imgshape[1]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]

    current_position = 0

    for index, start in enumerate(starts):

        mask[int(start) : int(start + lengths[index])] = 1

        current_position += lengths[index]

    return np.flipud(np.rot90(mask.reshape(height, width), k=1))





def create_bboxes(mask, area_threshold=100):

    label_image = label(mask)

    regions = regionprops(label_image)

    bboxes = []

    for r in regions:

        ymin, xmin, ymax, xmax = r.bbox

        bbox = [xmin, ymin, xmax, ymax]

        bboxes.append(bbox)

    bboxes = np.asarray(bboxes)

    area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

    bboxes_valid = np.where(area > area_threshold)[0]

    bboxes = bboxes[bboxes_valid]

    area = area[bboxes_valid]

    return bboxes, area





def create_instance_masks(mask, bboxes):

    mask_new = np.zeros(((len(bboxes),) + mask.shape), dtype=np.uint8)

    for idx, b in enumerate(bboxes):

        xmin, ymin, xmax, ymax = b

        mask_subset = mask[ymin:ymax, xmin:xmax]

        mask_new[idx][ymin:ymax, xmin:xmax] = mask_subset

    return mask_new
class SteelMrcnnDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, df, transform=None):

        self.df = df

        self.transform = transform

        self.image_dir = image_dir

        self.image_info = collections.defaultdict(dict)

        self.ids = self.df["ImageId"].unique().tolist()



        counter = 0

        for idx, image_id in tqdm(enumerate(self.ids)):

            df_img = self.df.loc[self.df["ImageId"] == image_id]

            image_path = os.path.join(self.image_dir, image_id)

            self.image_info[counter]["image_id"] = image_id

            self.image_info[counter]["image_path"] = image_path

            self.image_info[counter]["annotations"] = df_img.iloc[0][DEFECT_COLS]

            counter += 1



    def __getitem__(self, idx):

        info = self.image_info[idx]

        img_path = self.image_info[idx]["image_path"]



        img = cv2.imread(img_path)[:, :, ::-1]

        img_df = info["annotations"]

        

        mask_cols = img_df.index[img_df.notnull()]

        mask_cols = [x for x in mask_cols if 'def_' in x]

        mask_df = img_df[mask_cols]



        labels = []

        masks_instances = []

        areas = []

        bboxes = []

        for c in mask_df.index:

            class_ = int(c.split('_')[1])

            mask = rle2mask(mask_df[c], img.shape)

            bbox, area = create_bboxes(mask)

            classes = np.full(len(bbox), class_)

            labels.append(classes)

            masks_instance = create_instance_masks(mask, bbox)

            masks_instances.append(masks_instance)

            bboxes.append(bbox)

            areas.append(area)



        labels = np.concatenate(labels)

        masks_instances = np.concatenate(masks_instances)

        areas = np.concatenate(areas)

        bboxes = np.concatenate(bboxes)

        assert len(masks_instances) == len(labels) == len(areas) == len(bboxes)



        boxes = bboxes.tolist()

        image_id = torch.tensor([idx])



        if self.transform is not None:

            augmented = self.transform(

                image=img, bboxes=boxes, mask=masks_instances, labels=labels

            )

            img = augmented["image"]

            masks_instances = augmented["mask"]

            boxes = augmented["bboxes"]

            labels = augmented["labels"]



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        masks = torch.as_tensor(masks_instances, dtype=torch.uint8)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        img = self.preproc_image(img)



        image_id = torch.tensor([idx])

        iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["masks"] = masks

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd



        return img, target



    def __len__(self):

        return len(self.image_info)



    def preproc_image(self, img):

        img = np.moveaxis(img, -1, 0).astype(np.float32)

        img = torch.from_numpy(img)

        img = img / 255.0

        return img
def get_aug(aug):

    return albu.Compose(

        aug,

    )





def train_one_epoch(

    model, optimizer, data_loader, device, epoch, print_freq, save=True, logger=None, update_freq=1, run_dir=None,

):

    model.train()

    metric_logger = MetricLogger(delimiter="  ", logger=logger)

    metric_logger.add_meter("lr", SmoothedValue(

        window_size=1, fmt="{value:.6f}"))

    header = "Epoch: [{}]".format(epoch)



    lr_scheduler = None

    if epoch == 0:

        warmup_factor = 1.0 / 1000

        warmup_iters = min(1000, len(data_loader) - 1)



        lr_scheduler = warmup_lr_scheduler(

            optimizer, warmup_iters, warmup_factor)



    step = 0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        images = list(image.cuda(non_blocking=True) for image in images)

        targets = [

            {k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets

        ]

        # images = list(image.to(device) for image in images)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # print("img0 shape: {}".format(images[0].shape))



        images = torch.stack(images, axis=0)

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes

        loss_dict_reduced = reduce_dict(loss_dict)

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())



        optimizer.zero_grad()

        losses.backward()

        if step % update_freq == 0:

            optimizer.step()



        if lr_scheduler is not None:

            lr_scheduler.step()



        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        step += 1



    if save:

        checkpoint_fname = os.path.join(

            run_dir, "model_epoch_{}.pth.tar".format(epoch))

        print("save model weights at: {}".format(checkpoint_fname))

        torch.save(model.state_dict(), checkpoint_fname)



    return
SEED = 1337

NUM_DEBUG = None

TRAIN = True

CHECKPOINT_RESUME = None



CHECKPOINT_DIR = "./"

RUN_NAME = "mrcnn_trialOne"

ROOT_IMG = "../input/severstal-steel-defect-detection/train_images/"

DEFECT_COLS = ["def_{}".format(x) for x in range(1, 5)]



NUM_EPOCHS = 5

UPDATE_FREQ = 1

NUM_CLASSES = len(DEFECT_COLS) + 1

PARALLEL = False

DEVICE = "cuda"



BATCH_SIZE = 12

HEIGHT = 256

WIDTH = 1600

HEIGHT_VAL = 256

WIDTH_VAL = 1600
df_path = "../input/severstal-steel-defect-detection/train.csv"

train_df = pd.read_csv(df_path)

train_df = train_df[train_df["EncodedPixels"].notnull()]



# some preprocessing

# https://www.kaggle.com/amanooo/defect-detection-starter-u-net

train_df["ImageId"], train_df["ClassId"] = zip(

    *train_df["ImageId_ClassId"].str.split("_")

)

train_df["ClassId"] = train_df["ClassId"].astype(int)

train_df = train_df.pivot(index="ImageId", columns="ClassId", values="EncodedPixels")

train_df["defects"] = train_df.count(axis=1)

train_df = train_df.reset_index()

COLNAMES = ["ImageId"] + DEFECT_COLS + ["defects"]

train_df.columns = COLNAMES



if NUM_DEBUG is not None:

    train_df = train_df.iloc[:NUM_DEBUG, :]



tr_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df["defects"], random_state=SEED)

print("train df shape: {}".format(tr_df.shape))

print("valid df shape: {}".format(val_df.shape))

tr_df.head()
train_aug = get_aug(

    [

        albu.RandomBrightnessContrast(0.2),

        albu.RandomGamma((90, 110)),

        albu.HorizontalFlip(p=0.5),

        albu.RandomScale((0.15, 0.35), p=1),

        albu.PadIfNeeded(min_height=HEIGHT, min_width=WIDTH, border_mode=0),

        # RandomCrop(HEIGHT, WIDTH),

        albu.RandomSizedBBoxSafeCrop(HEIGHT, WIDTH),

    ]

)

valid_aug = get_aug([albu.Resize(HEIGHT, WIDTH)])





# TODO: integrate augmentation

train_dataset = SteelMrcnnDataset(ROOT_IMG, tr_df, transform=None)

train_loader = torch.utils.data.DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=6,

    collate_fn=lambda x: tuple(zip(*x)),

    pin_memory=True,

)



valid_dataset = SteelMrcnnDataset(ROOT_IMG, val_df, transform=None)

valid_loader = torch.utils.data.DataLoader(

    valid_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=6,

    collate_fn=lambda x: tuple(zip(*x)),

    pin_memory=True,

)
rid = np.random.randint(0, len(train_dataset))

img_, target_ = train_dataset.__getitem__(rid)

boxes_ = target_['boxes'].cpu().detach().numpy()

fig, ax = plt.subplots(2, 1, figsize=(24, 10))

img = img_.cpu().detach().numpy() * 255

img = np.moveaxis(img, 0, -1).astype(np.uint8)

ax[0].imshow(img)

for i in range(len(boxes_)):

    ax[0].imshow(target_['masks'][i].numpy(), alpha=0.3, cmap='gray')

    ax[1].imshow(target_['masks'][i].numpy(), alpha=0.5, cmap='gray')

    # ax[0].plot(boxes_[i][0], boxes_[i][1], 'ro')

    # ax[0].plot(boxes_[i][2], boxes_[i][3], 'ro')

plt.show()
hidden_layer = 256

model_ft = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

in_features = model_ft.roi_heads.box_predictor.cls_score.in_features

in_features_mask = model_ft.roi_heads.mask_predictor.conv5_mask.in_channels

model_ft.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

model_ft.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)





if PARALLEL:

    print('GPU-parallel training')

    model_ft = torch.nn.DataParallel(model_ft).cuda()

else:

    print('single-GPU training')

    model_ft.to(DEVICE)



for param in model_ft.parameters():

    param.requires_grad = True

params = [p for p in model_ft.parameters() if p.requires_grad]





optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)



run_dir = os.path.join(CHECKPOINT_DIR, RUN_NAME)

print("run directory: {}".format(run_dir))

if not os.path.isdir(run_dir):

    os.mkdir(run_dir)

logger = RunLogger(os.path.join(run_dir, "run_stats.log"))

logger.initialize()
RUN_DIR = os.path.join(CHECKPOINT_DIR, RUN_NAME)





# training

if TRAIN:

    print("running training...")

    logger.log("running training...")

    if CHECKPOINT_RESUME is not None:

        model_ft.load_state_dict(torch.load(CHECKPOINT_RESUME))

    for epoch in range(NUM_EPOCHS):

        train_one_epoch(

            model_ft,

            optimizer,

            train_loader,

            DEVICE,

            epoch,

            print_freq=100,

            logger=logger,

            update_freq=UPDATE_FREQ,

            run_dir=RUN_DIR,

        )

        lr_scheduler.step()
TEST = True



checkpoints = sorted(glob.glob(run_dir + "/*.pth.tar"))[-1:]

print("checkpoints for evaluation:")

pprint(checkpoints)





if TEST:

    print("running test...")

    logger.log("running test...")

    for param in model_ft.parameters():

        param.requires_grad = False

    model_ft.eval()

    eval_hist = []

    c = checkpoints[-1]

    print("loading: {}".format(c))

    logger.log("\nloading: {}".format(c))

    model_ft.load_state_dict(torch.load(c))
THRESHOLD_CONF = 0.5





for i in range(1):

    rid = np.random.randint(0, len(valid_dataset))

    img_, target_ = valid_dataset.__getitem__(rid)

    img = (img_ * 255).cpu().detach().numpy().astype(int)

    img = np.moveaxis(img, 0, -1)

    true_masks = target_["masks"].cpu().detach().numpy()

    fig, ax = plt.subplots(4, 1, figsize=(30, 12))

    ax[0].imshow(img)



    val_pred = model_ft(torch.unsqueeze(img_, 0).cuda())[0]

    pred_boxes = val_pred['boxes'].cpu().detach().numpy()

    pred_labels = val_pred['labels'].cpu().detach().numpy()

    pred_scores = val_pred['scores'].cpu().detach().numpy()

    pred_masks = val_pred['masks'].cpu().detach().numpy()[:, 0]

    

    pred_valid_idx = pred_scores > THRESHOLD_CONF

    pred_boxes = pred_boxes[pred_valid_idx]

    pred_labels = pred_labels[pred_valid_idx]

    pred_scores = pred_scores[pred_valid_idx]

    pred_masks = pred_masks[pred_valid_idx]

    pred_mask_single = np.sum(pred_masks, axis=0)

    

    for i in range(len(true_masks)):

        ax[1].imshow(true_masks[i], alpha=0.5, cmap='gray')

    for i in range(len(pred_masks)):

        ax[2].imshow(pred_masks[i] > THRESHOLD_CONF, alpha=0.6, cmap='gray')

    ax[3].imshow(pred_mask_single > THRESHOLD_CONF, cmap='gray')

        

    ax[0].set_title("IMAGE:")

    ax[1].set_title("TRUE MASK:")

    ax[2].set_title("PREDICTED MASKS:")

    ax[3].set_title("PREDICTED SINGLE MASK:")



    plt.show()
pred_mask_single = np.sum(pred_masks, axis=0)