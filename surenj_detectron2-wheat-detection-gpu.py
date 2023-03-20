




import detectron2

from detectron2.utils.logger import setup_logger

setup_logger()



import os

import json

import numpy as np

import pandas as pd

import cv2

import random

import itertools

import torch, torchvision



from detectron2 import model_zoo

from detectron2.engine import DefaultPredictor

from detectron2.config import get_cfg

from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.structures import BoxMode



from detectron2.engine import DefaultTrainer

from detectron2.config import get_cfg

from detectron2.utils.visualizer import ColorMode
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))

cfg.MODEL.WEIGHTS = "/kaggle/input/detectron2/model_0019999.pth"

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

predictor = DefaultPredictor(cfg)
test_image_dir = "/kaggle/input/global-wheat-detection/test"

test_images = os.listdir(test_image_dir)

results = []

for image in test_images:

    image_id = image[:image.find(".")]



    img = cv2.imread(os.path.join(test_image_dir, image))

    outputs = predictor(img)

    out = outputs["instances"].to("cpu")



    scores = out.get_fields()["scores"].numpy()

    scores = scores.tolist()

    scores = [round(scores, 5) for scores in scores]



    boxes = out.get_fields()['pred_boxes'].tensor.numpy().astype(int)

    boxes[:,2] = boxes[:, 2] - boxes[:, 0]

    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

    boxes = boxes.tolist()



    for i in range(len(scores)):

        boxes[i].insert(0, scores[i])

  

    prediction = []

    for boxes in boxes:

        prediction.extend(boxes)

    prediction = [str(prediction) for prediction in prediction]



    PredictionString = ""

    for i in prediction:

        PredictionString += f"{i} "

    PredictionString = PredictionString.rstrip()



    result = {"image_id":image_id, "PredictionString":PredictionString}

    results.append(result)
results[0]
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])

test_df
test_df.to_csv('submission.csv', index=False)