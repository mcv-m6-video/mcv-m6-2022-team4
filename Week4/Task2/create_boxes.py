import random
import warnings
import os
import numpy as np
import cv2
warnings.filterwarnings("ignore")

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.structures import BoxMode
import time
import pickle
import torch

"""
output_dir = "pretrained_" + model_name + "_" + str(current_time)
os.mkdir(output_dir)

cfg.OUTPUT_DIR = output_dir

"""


model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(model))


cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "../Week3/finetuned_fasterrcnn_01_17/3/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

preds = []
pred_boxes = {}

with open("train_imgs.txt") as f:
    train_imgs = f.readlines()
    
with open("test_imgs.txt") as f:
    test_imgs = f.readlines()

all_imgs = [*train_imgs, *test_imgs]
print(len(all_imgs))

for i_num, i in enumerate(all_imgs):
    
    im = cv2.imread(i.strip())
    instances = predictor(im)
    outputs = instances["instances"].to("cpu")
    #outputs.pred_classes = torch.tensor(np.where(outputs.pred_classes == 2, 0, -1))
    #outputs = outputs[outputs.pred_classes != -1]
    instances["instances"] = outputs
    
    preds.append(instances)
    print(i_num)
    img_key = i.strip().split("/")[1][:-4]
    pred_boxes[img_key] = []
    
    for i, box in enumerate(instances["instances"].pred_boxes):
        box_list = [p.tolist() for p in list(box)]
        pred_boxes[img_key].append({
                              "xtl": box_list[0],
                              "ytl": box_list[1],
                              "xbr": box_list[2],
                              "ybr": box_list[3],
                              "conf": instances["instances"].scores[i].item()          
                                })

with open("pred_boxes.pkl", "wb") as f:
    pickle.dump(pred_boxes, f)  