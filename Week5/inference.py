from utils import *
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import warnings
import os
import numpy as np
warnings.filterwarnings("ignore")

from LossEvalHook import *
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper, build_detection_test_loader
from detectron2.structures import BoxMode
import time
import pickle
import torch


def get_aicity_data(mode):

    dataset_dicts = []

    with open("{}_imgs.txt".format(mode), "r") as f:
        imgs = f.readlines()
        
    seq_cams = set([img.strip().split("/")[1][:8] for img in imgs])
    
    gts = {}
    img_sizes = {}
    for seq_cam in seq_cams:
        
        img_sizes[seq_cam] = cv2.imread(os.path.join("frames", seq_cam + "_0.jpg")).shape
        gt_path = os.path.join(data_path, seq_cam.split("_")[0], seq_cam.split("_")[1], "gt", "gt.txt")
        gts[seq_cam] = parse_aicity_rects(gt_path)
    
    for img in imgs:

        label = {}
        label["file_name"] = img.strip()
        label["image_id"] = img.strip().split("/")[1][:-4]
        
        img_size = img_sizes[img.strip().split("/")[1][:8]]
        label["height"] = img_size[0]
        label["width"] = img_size[1]
        
        img_seq_cam = label["image_id"][:8]
        img_frame_id = label["image_id"][9:]
        
        boxes = []        
        if img_frame_id not in gts[img_seq_cam].keys():

            boxes.append({
                          "bbox": [0, 0, 0, 0],
                          "bbox_mode" : BoxMode.XYXY_ABS,
                          "category_id": 0
            })
            
        
        else:
            for box in gts[img_seq_cam][img_frame_id]:
            
                xtl, ytl, xbr, ybr = box["bbox"]
                boxes.append({
                                "bbox": [xtl, ytl, xbr, ybr],
                                "bbox_mode" : BoxMode.XYXY_ABS,
                                "category_id": 0
                      })
        label["annotations"] = boxes
        dataset_dicts.append(label)
            
    return dataset_dicts
    

data_path = "/export/home/group06/m6/data/train/"
val_seqs = ["01", "03", "04"]

out_path = "cam_preds"
os.makedirs(out_path, exist_ok=True)


model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(model))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

for val_seq in val_seqs:

    split_train_test(val_seq)
    print(val_seq)
    
    cfg.MODEL.WEIGHTS = "finetuned_fasterrcnn_{}/model_final.pth".format(val_seq)
    predictor = DefaultPredictor(cfg)
    

    with open("test_imgs.txt") as f:
        test_imgs = f.readlines()
    
    cams = {}
    
    for img in test_imgs:
        cam = img.split("/")[1].split("_")[1]
        if cam not in cams.keys():
            cams[cam] = []
        else:
            cams[cam].append(img)
            
    for cam in cams.keys():
    
        print(cam)
        print(len(cams[cam]))
        
        out_file = os.path.join(out_path, "S" + val_seq + "_" + cam + "_" + "pred_boxes.pkl")
        
        preds = []
        pred_boxes = {}
    
        for i_num, i in enumerate(cams[cam]):
            
            im = cv2.imread(i.strip())
            instances = predictor(im)
            outputs = instances["instances"].to("cpu")
            #outputs.pred_classes = torch.tensor(np.where(outputs.pred_classes == 2, 0, -1))
            #outputs = outputs[outputs.pred_classes != -1]
            instances["instances"] = outputs
            
            preds.append(instances)

            img_key = i.strip().split("/")[1][:-4]
            pred_boxes[img_key] = []
            
            for i, box in enumerate(instances["instances"].pred_boxes):
                box_list = [p.tolist() for p in list(box)]
                pred_boxes[img_key].append({
                                      "bbox": [box_list[0], box_list[1], box_list[2], box_list[3]],
                                      "conf": instances["instances"].scores[i].item(),
                                      "id": "-1"   
                                        })
        
        with open(out_file, "wb") as f:
            pickle.dump(pred_boxes, f)  
