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
from utils import *

"""
output_dir = "pretrained_" + model_name + "_" + str(current_time)
os.mkdir(output_dir)

cfg.OUTPUT_DIR = output_dir

"""


model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(model))


cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)


val_seq = ["01", "03", "04"]

out_path = "cam_preds"
os.makedirs(out_path, exist_ok=True)

for seq in val_seq:
    print(seq)
    split_train_test(seq)
    
    cfg.MODEL.WEIGHTS = "finetuned_fasterrcnn_seq{}/model_final.pth".format(seq)
    

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
        
        out_file = os.path.join(out_path, "S" + seq + "_" + cam + "_" + "pred_boxes.pkl")
        
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
                                      "xtl": box_list[0],
                                      "ytl": box_list[1],
                                      "xbr": box_list[2],
                                      "ybr": box_list[3],
                                      "conf": instances["instances"].scores[i].item()          
                                        })
        
        with open(out_file, "wb") as f:
            pickle.dump(pred_boxes, f)  