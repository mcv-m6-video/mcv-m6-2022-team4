from utils import *
from read_files import *
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import warnings
import os
import numpy as np
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



scale = 4
img_size = (int(1920/scale), int(1080/scale))


# This function writes frames as images to a folder,
# so we can work on them easily. Returns True if already written
write_frames(img_size)



def get_aicity_data(mode):

    dataset_dicts = []
    _, frame_annots = get_annotations(scale)

    for frm in frame_annots.keys():

        label = {}

        label["file_name"] = os.path.join("frames", frm + ".jpg")
        label["image_id"] = frm
        label["height"] = img_size[1]
        label["width"] = img_size[0]

        boxes = []
        #size = [label["height"], label["width"]]
        for ants in frame_annots[frm]:
                
            boxes.append({
                            "bbox": [ants["xtl"], ants["ytl"], ants["xbr"], ants["ybr"]],
                            "bbox_mode" : BoxMode.XYXY_ABS,
                            "category_id": 0,
                            #"segmentation": [[]]
                            #"segmentation": [segmentation]
                  })

        label["annotations"] = boxes
        dataset_dicts.append(label)
            
    return dataset_dicts
    
    
for d in ['train']:

    DatasetCatalog.register("aicity_data", lambda d= d:get_aicity_data(d))
    MetadataCatalog.get("aicity_data").set(thing_classes=["Car"])  
    
data = DatasetCatalog.get("aicity_data")
metadata = MetadataCatalog.get("aicity_data")


model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
data = DatasetCatalog.get("aicity_data")
#vis_imgs = random.sample(data, 20)



cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(model))

cfg.DATASETS.TRAIN = ("aicity_data",)
cfg.DATASETS.VAL = ()
#cfg.DATASETS.VAL = ('kitti_val',)
#cfg.DATASETS.TEST = ('kitti_test',)
cfg.DATASETS.TEST = ()




t = time.localtime()
current_time = time.strftime("%H_%M", t)
model_name = model.split("/")[1].split("_")
model_name = model_name[0] + model_name[1]
output_dir = "pretrained_" + model_name + "_" + str(current_time)

os.mkdir(output_dir)

cfg.OUTPUT_DIR = output_dir

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "finetuned_fasterrcnn_01_17/3/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

preds = []
pred_boxes = {}

for i_num, i in enumerate(data):
    
    im = cv2.imread(i["file_name"])
    instances = predictor(im)
    outputs = instances["instances"].to("cpu")
    #outputs.pred_classes = torch.tensor(np.where(outputs.pred_classes == 2, 0, -1))
    #outputs = outputs[outputs.pred_classes != -1]
    instances["instances"] = outputs
    
    preds.append(instances)
    pred_boxes[str(i_num)] = []
    
    for i, box in enumerate(instances["instances"].pred_boxes):
        box_list = [p.tolist() for p in list(box)]
        pred_boxes[str(i_num)].append({
                              "xtl": box_list[0],
                              "ytl": box_list[1],
                              "xbr": box_list[2],
                              "ybr": box_list[3],
                              "conf": instances["instances"].scores[i].item()          
                                })


    
evaluator = COCOEvaluator("aicity_data", cfg, True, output_dir=output_dir)
evaluator.reset()
evaluator.process(data, preds)
results = evaluator.evaluate()

print(results)

with open(os.path.join(output_dir, "results.pkl"), "wb") as f:
    pickle.dump(results, f)   

with open(os.path.join(output_dir, "pred_boxes.pkl"), "wb") as f:
    pickle.dump(pred_boxes, f)     