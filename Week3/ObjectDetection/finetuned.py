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

scale = 4
img_size = (int(1920/scale), int(1080/scale))

write_frames(img_size)


def get_aicity_data(mode):

    dataset_dicts = []
    _, frame_annots = get_annotations(scale)

    with open("{}_imgs.txt".format(mode), "r") as f:
        imgs = f.readlines()

    for img in imgs:

        label = {}
        label["file_name"] = img.strip()
        label["image_id"] = img.split("/")[1].strip()[:-4]
        label["height"] = img_size[1]
        label["width"] = img_size[0]
        boxes = []
        for ants in frame_annots[label["image_id"]]:
                
            boxes.append({
                            "bbox": [ants["xtl"], ants["ytl"], ants["xbr"], ants["ybr"]],
                            "bbox_mode" : BoxMode.XYXY_ABS,
                            "category_id": 0
                            #"segmentation": [[]]
                            #"segmentation": [segmentation]
                  })

        label["annotations"] = boxes
        dataset_dicts.append(label)
            
    return dataset_dicts


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.VAL[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#model = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"

model_name = model.split("/")[1].split("_")
model_name = model_name[0] + model_name[1]
t = time.localtime()
current_time = time.strftime("%H_%M", t)
output_dir = "finetuned_" + model_name + "_" + str(current_time)

os.mkdir(output_dir)

frames = os.listdir("frames")
frames = sorted(frames, key=lambda x:int(x.split(".")[0]))

cvs = []
start_index = 0
stop_index = start_index + int(len(frames)/4)
while stop_index <= len(frames):
    cvs.append((start_index, stop_index))
    start_index += int(len(frames)/4)
    stop_index += int(len(frames)/4)
    
for i in range(4):

    for d in ["train", "test"]:
    
        DatasetCatalog.register("aicity_{}_{}".format(i, d), lambda d= d:get_aicity_data(d))
        MetadataCatalog.get("aicity_{}_{}".format(i, d)).set(thing_classes=["Car"])  
    
    metadata = MetadataCatalog.get("aicity_{}_train".format(i))

    print(i)
    split_train_test_B(cvs[i][0], cvs[i][1])

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model))

    cfg.DATASETS.TRAIN = ("aicity_{}_train".format(i),)
    cfg.DATASETS.VAL = ("aicity_{}_test".format(i),)
    #cfg.DATASETS.TEST = ('kitti_test',)
    cfg.DATASETS.TEST = ()
    #cfg.TEST.EVAL_PERIOD = 100

    #cfg.INPUT.MIN_SIZE_TRAIN = 450
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    cfg.SOLVER.MAX_ITER = 500
    #cfg.SOLVER.BASE_LR = 0.005


    #model_name = model.split("/")[1].split("_")
    #model_name = model_name[0] + model_name[1]
    
    
    output_dir_iter = os.path.join(output_dir, str(i))
    os.mkdir(output_dir_iter)

    os.mkdir(os.path.join(output_dir_iter, "inference"))
    
    cfg.OUTPUT_DIR = output_dir_iter

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    val_dicts = get_aicity_data("test")


    evaluator = COCOEvaluator("aicity_{}_test".format(i), cfg, True, output_dir=output_dir_iter)
    val_loader = build_detection_test_loader(cfg, "aicity_{}_test".format(i))
    inference_results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(inference_results)

    with open(os.path.join(output_dir_iter, "inference", "metrics.pkl"), "wb") as f:
        pickle.dump(inference_results, f) 
        
        
        
"""
for d in ["train", "test"]:
    
    DatasetCatalog.register("aicity_{}".format(d), lambda d= d:get_aicity_data(d))
    MetadataCatalog.get("aicity_{}".format(d)).set(thing_classes=["Car"])  
    
metadata = MetadataCatalog.get("aicity_train")

split_train_test_A()

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(model))

cfg.DATASETS.TRAIN = ("aicity_train",)
cfg.DATASETS.VAL = ("aicity_test",)
#cfg.DATASETS.TEST = ('kitti_test',)
cfg.DATASETS.TEST = ()
#cfg.TEST.EVAL_PERIOD = 100

#cfg.INPUT.MIN_SIZE_TRAIN = 450
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
cfg.SOLVER.MAX_ITER = 500
#cfg.SOLVER.BASE_LR = 0.005


#model_name = model.split("/")[1].split("_")
#model_name = model_name[0] + model_name[1]


os.mkdir(os.path.join(output_dir, "inference"))

cfg.OUTPUT_DIR = output_dir

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
val_dicts = get_aicity_data("test")


evaluator = COCOEvaluator("aicity_test", cfg, True, output_dir=output_dir)
val_loader = build_detection_test_loader(cfg, "aicity_test")
inference_results = inference_on_dataset(trainer.model, val_loader, evaluator)
print(inference_results)

with open(os.path.join(output_dir, "inference", "metrics.pkl"), "wb") as f:
    pickle.dump(inference_results, f) 

"""
