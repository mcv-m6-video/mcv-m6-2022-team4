import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import itertools
import pandas as pd
import time
from utils import parse_aicity_rects

import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import argparse
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path
import random
import datetime as dt

import motmetrics as mm
from sort import Sort


gts_path = "../data/train/"

for cam_pred in os.listdir("cam_preds"):

    cam_pred_id = cam_pred[:8]
    print(cam_pred_id)
    
    with open(os.path.join("cam_preds", cam_pred), "rb") as f:
        fine_boxes = pickle.load(f)
              
    kal_preds = []
    kal_track = Sort(max_age=5, min_hits=3)
    
    kal_path = os.path.join("cam_preds", cam_pred_id + "_" + "kal_iou_track.txt")
    for frame in fine_boxes.keys():
    
        box_arr = np.array([[item[1] for item in box.items()] for box in fine_boxes[frame]])   
        pred = kal_track.update(box_arr)
        kal_preds.append(pred)
    
    with open(kal_path, 'w') as fp:
        for i, frame in enumerate(kal_preds):
            for box in frame:
                line = f'{i+1},{int(box[-1]-1)},{box[0]},{box[1]},{box[2]-box[0]},{box[3]-box[1]},{1},-1,-1,-1\n'
                fp.write(line)   
    
    rects = parse_aicity_rects(os.path.join("cam_preds", cam_pred_id + "_" + "kal_iou_track.txt"), zero_index=0)
    
    cam_pred_id.split("_")
    cam_gt_path = os.path.join(gts_path, cam_pred_id.split("_")[0], cam_pred_id.split("_")[1], "gt", "gt.txt")
    
    gt = mm.io.load_motchallenge(cam_gt_path)
    det = mm.io.load_motchallenge(kal_path)
    acc = mm.utils.compare_to_groundtruth(gt, det)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["idf1", "idp", "idr"], name='acc')
    print(summary)
