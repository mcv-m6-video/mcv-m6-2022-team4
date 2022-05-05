import pickle
import os
import cv2
import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils import parse_aicity_rects

kal_dets = {}
for res in os.listdir("kal_dets_roi_parked"):

  if res.endswith(".txt"):
      kal_dets[res[:8]] = parse_aicity_rects(os.path.join("kal_dets_roi_parked", res))

   
all_cam_cars = {} 

for seq_cam in kal_dets.keys():

    print(seq_cam)
    file_path = "/export/home/group06/m6/data/train/{}/{}".format(seq_cam[:3], seq_cam[4:8])
    
    all_cars = {}  
    for key in kal_dets[seq_cam].keys():
      
        img = cv2.imread(os.path.join("frames", seq_cam + "_" + key + ".jpg"))
        for i, box in enumerate(kal_dets[seq_cam][key]):

            x1, y1, x2, y2 = list(map(lambda x:int(x), box["bbox"]))
            
            if img[y1:y2, x1:x2].size > 0:
                
                all_cars[key + "_" + str(i)] = img[y1:y2, x1:x2]
        
    all_cam_cars[seq_cam] = all_cars
         
#print(all_cars)          
            
with open("all_cars.pkl", "wb") as f:
    pickle.dump(all_cam_cars, f)
