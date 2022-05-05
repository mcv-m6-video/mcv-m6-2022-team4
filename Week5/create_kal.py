import pandas as pd
import os
import cv2
import pickle
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sort import Sort
import motmetrics as mm

data_path = "/export/home/group06/m6/data/train/"


os.makedirs("kal_dets_roi_parked", exist_ok=True)

roi_boxes = True
all_res = {}


for seq in os.listdir(data_path):
    
    if seq in ["S01", "S03", "S04"]:
        for cam in os.listdir(os.path.join(data_path, seq)):
            
            print(seq, cam)
            keys = []
            file_path = "/export/home/group06/m6/data/train/{}/{}".format(seq, cam)
            roi = cv2.imread(os.path.join(file_path, "roi.jpg"), 0)
            gt_path = os.path.join(file_path, "gt/gt.txt")

            gts = parse_aicity_rects(gt_path)
            with open(os.path.join("cam_preds", "_".join((seq, cam, "pred_boxes.pkl"))), "rb") as f:
                dets = pickle.load(f)

            dets = {key[9:]:value for key, value in dets.items()}
            dets = {k:v for k,v in sorted(dets.items(), key=lambda x:int(x[0]))}

            kal_preds = []
            kal_track = Sort(max_age=5, min_hits=3, check_parked=True, parked_age=5)

            kal_path = os.path.join("kal_dets_roi_parked", "_".join((seq, cam, "kal_iou_track.txt")))

            for frame in dets.keys():
                if len(dets[frame]) == 0:
                    continue
                
                if roi_boxes:
                    
                    img_path = os.path.join("frames", "_".join((seq, cam, str(frame))) + ".jpg")
                    img = cv2.imread(img_path)
                    
                    all_frame_boxes = []
                    for box in dets[frame]:
                        x1, y1, x2, y2 = list(map(lambda x:int(x), box["bbox"]))
                        box_frame = img.copy()
                        box_frame = np.zeros_like(img)
                        box_frame[y1:y2, x1:x2] = 255
                        res_img = cv2.bitwise_and(box_frame, box_frame, mask=roi)

                        if box_frame[box_frame == 255].shape == res_img[res_img == 255].shape:
                            all_frame_boxes.append(box["bbox"])
                            
                    if len(all_frame_boxes) == 0:
                        continue
                    box_arr = np.array(all_frame_boxes)
                    
                else: 
                    box_arr = np.array([[item for item in box["bbox"]] for box in dets[frame]])   

                keys.append(frame)
                pred = kal_track.update(box_arr)
                kal_preds.append(pred)

            with open(kal_path, 'w') as fp:

                min_idx = np.min([np.min(kal_pred[:, 4]) for kal_pred in kal_preds if kal_pred.size != 0])
                for i, frame in zip(keys, kal_preds):
                    for box in frame:
                        line = f'{int(i)+1},{int(box[-1]-min_idx)},{box[0]},{box[1]},{box[2]-box[0]},{box[3]-box[1]},{1},-1,-1,-1\n'
                        fp.write(line) 

            gt = mm.io.load_motchallenge(gt_path)
            det = mm.io.load_motchallenge(kal_path)
            acc = mm.utils.compare_to_groundtruth(gt, det)
            mh = mm.metrics.create()
            summary = mh.compute(acc, metrics=["idf1", "idp", "idr", "precision", "recall"], name='acc')
            print(summary)
            all_res[seq + "_" + cam] = summary
            summary.to_csv(os.path.join("kal_dets_roi_parked", "_".join((seq, cam, "summary")) + ".csv"), index=False)
            
            
for seq in ["S01", "S03", "S04"]:
    
    print(seq)
    seq_res = {k:v for k,v in all_res.items() if k[:3] == seq}
    
    idf1 = np.mean([val["idf1"] for val in seq_res.values()])
    print("Avg idf1:", idf1)
    idp = np.mean([val["idp"] for val in seq_res.values()])
    print("Avg idp:", idp)
    idr = np.mean([val["idr"] for val in seq_res.values()])
    print("Avg idr:", idr)
    precision = np.mean([val["precision"] for val in seq_res.values()])
    print("Avg precision:", precision)
    recall = np.mean([val["recall"] for val in seq_res.values()])
    print("Avg recall:", recall)