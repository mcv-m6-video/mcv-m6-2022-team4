import os
import shutil
import numpy as np
from utils import etree_to_dict
import xml.etree.ElementTree as ET
import cv2


def get_det_gt():
    
    gts = {}
    path = "AICity_data/train/S03/c010/det"
    for txt in os.listdir(path):
        with open(os.path.join(path, txt), "r") as file:
            labels = [list(map(lambda x:float(x), pred)) for pred in
                      list(map(lambda x:x.strip().split(","), file.readlines()))]
              
        temp_dict = {}
        
        for label in labels:
            
            label_id = str(int(label[0]-1))
            
            if label_id not in temp_dict.keys():
                temp_dict[label_id] = []
                
            coords = {
                "xtl": label[2],
                "ytl": label[3],
                "xbr": round(label[2] + label[4]),
                "ybr": round(label[3] + label[5]),
                "conf": label[6]
            }
            
            temp_dict[label_id].append(coords)
      
        gts[txt.split(".")[0][4:]] = temp_dict
        
    return gts


def get_annotations(scale):
    
    tree = ET.parse('../ai_challenge_s03_c010-full_annotation.xml')
    root = tree.getroot()
    xml_dict = etree_to_dict(root)
    
    car_annots = [car for car in xml_dict["annotations"]["track"] if car["@label"] == "car"]  
    frame_annots = {}

    for i, box in enumerate(car_annots):
        for frame in box["box"]:

            if "@frame" in frame.keys():
                if frame["@frame"] not in frame_annots.keys():
                    frame_annots[frame["@frame"]] = []

                temp_frame = frame.copy()
                temp_frame.pop("@frame")
                temp_frame["car_id"] = i
                frame_annots[frame["@frame"]].append(temp_frame)
       
    frame_annots_bbs = {}
    for key in frame_annots:
        key_list = []
        for box in frame_annots[key]:
            temp_dict = {
                "parked": box["attribute"]["#text"],
                "xtl": float(box["@xtl"])/scale,
                "ytl": float(box["@ytl"])/scale,
                "xbr": float(box["@xbr"])/scale,
                "ybr": float(box["@ybr"])/scale
            }
            key_list.append(temp_dict)

        frame_annots_bbs[key] = key_list
        
    frame_annots_bbs = dict(sorted(frame_annots_bbs.items(), key=lambda t: int(t[0])))
    
    return car_annots, frame_annots_bbs


def write_frames():

    if os.path.exists("frames"):
        if len(os.listdir("frames")) == 2141:
            return True
        
        else:
            if os.path.exists("frames"):
                shutil.rmtree("frames")

    if not os.path.exists("frames"):
        os.mkdir("frames")

    vid = cv2.VideoCapture("../AICity_data/train/S03/c010/vdo.avi")

    success, img = vid.read()
    counter = 0
    cv2.imwrite(os.path.join("frames", "0.jpg"), img)

    while success:
        _, img = vid.read()
        counter += 1
        if img is None:
            break
        else:
            cv2.imwrite(os.path.join("frames", str(counter) + ".jpg"), img)


def split_bg_fg_frames(pctg=0.25):

    all_imgs = os.listdir("frames")
    all_imgs = sorted(all_imgs, key=lambda x:int(x.split(".")[0]))
    #random.shuffle(all_imgs)
    frame_num = int(len(all_imgs)*pctg)
    bg_imgs = all_imgs[:frame_num]
    fg_imgs = all_imgs[frame_num:]        

    with open("bg_imgs.txt", "w") as f:
        for img in bg_imgs:
            f.write("%s\n" % os.path.join("frames", img))

    with open("fg_imgs.txt", "w") as f:
        for img in fg_imgs:
            f.write("%s\n" % os.path.join("frames", img))
