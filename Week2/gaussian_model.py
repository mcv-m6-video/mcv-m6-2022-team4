import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")

color_spaces = {
    "GRAY": cv2.COLOR_BGR2GRAY,
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "YCRCB": cv2.COLOR_BGR2YCrCb,
    "LAB": cv2.COLOR_BGR2LAB
    }


def get_boxes(im, img_size):
    
    ROI_PATH = "../AICity_data/train/S03/c010/roi.jpg"
    roi = cv2.imread(ROI_PATH, 0)/255
    roi = cv2.resize(roi, img_size)
    
    if len(im.shape) != 2:
        im = im.mean(axis=2).astype(np.uint8)
        
    im = im * roi.astype(np.uint8)
    
    im = cv2.morphologyEx(im, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    im = cv2.morphologyEx(im, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    im = cv2.morphologyEx(im, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    im = cv2.morphologyEx(im, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    
    det_recs = []

    contours, _ = cv2.findContours(im.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    im_area = im.shape[0]*im.shape[1]
    
    for i, cnt in enumerate(contours):
        
        x, y, w, h = cv2.boundingRect(cnt)

        area = h*w
        area_pct = 100 * area / im_area

        ar = h/w

        if w < 20 or h < 20 or ar > 2 or ar < 0.25 or area_pct < 0.1 or area_pct > 25: 
            continue
        else:
            det_recs.append([x, y, x+w, y+h])


    return im, det_recs


def get_bg(clr_spc, img_size):
    
    with open("bg_imgs.txt") as f:
        
        bg_imgs = f.readlines()
        
        if clr_spc == "GRAY": 
            bg_arr = np.zeros((len(bg_imgs), img_size[1], img_size[0]))
        else:
            bg_arr = np.zeros((len(bg_imgs), img_size[1], img_size[0], 3))
        
        for i, img_name in enumerate(bg_imgs):

            img = cv2.cvtColor(cv2.imread(img_name.strip()), color_spaces[clr_spc])
            img = cv2.resize(img, img_size)
            bg_arr[i] = img
            
    return bg_arr.mean(axis=0), bg_arr.std(axis=0)


def get_fg_boxes_gaussian(clr_spc, img_size, alpha, bg_mean, bg_std):
    
    with open("fg_imgs.txt") as f:
        
        imgs = []
        
        boxes = {}
        
        fg_imgs = f.readlines()

        for i, img_name in enumerate(fg_imgs):
            
            img = cv2.cvtColor(cv2.imread(img_name.strip()), color_spaces[clr_spc])
            img = cv2.resize(img, img_size)         
            
            fg_img = (abs(img - bg_mean) >= alpha * (bg_std + 2)).astype(np.uint8) * 255
            
            if len(fg_img.shape) != 2:
                fg_img = fg_img.mean(axis=2).astype(np.uint8)
                
            imgs.append(fg_img)
            
            fg_img, det_boxes = get_boxes(fg_img, img_size)
            
            box_list = []
            for box in det_boxes:
                
                box_list.append({
                    "xtl": box[0],
                    "ytl": box[1],
                    "xbr": box[2],
                    "ybr": box[3] 
                })
            
            boxes[img_name[7:-5]] = box_list
            
    return imgs, boxes


def get_fg_boxes_adptv_gaussian(clr_spc, img_size, alpha, p, bg_mean, bg_std):
    
    with open("fg_imgs.txt") as f:
        
        adptv_bg_mean = bg_mean
        adptv_bg_std = bg_std
        
        imgs = []
        
        boxes = {}
        
        fg_imgs = f.readlines()

        for i, img_name in enumerate(fg_imgs):
            
            img = cv2.cvtColor(cv2.imread(img_name.strip()), color_spaces[clr_spc])
            img = cv2.resize(img, img_size)         
            
            bm = (abs(img - adptv_bg_mean) < alpha * (adptv_bg_std + 2))
            adptv_bg_mean[bm] = (p * img[bm] + (1 - p) * adptv_bg_mean[bm])
            aux = (img - adptv_bg_mean) 
            adptv_bg_std[bm] = np.sqrt(p * aux[bm] * aux[bm] + (1 - p) * (adptv_bg_std[bm] * adptv_bg_std[bm]))

            fg_img = (abs(img - adptv_bg_mean) >= alpha * (adptv_bg_std + 2)).astype(np.uint8) * 255
            
            if len(fg_img.shape) != 2:
                fg_img = fg_img.mean(axis=2).astype(np.uint8)
            imgs.append(fg_img)
            
            fg_img, det_boxes = get_boxes(fg_img, img_size)
            
            box_list = []
            for box in det_boxes:
                
                box_list.append({
                    "xtl": box[0],
                    "ytl": box[1],
                    "xbr": box[2],
                    "ybr": box[3] 
                })
            
            boxes[img_name[7:-5]] = box_list
            
    return imgs, boxes
