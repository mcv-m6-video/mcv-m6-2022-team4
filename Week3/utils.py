import cv2
from PIL import Image
import os
import shutil
from collections import defaultdict
import numpy as np
import random
import pandas as pd
import colorsys


# Found this code online from stackoverflow
def etree_to_dict(t):
    
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k:v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def get_boxes_in_frame(frame_id, frame_annots, det_annots, resize=(384, 256), gt=True):
    
    frame = cv2.imread(os.path.join("frames", str(frame_id) + ".jpg"))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, resize)
    
    if gt:
        for car in frame_annots[str(frame_id)]:

            coords = [round(float(car["xtl"])),
                      round(float(car["ytl"])),
                      round(float(car["xbr"])), 
                      round(float(car["ybr"]))]

            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color=(0,255,0), thickness=3)
            
        
    for det in det_annots[str(frame_id)]:

        coords = [round(float(det["xtl"])),
                  round(float(det["ytl"])),
                  round(float(det["xbr"])), 
                  round(float(det["ybr"]))]

        cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color=(255,0,0), thickness=2)        
        
    
        
    return frame


def get_car_bboxes(car_id, car_annots):

    frames_list = []
    car_index = car_id

    car = car_annots[car_index]["box"]
    
    for frame in car:
        
        frame_id = frame["@frame"]

        coords = [round(float(frame["@xtl"])),
                  round(float(frame["@ytl"])),
                  round(float(frame["@xbr"])), 
                  round(float(frame["@ybr"]))]

        frame = cv2.imread(os.path.join("frames", frame_id + ".jpg"))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color=(255,0,0), thickness=2)
        
        frames_list.append(cv2.resize(frame, (720, 540)))
        
    return frames_list


def get_video_of_car(car_id):
    
    car = get_car_bboxes(car_id)
    
    out = cv2.VideoWriter(car_id + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, (720, 540))
 
    for i in range(len(car)):
        out.write(cv2.cvtColor(car[i], cv2.COLOR_BGR2RGB))

    out.release()


def generate_noisy_bboxes(frame_dict, dropout_rate=0.25, gen_rate=0.25, std_size=0.2, std_ar=0.2):
    """
    Input:
        - frame dict: gt_boxes
        - tol_droput: probability of removing boxes
        - std_size: standartd deviation of gaussain controlling area
        - std_ar: standartd deviation of gaussain controlling aspectr ratio
    """
    noisy_dct = {}

    for frame, bbs in frame_dict.items():
        
        boxes = []
    
        for bb in bbs:
            if random.random() > dropout_rate:     #random dropout
                
                xtl, ytl, xbr, ybr = bb["xtl"], bb["ytl"], bb["xbr"], bb["ybr"]

                w = abs(xtl - xbr)
                h = abs(ytl - ybr)

                #position noise
                center = np.array([w/2+xtl, h/2+ytl])
                #center += np.random.normal(0, std_pos*w, 2)
                
                #size noise
                scale_f = np.random.normal(1, std_size)
                h *= scale_f
                w *= scale_f

                #Aspect ratio noise
                h *= np.random.normal(1, std_ar)
                w *= np.random.normal(1, std_ar)
                
                xtl = center[0] - w//2
                xbr = center[0] + w//2
                ytl = center[1] - h//2
                ybr = center[1] + h//2
                
                boxes.append({
                        "xtl": round(xtl, 3),
                        "ytl": round(ytl, 3),
                        "xbr": round(xbr, 3),
                        "ybr": round(ybr, 3)
                    })

                
            if gen_rate > random.random():
                
                xtl = round(np.random.uniform(0, 1700, 1)[0], 3)
                ytl = round(np.random.uniform(0, 900, 1)[0], 3)
                    
                boxes.append({
                        "xtl": xtl,
                        "ytl": ytl,
                        "xbr": round(xtl + np.random.uniform(0, 200, 1)[0], 3),
                        "ybr": round(ytl + np.random.uniform(0, 100, 1)[0], 3)
                    })
                
        noisy_dct[frame] = boxes

    return noisy_dct


def bb_dict_to_txt(filename, bb_dict):
    
    with open(filename, 'w') as fp:
        for frame in bb_dict.keys():
            for box in bb_dict[frame]:
                if "conf" in box.keys():
                    line = f'{int(frame)},-1,{box["xtl"]},{box["ytl"]},{box["xbr"]-box["xtl"]},{box["ybr"]-box["ytl"]},{box["conf"]},-1,-1,-1\n'
                else:
                    line = f'{int(frame)},-1,{box["xtl"]},{box["ytl"]},{box["xbr"]-box["xtl"]},{box["ybr"]-box["ytl"]},{1},-1,-1,-1\n'
                fp.write(line)   


def get_random_col():
    
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (b, g, r)


def resize_keep_ap(im, sf=None, width=None, height=None):
    shape = im.shape[:2][::-1]

    if sf:
        return cv2.resize(im, tuple(np.int0(sf*np.array(shape))))

    if width:
        return cv2.resize(im, (width, width*shape[1]//shape[0]))

    if height:
        return cv2.resize(im, (height*shape[0]//shape[1], height))

    return im


def get_optimal_font_scale(text, width):
    
    for scale in reversed(range(0, 60, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        if (new_width <= width):
            return scale/10
    return 1


def pretty_rects(im, objs, color_id, color, conf_thresh=0.0, tracking = False):
    
    for obj in objs:
        if float(obj["conf"]) < conf_thresh:
            continue
        if tracking:
            if str(obj['id']) not in color_id:
                color_id[str(obj['id'])] = get_random_col()
            color = color_id[str(obj['id'])]
            #color = (0, 255, 0)

        bb = obj['bbox']
        h = bb[3] - bb[1]
        w = bb[2] - bb[0]
        
        im = cv2.rectangle(im, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 2)

        text = f'{obj["id"]}'

        im = cv2.rectangle(im, 
            (int(bb[0]), int(bb[1])),
            (int(bb[0]+0.7*w), int(bb[1]+0.2*h)),
            color, -1)

        cv2.putText(im, text, (int(bb[0]), int(bb[1]+0.15*h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, get_optimal_font_scale(text, 0.4*w), (0,0,0))
    return im, color_id
