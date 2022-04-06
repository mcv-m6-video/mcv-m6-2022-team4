import cv2
from PIL import Image
import os
import shutil
from collections import defaultdict
import numpy as np
import random
import pandas as pd
import colorsys
from pathlib import Path
from tqdm import tqdm


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


def get_frames_from_video(video_path: str,
                          start_frame: int = 0,
                          end_frame: int = np.inf):

    if not Path(video_path).exists:
        raise FileNotFoundError(f'Video path not found: {video_path}.')

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_frame < 0:
        raise ValueError(f"Start frame ({start_frame}) should be greater than 0.")
    if end_frame == np.inf:
        end_frame = frame_count
    elif end_frame > frame_count:
        raise ValueError(f"End frame ({end_frame}) is greater than {frame_count} which is the number of video frames.")

    for frame_idx in tqdm(range(start_frame, end_frame, 1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        has_frames, frame = cap.read()

        if has_frames:
            yield (frame_idx + 1, frame)



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


# +
def read_flow(path:str):
    """
    Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    contains the u-component, the second channel the v-component and the third
    channel denotes if a valid ground truth optical flow value exists for that
    pixel (1 if true, 0 otherwise)
    """
    # cv2 flips the order of reading channels
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)
    
    # valid channel
    valid = img[:,:,0]
    
    # get flow vectors
    u_flow = (img[:,:,2] - 2**15)/64
    v_flow = (img[:,:,1] - 2**15)/64
    
    # remove invalid flow values
    u_flow[valid == 0] = 0
    v_flow[valid == 0] = 0
    
    # return image in correct order
    return np.dstack((u_flow, v_flow, valid))


def plot_flow(img):
    """
    plot u and v flows along with valid pixels
    """
    fig, axes = plt.subplots(1,3, figsize=(16,8))
    images = [img[:,:,0], img[:,:,1], img[:,:,2]]
    titles = ['u_flow','v_flow','valid']
    
    for ax,image,title in zip(axes.flatten(), images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
    plt.tight_layout()
    #plt.savefig("figures_german/affinity.pdf")
    plt.show()


def get_metrics(gt_flow:np.ndarray, pred_flow:np.ndarray, mask:np.ndarray=None, th:int=3):
    """
    Calculate metrics from ground truth and predicted optical flow.
    The mask is usually the third channel of gt_flow
    
    Arguments:
        gt_flow,pred_flow are (H,W,3)
        th: motion_vector error greater than threshold is an erroneous pixel
        
    Returns:
     1. Mean Square Error in Non-occluded areas
     2. Percentage of Erroneous Pixels in Non-occluded areas
    """
    if mask is None:
        mask = gt_flow[:,:,2]  
    
    error = np.sqrt(np.sum((gt_flow[:,:,:2] - pred_flow[:,:,:2])**2, axis=-1))    
    msen = np.mean(error[mask != 0])
    pepn = 100 * np.sum(error[mask != 0] > th) / (mask != 0).sum()
    return msen, pepn


def show_field(flow, gray, step=30, scale=0.5):
    
    gray = np.copy(gray)
    plt.figure(figsize=(16,8))
    plt.imshow(gray, cmap='gray')
    
    U = flow[:, :, 0]
    V = flow[:, :, 1]
    H = np.hypot(U, V)

    (h, w) = flow.shape[0:2]
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    x = x[::step, ::step]
    y = y[::step, ::step]
    U = U[::step, ::step]
    V = V[::step, ::step]
    H = H[::step, ::step]

    plt.quiver(x, y, U, V, H, scale_units='xy', angles='xy', scale=scale)
    
    plt.axis('off')
    plt.show()
