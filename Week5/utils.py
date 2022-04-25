import os
import cv2
import pandas as pd
import numpy as np



def split_train_test(seq="03"):

    val_imgs = []
    train_imgs = []

    all_imgs = os.listdir("frames")
    for img in all_imgs:
        if img[1:3] == seq:
            val_imgs.append(img)  
        else:
            train_imgs.append(img)
    

    with open("train_imgs.txt", "w") as f:
        for img in train_imgs:
            f.write("%s\n" % os.path.join("frames", img))

    with open("test_imgs.txt", "w") as f:
        for img in val_imgs:
            f.write("%s\n" % os.path.join("frames", img))


def write_frames(data_path):

    for seq in os.listdir(data_path):
        for cam in os.listdir(os.path.join(data_path, seq)):
        
            print(seq, cam)
            vid = cv2.VideoCapture(os.path.join(data_path, seq, cam, "vdo.avi"))
        
            success, img = vid.read()
            counter = 0
            #img = cv2.resize(img, (img_size))
            
            img_pth = "_".join((seq, cam, str(counter)))
            
            cv2.imwrite(os.path.join("frames", img_pth + ".jpg"), img)
        
            while success:
                _, img = vid.read()
                counter += 1
                if img is None:
                    break
                else:
                    img_pth = "_".join((seq, cam, str(counter)))
                    #img = cv2.resize(img, (img_size))
                    cv2.imwrite(os.path.join("frames", img_pth + ".jpg"), img)
                    
def save_aicity_rects(path, det_rects):

    COL_NAMES = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']
    dic_csv = {
        'frame' : [], 'id': [], 'bb_left': [], 'bb_top': [], 'bb_width': [], 'bb_height': [], 'conf': [], 'x': [], 'y': [], 'z': []
    }
    for f in det_rects:
        for det in det_rects[f]:
            dic_csv['frame'].append(str(int(f)+1))
            dic_csv['id'].append(det['id'])
            dic_csv['bb_left'].append(det['bbox'][0])
            dic_csv['bb_top'].append(det['bbox'][1])
            dic_csv['bb_width'].append(det['bbox'][2]-det['bbox'][0])
            dic_csv['bb_height'].append(det['bbox'][3]-det['bbox'][1])
            dic_csv['conf'].append(det['conf'])
            dic_csv['x'].append(-1)
            dic_csv['y'].append(-1)
            dic_csv['z'].append(-1)

    df = pd.DataFrame(dic_csv, columns=COL_NAMES)
    df.to_csv(path, index = False, header=False)
    
    
def parse_aicity_rects(path, zero_index=1):

    cols = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z']

    ret_dict = {}
    dtf = pd.read_csv(path, delimiter=',', names=cols)

    for i, row in dtf.iterrows():
        frame_num = str(int(row.frame) - zero_index)

        if frame_num not in ret_dict:
            ret_dict[frame_num] = []

        obj = {
            'bbox': [row.bb_left, row.bb_top, row.bb_left+row.bb_width, row.bb_top+row.bb_height],
            'conf': float(row.conf),
            'id': int(row.id)
        }
        ret_dict[frame_num].append(obj)
        
    return ret_dict
    
    

def get_boxes_in_frame(frame_id, seq, cam, gt_annots, det_annots, gt=True):

    img_path = (os.path.join("frames", "_".join((seq, cam, str(frame_id))) + ".jpg"))

    frame = cv2.imread(img_path)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if gt:
        if str(frame_id) in gt_annots.keys():
            for car in gt_annots[str(frame_id)]:
                bbox = car["bbox"]
                xtl, ytl, xbr, ybr = bbox[0], bbox[1], bbox[2], bbox[3]
                #print(xtl, ytl, xbr, ybr)
    
                coords = [round(float(xtl)),
                          round(float(ytl)),
                          round(float(xbr)), 
                          round(float(ybr))]
    
                cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color=(0,255,0), thickness=2)
            
    if str(frame_id) in det_annots.keys():
        for det in det_annots[str(frame_id)]:
            bbox = det["bbox"]
            xtl, ytl, xbr, ybr = bbox[0], bbox[1], bbox[2], bbox[3]
    
            coords = [round(float(xtl)),
                          round(float(ytl)),
                          round(float(xbr)), 
                          round(float(ybr))]
    
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color=(0,0,255), thickness=2)        
           
    return frame



def create_vid(seq, cam, start_index, stop_index, gt, preds, vid_name):

    frames_with_boxes = []

    while start_index < stop_index:

        frame_with_boxes = get_boxes_in_frame(start_index, seq, cam, gt, preds)
        frames_with_boxes.append(frame_with_boxes)
        
        start_index += 1

    out = cv2.VideoWriter(vid_name + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, (frames_with_boxes[0].shape[1], frames_with_boxes[0].shape[0]))

    for i in range(len(frames_with_boxes)):
        out.write(frames_with_boxes[i])

    out.release()
    frames_with_boxes = []