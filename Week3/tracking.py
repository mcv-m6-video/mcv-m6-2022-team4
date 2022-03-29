import pandas as pd
from read_files import parse_aicity_rects, save_aicity_rects, get_annotations
from utils import bb_dict_to_txt
import pickle
from metrics import get_iou

tracked_object_dic = {}


# +
class tracked_object:
    TRACKS_COUNTER = 0
    def __init__(self, idd, bbox, tracker_life = 5):
        self.id = idd
        self.bbox = bbox
        self.tracker_life = tracker_life
        self.not_live = 0
        self._add_track()

    def _add_track(self):
        tracked_object_dic[self.id] = self
        return


def adj_track(det_bbox, tracked_object_dic):
      
    best_iou = 0
    idx = -1

    for id_t in tracked_object_dic:

        iou = get_iou(tracked_object_dic[id_t].bbox, det_bbox)    
        if iou > best_iou:
            best_iou = iou
            idx = id_t

    if int(idx) != -1 and best_iou > 0.3:
        tracked_object_dic[idx].tracker_life = 5
        tracked_object_dic[idx].bbox = det_bbox
    else:
        idx = str(tracked_object.TRACKS_COUNTER)
        tracked_object(idx, det_bbox)
        tracked_object.TRACKS_COUNTER+=1
    return idx


def decrease_memory(tracked_object_dic):
    deleting_list = []
    for idx in tracked_object_dic:
        if tracked_object_dic[idx].tracker_life > 0:
            tracked_object_dic[idx].tracker_life -= 1
        else:
            deleting_list.append(idx)
        
    for idx in deleting_list:
        del tracked_object_dic[idx]


# -

def main():
    
    with open("fine_pred_boxes.pkl", "rb") as f:
        fine_boxes = pickle.load(f)
        
    bb_dict_to_txt("faster_rcnn.txt", fine_boxes)
    ret = parse_aicity_rects("faster_rcnn.txt")
    
    thres = 0.5

    order = sorted(ret, key=lambda x: int(x))
    for f in order:
        for det in ret[f]:
            if det['conf'] > thres:
                det['id'] = adj_track(det['bbox'], tracked_object_dic)

        decrease_memory(tracked_object_dic)
    save_aicity_rects("det_iou_track.txt", ret)
    
    
    """
    _, frame_annots_scale = get_annotations(4)
        
    bb_dict_to_txt("gt.txt", frame_annots_scale)
    ret = parse_aicity_rects("gt.txt")
    
    thres = 0.75

    order = sorted(ret, key=lambda x: int(x))
    for f in order:
        for det in ret[f]:
            if det['conf'] > thres:
                det['id'] = adj_track(det['bbox'], tracked_object_dic)

        decrease_memory(tracked_object_dic)
    save_aicity_rects("gt_iou_track.txt", ret)
    """

if __name__ == '__main__':
    main()
