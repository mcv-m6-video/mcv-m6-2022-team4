import pandas as pd
from read_files import parse_aicity_rects, save_aicity_rects, get_annotations
from utils import bb_dict_to_txt
import pickle
from metrics import get_iou, iou_single_boxes
from copy import deepcopy
import random

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

    if int(idx) != -1 and best_iou > 0.5:
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

class Track(object):

    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.tracking = detection if isinstance(detection, list) else [detection]
        self.color = (int(random.random() * 256),
                      int(random.random() * 256),
                      int(random.random() * 256))
        self.finished = False

    def add_detection_to_tracking(self, detection):
        self.tracking.append(detection)

    @property
    def previous_detection(self):
        return self.tracking[-1]


class MaxOverlapTracker():
    def __init__(self):
        self.track_number = 0

    def track_by_max_overlap(self, tracks, detections, optical_flow=None):
        current_detections = deepcopy(detections)
        tracks_on_frame = []

        # Check if current_detections can be matched with detections from current tracks
        for track in tracks:
            if track.finished:
                continue
            best_matched = self.match_detections(track.previous_detection, current_detections, optical_flow)
            if best_matched:
                track.add_detection_to_tracking(best_matched)
                tracks_on_frame.append(track)
                current_detections.remove(best_matched)
            else:
                track.finished = True

        # For the unkown detection create new detections
        for detection in current_detections:
            new_tracking = Track(self.track_number, detection)
            tracks.append(new_tracking)
            tracks_on_frame.append(new_tracking)
            self.track_number += 1
        return tracks, tracks_on_frame

    def match_detections(self, previous_detection, current_detections, optical_flow):
        prev_det = deepcopy(previous_detection)
        
        if optical_flow is not None:
            prev_det["xtl"] += optical_flow[int(prev_det["ytl"]), int(prev_det["xtl"]), 0]
            prev_det["ytl"] += optical_flow[int(prev_det["ytl"]), int(prev_det["xtl"]), 1]
            prev_det["xbr"] += optical_flow[int(prev_det["ybr"]), int(prev_det["xbr"]), 0]
            prev_det["ybr"] += optical_flow[int(prev_det["ybr"]), int(prev_det["xbr"]), 1]

        max_iou = 0
        prev_box = [prev_det["xtl"], prev_det["ytl"], prev_det["xbr"], prev_det["ybr"]]
        
        for detection in current_detections:
            
            det_box = [detection["xtl"], detection["ytl"], detection["xbr"], detection["ybr"]]
            iou = iou_single_boxes(prev_box, det_box)
            if iou > max_iou:
                max_iou = iou
                best_match = detection
        if max_iou > 0:
            best_match.id = prev_det.id
            return best_match
        else:
            return None
