import numpy as np


def iou_single_boxes(box1, box2):

    # determine intersection rectangle coordinates
    left = max(box1[0], box2[0])
    top = max(box1[1], box2[1])
    right = min(box1[2], box2[2])
    bottom = min(box1[3], box2[3])

    # check non-overlapping rectangle
    if right < left or bottom < top:
        return 0

    # compute intersection area
    intersection_area = (right - left) * (bottom - top)

    # compute area of bbs
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou


def get_opflow_metrics(GT_u, GT_v, u, v, mask, th):
    
    motion_vectors = np.square(GT_u - u) + np.square(GT_v - v)
    motion_vectors_dist = np.sqrt(motion_vectors)

    msen = np.mean(np.sqrt(motion_vectors)[mask != 0])
    pepn = (motion_vectors_dist[mask != 0] > th).sum() / (mask != 0).sum()  # erroneous / total
    
    return motion_vectors_dist, msen, pepn


def get_iou(bb1, bb2):
    
    """
    Parameters
    ----------
    bb1 : Tuple
        (tlx, tly, brx, bry)
    bb2 : Tuple
        K(tlx, tly, brx, bry)

    """
    
    """
    assert bb1[0] <= bb1[2]
    assert bb1[1] <= bb1[3]
    assert bb2[0] <= bb2[2]
    assert bb2[1] <= bb2[3]
    """

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])


    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    
    return round(iou, 3)


def get_iou_for_frame(gt_frame, det_frame):
    
    all_ious = []

    for gt_box in gt_frame:

        gt_frame_bboxes = [gt_box["xtl"], gt_box["ytl"], gt_box["xbr"], gt_box["ybr"]]
        gt_frame_bboxes = list(map(lambda x:float(x), gt_frame_bboxes))

        best_iou = 0
        for det_box in det_frame:
            det_frame_bboxes = [det_box["xtl"], det_box["ytl"], det_box["xbr"], det_box["ybr"]]

            det_frame_bboxes = list(map(lambda x:float(x), det_frame_bboxes))

            iou = get_iou(gt_frame_bboxes, det_frame_bboxes)

            if iou > best_iou:
                best_iou = iou

        if best_iou != 0:
            all_ious.append(best_iou)    
            
    return round(np.mean(all_ious), 3)


def temporal_iou(gt, preds):
    
    ious = []
    
    for key in gt.keys():
        for key_2 in preds.keys():
            if key==key_2:
                ious.append(get_iou_for_frame(gt[key], preds[key_2]))        
    
    return ious


def pascal_voc_ap(rec, prec):

    ap = 0.
    for t in np.arange(0., 1.01, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0
        else:
            p = np.max(prec[rec >= t])
        ap = ap + p / 11.
        
    return ap


def get_AP(gt_rects, det_rects, ovthresh=0.5):

    class_recs = {}
    npos = 0
    for frame, bboxs in gt_rects.items():
        class_recs[frame] = {
            'bbox': list(map(lambda x:[x["xtl"], x["ytl"], x["xbr"], x["ybr"]], bboxs)),
            'difficult': np.array([False]*len(bboxs)).astype(bool),
            'det': [False]*len(bboxs)
        }
        npos += len(bboxs)


    image_ids = []
    confidence = []
    BB = []

    for frame, objs in det_rects.items():
        for obj in objs:
            image_ids.append(frame)

            if "conf" not in obj.keys():
                confidence.append(np.random.uniform(size=1)[0])
            else:
                confidence.append(obj['conf']) # unkwnown
            BB.append([obj["xtl"], obj["ytl"], obj["xbr"], obj["ybr"]])

    confidence = np.array(confidence)
    BB = np.array(BB)

    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = np.array(R['bbox']).astype(float)

        if BBGT.size > 0:

            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/float(npos)

    prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = pascal_voc_ap(rec, prec)

    return ap
