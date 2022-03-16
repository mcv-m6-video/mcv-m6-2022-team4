#!/usr/bin/env python
# coding: utf-8

from metrics import *
from utils import *
from read_files import *
import pandas as pd
import warnings
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")



# This function writes frames as images to a folder,
# so we can work on them easily. Return True if already written
write_frames()


# Get detections for 3 detectors
det_gts = get_det_gt()

# Get gound truths
car_annots, frame_annots = get_annotations()


# Function to do grid-search on noise parameters
def grid_search_noise_params():

    df_res = pd.DataFrame(columns=["Dropout_Rate", "Generation_Rate", "Std_Size_Rate", "Std_AR_Rate", "IOU", "mAP"])

    sel_dr = np.arange(0, 0.31, 0.1)
    sel_gr = np.arange(0, 0.31, 0.1)
    sel_ss = np.arange(0, 0.16, 0.05)
    sel_sa = np.arange(0, 0.16, 0.05)

    combs = itertools.product(sel_dr, sel_gr, sel_ss, sel_sa)

    for i, comb in enumerate(combs):

        print(i)
        noisy_Gt = generate_noisy_bboxes(frame_annots, dropout_rate=comb[0], 
                                         gen_rate=comb[1], std_size=comb[2], std_ar=comb[3])

        ap = 0

        for i in range(5):
            ap += get_AP(frame_annots, noisy_Gt)

        mAP = round(ap/5, 4)

        ious = []

        for key in frame_annots.keys():
            for key_2 in noisy_Gt.keys():
                if key==key_2:
                    ious.append(get_iou_for_frame(frame_annots[key], noisy_Gt[key_2]))

        df_res = df_res.append({
                                "Dropout_Rate": comb[0],
                                "Generation_Rate": comb[1],
                                "Std_Size_Rate": comb[2],
                                "Std_AR_Rate": comb[3],
                                "IOU": round(np.mean(ious), 4),
                                "mAP": mAP

        }, ignore_index=True)

    df_res.to_csv("noise_tune.csv", index=False)


# Results of the grid-search
df_res = pd.read_csv("noise_tune.csv")
df_res = df_res.dropna()



# We visualize each parameter to see its effect

df_vis = df_res[(df_res["Generation_Rate"]==0) & (df_res["Std_Size_Rate"]==0) & (df_res["Std_AR_Rate"]==0)]                [["Dropout_Rate", "IOU", "mAP"]]

df_vis = pd.melt(df_vis, id_vars='Dropout_Rate', value_name='score')

plt.figure(figsize=(10, 6))
sns.barplot(x="Dropout_Rate", hue="variable", y="score", data=df_vis)
plt.ylabel(ylabel="")
legend = plt.legend(fontsize=14)
#legend.title("Whatever else")
plt.xlabel(xlabel="Dropout Rate", fontsize=14)
plt.savefig("dropout.png")
plt.show()


df_vis = df_res[(df_res["Dropout_Rate"]==0) & (df_res["Std_Size_Rate"]==0) & (df_res["Std_AR_Rate"]==0)]                [["Generation_Rate", "IOU", "mAP"]]

df_vis = pd.melt(df_vis, id_vars='Generation_Rate', value_name='score')

plt.figure(figsize=(10, 6))
sns.barplot(x="Generation_Rate", hue="variable", y="score", data=df_vis)
plt.ylabel(ylabel="")
legend = plt.legend(fontsize=14)
#legend.title("Whatever else")
plt.xlabel(xlabel="Generation Rate", fontsize=14)
plt.savefig("generation.png")
plt.show()


# In[ ]:


df_vis = df_res[(df_res["Dropout_Rate"]==0) & (df_res["Generation_Rate"]==0) & (df_res["Std_AR_Rate"]==0)]                [["Std_Size_Rate", "IOU", "mAP"]]

df_vis = pd.melt(df_vis, id_vars='Std_Size_Rate', value_name='score')

plt.figure(figsize=(10, 6))
sns.barplot(x="Std_Size_Rate", hue="variable", y="score", data=df_vis)
plt.ylabel(ylabel="")
legend = plt.legend(fontsize=14)
#legend.title("Whatever else")
plt.xlabel(xlabel="Scale Change Rate", fontsize=14)
plt.savefig("scale.png")
plt.show()


df_vis = df_res[(df_res["Dropout_Rate"]==0) & (df_res["Generation_Rate"]==0) & (df_res["Std_Size_Rate"]==0)]                [["Std_AR_Rate", "IOU", "mAP"]]

df_vis = pd.melt(df_vis, id_vars='Std_AR_Rate', value_name='score')

plt.figure(figsize=(10, 6))
sns.barplot(x="Std_AR_Rate", hue="variable", y="score", data=df_vis)
plt.ylabel(ylabel="")
legend = plt.legend(fontsize=14)
#legend.title("Whatever else")
plt.xlabel(xlabel="Aspect Ratio Change Rate", fontsize=14)
plt.savefig("ar.png")
plt.show()


# Get temporal IOU for noisy boxes
noisy_Gt = generate_noisy_bboxes(frame_annots, dropout_rate=0.2, gen_rate=0.2, std_size=0.2, std_ar=0.2)

iou_res = temporal_iou(frame_annots, noisy_Gt)

plt.figure(figsize=(12, 8))
title = "Mean IOU for {}: {} ".format("Noisy", str(round(np.mean(iou_res), 3)))
plt.title(title)

plt.plot(iou_res, color="lightblue", label="Noisy")
#plt.plot(temporal_iou(frame_annots, det_gts["mask_rcnn"]), color="green", label="mask_rcnn")

plt.xticks(np.arange(0, 2001, 500), fontsize=14)
plt.yticks(np.arange(0.25, 1.01, 0.1), fontsize=14)
plt.legend(fontsize=14)
plt.savefig("Noisy_tiou.png")


# Get temporal IOU for the detectors
det_results = {}

color = ["lightblue", "salmon", "orchid"]
for i, key in enumerate(det_gts.keys()):
    
    mAP = get_AP(frame_annots, det_gts[key])
    
    iou_res = temporal_iou(frame_annots, det_gts[key])
    
    det_results[key] = {
        "mAP": mAP,
        "mIOU": round(np.mean(iou_res), 3),
        "t_IOU": iou_res
    }
    plt.figure(figsize=(12,8))
    title = "Mean IOU for {}: {} ".format(key, str(round(np.mean(iou_res), 3)))
    plt.title(title, fontsize=20)
    
    plt.plot(iou_res, label=key, color=color[i])
    plt.xticks(np.arange(0, 2001, 500), fontsize=14)
    plt.yticks(np.arange(0.25, 1.01, 0.1), fontsize=14)
    plt.savefig(key + "_tiou.png", edgecolor='none')
    
    

plt.figure(figsize=(15,8))
plt.plot(temporal_iou(frame_annots, det_gts["mask_rcnn"]), color=color[0], label="mask_rcnn")
plt.plot(temporal_iou(frame_annots, det_gts["ssd512"]), color=color[1], label="ssd512")
plt.plot(temporal_iou(frame_annots, det_gts["yolo3"]), color=color[2], label="yolo3")

plt.xticks(np.arange(0, 2001, 500), fontsize=14)
#plt.plot(temporal_iou(frame_annots, noisy_Gt), label="noisy")
plt.yticks(np.arange(0.25, 1.01, 0.1), fontsize=14)

plt.legend(fontsize=14)
plt.savefig("combined_tiou.png", edgecolor='none')


# Create gifs and videos with ground truth and predicted boxes
def create_gif(start_index, stop_index, gt, preds, gif_name, size):
    
    frames_with_boxes = []
    while start_index < stop_index:

        frame_with_boxes = get_boxes_in_frame(str(start_index), gt, preds, size)

        frames_with_boxes.append(Image.fromarray(frame_with_boxes))

        start_index += 1

    frame_one = frames_with_boxes[0]
    frame_one.save(gif_name + ".gif", format="GIF", append_images=frames_with_boxes,
                   save_all=True, duration=100, loop=0)

    frames_with_boxes = []
    frame_one = []


create_gif(1800, 1900, frame_annots, noisy_Gt, "noisy", size=(576, 324))


def create_vid(gt, preds, vid_name, size):

    frames_with_boxes = []

    for frame in noisy_Gt.keys():

        frame_with_boxes = get_boxes_in_frame(frame, gt, preds, resize=size)
        frames_with_boxes.append(frame_with_boxes)

    out = cv2.VideoWriter(vid_name + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

    for i in range(len(frames_with_boxes)):
        out.write(cv2.cvtColor(frames_with_boxes[i], cv2.COLOR_BGR2RGB))

    out.release()
    frames_with_boxes = []

create_vid(frame_annots, noisy_Gt, "noisy", size=(576, 324))

