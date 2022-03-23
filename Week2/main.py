#!/usr/bin/env python
# coding: utf-8

# In[1]:


from metrics import *
from utils import *
from read_files import *
from gaussian_model import *
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


color_spaces = {
    "GRAY": cv2.COLOR_BGR2GRAY,
    "RGB": cv2.COLOR_BGR2RGB,
    "HSV": cv2.COLOR_BGR2HSV,
    "YCRCB": cv2.COLOR_BGR2YCrCb,
    "LAB": cv2.COLOR_BGR2LAB
    }


clr_spc = "GRAY"
scale = 4
img_size = (int(1920/scale), int(1080/scale))


# This function writes frames as images to a folder,
# so we can work on them easily. Returns True if already written
write_frames()
_, frame_annots = get_annotations(scale)


# We are only going to use the ground truth for the parked cars
# that are in the foreground.
parked_annots = {}

for key in frame_annots.keys():
    non_park_boxes = []
    for box in frame_annots[key]:
        if box["parked"] != "true":
            non_park_boxes.append(box)
    parked_annots[key] = non_park_boxes
    
fg_annots = {key:parked_annots[key] for key in parked_annots if key in [str(i) for i in range(535, 2141)]}

# Create text files that splits the images into bg and fg
split_bg_fg_frames()

# Non adaptive gaussian with the best params
bg_mean, bg_std = get_bg(clr_spc, img_size)
fg_imgs, pred_boxes = get_fg_boxes_gaussian(clr_spc, img_size, 4, bg_mean, bg_std)

# Adaptive gaussian with the best params
bg_mean, bg_std = get_bg(clr_spc, img_size)
fg_imgs, pred_boxes = get_fg_boxes_adptv_gaussian(clr_spc, img_size, 3, 0.009, bg_mean, bg_std)


p_vals = np.arange(0.001, 0.009, 0.002)
bg_mean, bg_std = get_bg(clr_spc, img_size)


# Visualize different p values for adaptive gaussian
for p in p_vals:
    
    fg_imgs, pred_boxes = get_fg_boxes_adptv_gaussian(clr_spc, img_size, 3, p, bg_mean, bg_std)
    gif_list = [Image.fromarray(img) for img in fg_imgs[1506:]]
    
    frame_one = gif_list[0]
    gif_name = "p_" + str(p) + ".gif"
    frame_one.save(gif_name, format="GIF", append_images=gif_list,
                   save_all=True, duration=100, loop=0)

    gif_list = []
    frame_one = []


# Create gif for the last 10 seconds for each color space
for clr_spc in color_spaces.keys():
    
    if clr_spc != "GRAY":
    
        bg_mean, bg_std = get_bg(clr_spc, img_size)

        fg_imgs, pred_boxes = get_fg_boxes_adptv_gaussian(clr_spc, img_size, 3, 0.007, bg_mean, bg_std)
        gif_list = [Image.fromarray(img) for img in fg_imgs[1506:]]

        frame_one = gif_list[0]
        gif_name = str(clr_spc) + ".gif"
        frame_one.save(gif_name, format="GIF", append_images=gif_list,
                       save_all=True, duration=100, loop=0)

        gif_list = []
        frame_one = []
        

# Get search results
print("Grid search results for Non Adaptive Gaussian")
nonadp_res = pd.read_csv("gaussian_params.csv")
print(nonadp_res)
print("\n")


print("Grid search results for Adaptive Gaussian")
adp_res = pd.read_csv("adaptive_gaussian_params.csv")
print(adp_res)
print("\n")


print("Grid search results for Color Spaces")
nonadp_res = pd.read_csv("clrspc_adaptive_gaussian_params.csv")
print(nonadp_res)

