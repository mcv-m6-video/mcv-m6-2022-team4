import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pyflow
from utils import read_flow, plot_flow, get_metrics
from tqdm import tqdm
import itertools
import pandas as pd
import time


def exhaustive_search(template, target, metric='cv2.TM_CCORR_NORMED'):

    
    metric = eval(metric)
    result = cv2.matchTemplate(template, target, metric)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if metric in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        pos = min_loc
    else:
        pos = max_loc
    return pos


def find_template(template:np.ndarray, target:np.ndarray, search='exhaustive', metric='cv2.TM_CCORR_NORMED'):

    SEARCH_ALGOS = {
    'exhaustive': exhaustive_search,
    }

    return SEARCH_ALGOS[search](template, target, metric)


def get_optical_flow(img1:np.ndarray, img2:np.ndarray, block_size = 16, search_area = 16, 
                    comp = 'forward', search = 'exhaustive', metric = 'cv2.TM_CCORR_NORMED'):
    
    assert img1.shape == img2.shape, "Got different image sizes"
    h,w = img1.shape[:2]
    
    if comp == 'forward':
        pass
    elif comp == 'backward':
        img1, img2 = img2, img1
    else:
        print('check docs for available compensations')
        
    flow = np.zeros((h, w, 2), dtype=float)
    
    for i in tqdm(range(0, h-block_size, block_size)):
         for j in range(0, w - block_size, block_size):
                # get bbox of target where template will be searched
                top_left = (max(i-search_area, 0), max(j-search_area, 0))
                bottom_right = min(i+block_size+search_area, h), min(j+block_size+search_area, w)
                
                template = img1[i:i+block_size, j:j+block_size]
                target = img2[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]
                
                displacement = find_template(template, target, search, metric)
                
                v_flow = displacement[1] - (j-top_left[1]) 
                u_flow = displacement[0] - (i-top_left[0])
                
                flow[i:i+block_size, j:j+block_size] = [u_flow, v_flow]
    flow = np.dstack((flow[:,:,0], flow[:,:,1], np.ones_like(flow[:,:,0])))
    return flow


def main():
    
    img1 = cv2.imread('colored_000045_10.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('colored_000045_11.png', cv2.IMREAD_GRAYSCALE)
    gt_flow = read_flow('000045_10.png')
    
    df_res = pd.DataFrame(columns=["block_size", "search_area", "comp", "loss", "mse", "pepn", "inf_time"])
    
    block_size = [4, 8, 16, 32, 64] 
    search_area = [8, 16, 32, 64, 128]
    comp = ["forward", "backward"]
    METRICS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    
    combs = itertools.product(block_size, search_area, comp, METRICS)
    
    for comb in combs:
        print(comb)
        start_time = time.time()
        block_flow = get_optical_flow(img1, img2, block_size=comb[0], search_area=comb[1], comp=comb[2],search="exhaustive",
                                    metric=comb[3])

        mse,pepn = get_metrics(gt_flow, block_flow)
        print(f"Block_flow metrics, MSE:{mse}, PEPN:{pepn}")
        end_time = time.time()
        df_res = df_res.append({
                              "block_size": comb[0],
                              "search_area": comb[1],
                              "comp": comb[2],
                              "loss": comb[3],
                              "mse": mse,
                              "pepn": pepn,
                              "inf_time": end_time - start_time
                              }, ignore_index=True)
        
        df_res.to_csv("block_match.csv", index=False)
