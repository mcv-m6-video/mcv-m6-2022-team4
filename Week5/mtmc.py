import pickle
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from utils import parse_aicity_rects
import motmetrics as mm


data_path = "/export/home/group06/m6/data/train/"

def l2norm(X):

    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


with open("all_cars.pkl", "rb") as f:
    embeds = pickle.load(f)
    
all_res = {}
seqs = ["S01", "S03", "S04"]

for seq in seqs:

    kal_dets = {}
    for res in os.listdir("kal_dets_roi_parked"):
    
        if res.endswith(".txt") and res.startswith(seq):
            kal_dets[res[:8]] = parse_aicity_rects(os.path.join("kal_dets_roi_parked", res))

    seq_embeds = {k:v for k,v in embeds.items() if k[:3] == seq}
    
    car_map = {}
    hists = []
    
    i = 0
    for key in seq_embeds.keys():
    
        for car in seq_embeds[key].items():
            car_map[str(i)] = key + "_" + car[0]
            hist = cv2.calcHist([car[1]], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        
            hists.append(hist)
            i += 1
            
    hists = torch.tensor(np.array(hists))
    hists = torch.reshape(hists, (hists.shape[0], -1))
    hists = l2norm(hists)
    dists = hists.mm(hists.t())
    dist, indices = torch.sort(dists, descending=True)
    dist, indices = dist[:, 1:].numpy(), indices[:, 1:].numpy()
    
    hit_nums = [x[x>0.8].shape[0] for x in dist]
    hit_nums = np.array(hit_nums)
    
    id_list = {}
    all_ids = []
    for i, hit_num in enumerate(hit_nums):
        
        if hit_num > 2:
            pred_car = indices[i][:hit_num]
            
            if i not in all_ids:
                id_list[str(i)] = pred_car
                
                for pred in pred_car:
                    if pred not in all_ids:
                        all_ids.append(pred)
    
    
    mtmc_ids = {}

    for key in id_list:
        cams = []
        for embed in id_list[key]:
    
            cams.append(car_map[str(embed)][4:8])
            
        if len(set(cams)) > 1:
            mtmc_ids[car_map[key]] = list(map(lambda x:car_map[str(x)], id_list[key]))
            
            
    assigned_ids = {}

    for i, key in enumerate(mtmc_ids.items()):
        
        final_list = key[1]
        final_list.append(key[0])
        assigned_ids[str(i)] = final_list
        
    
    os.makedirs("mtmc_hist_preds", exist_ok=True)

    files = {}

    for key in assigned_ids.keys():   
        
        for elem in assigned_ids[key]:
            
            file_name = elem[:8]
            
            if file_name not in files.keys():
    
                f = open(os.path.join("mtmc_hist_preds", file_name + ".txt"), "a")
                files[f.name] = f
            
            frame = elem.split("_")[2]
            box_num = elem.split("_")[-1]
            
            box = kal_dets[file_name][frame][int(box_num)]
            bbox = box["bbox"]
            
            cur_file = files[os.path.join("mtmc_hist_preds", file_name + ".txt")]
            
            line = f'{int(frame)+1},{key},{bbox[0]},{bbox[1]},{bbox[2]-bbox[0]},{bbox[3]-bbox[1]},{box["conf"]},-1,-1,-1\n'
            cur_file.write(line)
            
    [f.close() for f in files.values()]
    
    for file in os.listdir("mtmc_hist_preds"):
    
        if file.endswith(".txt"):
    
            with open(os.path.join("mtmc_hist_preds", file)) as f:
                
                sorted_ids = sorted(f.readlines(), key=lambda x:int(x.split(",")[0]))
                
            with open(os.path.join("mtmc_hist_preds", file), "w") as f:
                
                f.writelines(sorted_ids)
            
    for cam in os.listdir(os.path.join(data_path, seq)):
    
        print(seq, cam)
        file_path = "/export/home/group06/m6/data/train/{}/{}".format(seq, cam)
        gt_path = os.path.join(file_path, "gt/gt.txt")
        det_path = os.path.join("mtmc_hist_preds", seq + "_" + cam + ".txt")
        
        gt = mm.io.load_motchallenge(gt_path)
        det = mm.io.load_motchallenge(det_path)
        acc = mm.utils.compare_to_groundtruth(gt, det)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["idf1", "idp", "idr", "precision", "recall"], name='acc')
        print(summary)
        all_res[seq + "_" + cam] = summary
        summary.to_csv(os.path.join("mtmc_hist_preds", "_".join((seq, cam, "summary")) + ".csv"), index=False)  
    

for seq in ["S01", "S03", "S04"]:
    
    print(seq)
    seq_res = {k:v for k,v in all_res.items() if k[:3] == seq}
    
    idf1 = np.mean([val["idf1"] for val in seq_res.values()])
    print("Avg idf1:", idf1)
    idp = np.mean([val["idp"] for val in seq_res.values()])
    print("Avg idp:", idp)
    idr = np.mean([val["idr"] for val in seq_res.values()])
    print("Avg idr:", idr)
    precision = np.mean([val["precision"] for val in seq_res.values()])
    print("Avg precision:", precision)
    recall = np.mean([val["recall"] for val in seq_res.values()])
    print("Avg recall:", recall)