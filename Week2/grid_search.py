import pandas as pd
import itertools


def tune_alpha_and_p_and_clrspc():
    
    df_res = pd.DataFrame(columns=["Color_Space", "Alpha", "mAP"])
    
    alpha_list = np.arange(2, 6, 1)
    
    #p = np.arange(0.003, 0.01, 0.002)

    for clr_spc in color_spaces.keys():
        print(clr_spc)
        if clr_spc != "GRAY":
            
            bg_mean, bg_std = get_bg(clr_spc, img_size)

            for alpha in alpha_list:
                print(alpha)
                aps = []
                
                _, pred_boxes = get_fg_boxes_adptv_gaussian(clr_spc, img_size, alpha, 0.007, bg_mean, bg_std)
                for i in range(5):
                    aps.append(get_AP(fg_annots, pred_boxes)*100)

                df_res = df_res.append({
                    "Color_Space": clr_spc,
                    "Alpha": alpha,
                    "mAP": round(np.mean(aps), 3)
                }, ignore_index=True)

            df_res.to_csv("clrspc_adaptive_gaussian.csv", index=False)
    return df_res


def tune_alpha_and_p():
    
    df_res = pd.DataFrame(columns=["Alpha", "P", "mAP"])
    
    alpha_list = np.arange(2, 10, 1)
    p = np.arange(0.001, 0.01, 0.002)

    combs = itertools.product(alpha_list, p)
    for comb in combs:
        aps = []
        print(comb)
        
        fg_imgs, pred_boxes = get_fg_boxes_adptv_gaussian(clr_spc, img_size, comb[0], comb[1])
        for i in range(10):
            aps.append(get_AP(fg_annots, pred_boxes)*100)
            
        print(np.mean(aps))
        
        df_res = df_res.append({
            "Alpha": comb[0],
            "P": comb[1],
            "mAP": round(np.mean(aps), 3)
        }, ignore_index=True)
        
    df_res.to_csv("adaptive_gaussian.csv", index=False)
    return df_res


def tune_alpha():
    
    df_res = pd.DataFrame(columns=["Alpha", "mAP"])
    
    alpha_list = np.arange(3, 11, 1)

    for alpha in alpha_list:
        aps = []
        print(alpha)
        
        fg_imgs, pred_boxes = get_fg_boxes_gaussian(clr_spc, img_size, alpha)
        for i in range(10):
            aps.append(get_AP(fg_annots, pred_boxes)*100)
            
        print(np.mean(aps))
        
        df_res = df_res.append({
            "Alpha": alpha,
            "mAP": round(np.mean(aps), 3)
        }, ignore_index=True)
        
    df_res.to_csv("gaussian.csv", index=False)
    
    return df_res
