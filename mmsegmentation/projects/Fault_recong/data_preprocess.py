import os
import numpy as np
import glob
import cv2
import random

def main(root_dir, save_dir):
    # create missing dir
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, "train", "image"))
        os.makedirs(os.path.join(save_dir, "train", "ann"))
        os.makedirs(os.path.join(save_dir, "val", "image"))
        os.makedirs(os.path.join(save_dir, "val", "ann"))
    
    # split data
    all_data = [i for i in range(173)]
    ratio = 0.9
    train_lst = random.sample(all_data, int(ratio * len(all_data)))
    val_lst = [item for item in all_data if item not in train_lst]
        
    for idx in range(173):
        feature = np.fromfile(os.path.join(root_dir,f'GYX3D2018-PSDM-VTI-CG1203-400Km2-DP-50_Feature{str(idx)}.bin'), dtype=np.double).reshape(128, 128)
        feature = (feature - feature.min()) / (feature.max() - feature.min())
        feature = (255 * feature).astype(np.uint8)
        label = np.fromfile(os.path.join(root_dir, f'Label{str(idx)}.bin'), dtype=np.double).reshape(128, 128).astype(np.uint8)
        # save
        if idx in train_lst:
            feature_save_path = os.path.join(save_dir, "train", "image", str(idx) + '.png')
            label_save_path = os.path.join(save_dir, "train", "ann", str(idx) + '.png')
        if idx in val_lst:
            feature_save_path = os.path.join(save_dir, "val", "image", str(idx) + '.png')
            label_save_path = os.path.join(save_dir, "val", "ann", str(idx) + '.png')
            
        cv2.imwrite(feature_save_path, feature)
        cv2.imwrite(label_save_path, label)
        
    

if __name__ == '__main__':
    root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/2Dfault/1700-0418/'
    save_dir = '/home/zhangzr/FaultRecongnition/Fault_data/2Dfault/converted/'
    main(root_dir, save_dir)