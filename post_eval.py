import h5py
from monai.metrics import DiceMetric
import os
import torch
import numpy as np
from tqdm import tqdm


def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def post_eval(predict_path, gt_path):
    pred_lst = os.listdir(predict_path)
    dice_scores = []
    for item in pred_lst:
        with h5py.File(os.path.join(predict_path, item), 'r') as f:
            pred = f['predictions'][:]
        with h5py.File(os.path.join(gt_path, item), 'r') as f:
            gt = f['label'][:]
        dice = dice_coefficient(gt, pred)
        dice_scores.append(dice)
    print(f'Dice Score is: {np.mean(dice_scores)}')
        
            
            
            
if __name__ == '__main__':
    predict_path = '/home/zhangzr/FaultRecongnition/MIM-Med3D/output/Fault_Finetune/unetr_base_vitmae_p16_m0.75_fault/preds'
    gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/crop/val'
    post_eval(predict_path, gt_path)
    
    