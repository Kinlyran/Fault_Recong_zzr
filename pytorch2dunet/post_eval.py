import h5py
from monai.metrics import DiceMetric
import os
import torch
import numpy as np
from tqdm import tqdm
import cv2

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

def post_eval(predict_path, gt_path):
    dice_scores = []
    for i in range(20):
        gts = np.zeros((128, 128, 128))
        for k in range(128):
            gt = cv2.imread(os.path.join(gt_path, f'cube_{i}_slice_{k}.png'), cv2.IMREAD_UNCHANGED)
            gts[:,:,k] = gt
        preds = np.zeros((128, 128, 128))
        for k in range(128):
            pred = cv2.imread(os.path.join(predict_path, f'cube_{i}_slice_{k}.png'), cv2.IMREAD_UNCHANGED)
            preds[:,:,k] = pred
        preds = torch.from_numpy(preds)
        gts = torch.from_numpy(gts)
        dice_metric(y_pred=preds, y=gts)
        dice = dice_metric.aggregate().item()
        dice_scores.append(dice)
    print(f'Dice Score is: {np.mean(dice_scores)}')
        
            
            
            
if __name__ == '__main__':
    predict_path = '/home/zhangzr/FaultRecongnition/pytorch2dunet/e2UNet_CKPTS/preds'
    gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/2d-simulate-data/val/ann'
    post_eval(predict_path, gt_path)
    
    