import h5py
from monai.metrics import DiceMetric
import os
import torch
import numpy as np
from tqdm import tqdm

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

def post_eval(predict_path, gt_path):
    pred_lst = os.listdir(predict_path)
    dice_scores = []
    for item in pred_lst:
        with h5py.File(os.path.join(predict_path, item), 'r') as f:
            pred = f['predictions'][:]
        with h5py.File(os.path.join(gt_path, item), 'r') as f:
            gt = f['label'][:]
        pred = torch.FloatTensor(pred).unsqueeze(0)
        gt = torch.FloatTensor(gt).unsqueeze(0)
        dice_metric(y_pred=pred, y=gt)
        dice = dice_metric.aggregate().item()
        dice_scores.append(dice)
    print(f'Dice Score is: {np.mean(dice_scores)}')
        
            
            
            
if __name__ == '__main__':
    predict_path = '/home/zhangzr/FaultRecongnition/MIM-Med3D/output/Fault_Baseline/unetr_base_supbaseline_p16_fault/preds'
    gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/crop/val'
    post_eval(predict_path, gt_path)
    
    