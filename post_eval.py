import h5py
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score


def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def compute_acc(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    return np.sum(y_true_f==y_pred_f) / y_true_f.shape

def compute_ap(y_true, y_score):
    y_true_f = y_true.flatten()
    y_score_f = y_score.flatten()
    return average_precision_score(y_true_f, y_score_f)
    
    


def post_eval_3d(predict_path, gt_path):
    pred_lst = os.listdir(predict_path)
    dice_scores = []
    acc_scores = []
    ap_scores = []
    for item in pred_lst:
        with h5py.File(os.path.join(predict_path, item), 'r') as f:
            pred = f['predictions'][:]
            score = f['score'][:]
        with h5py.File(os.path.join(gt_path, item), 'r') as f:
            gt = f['label'][:]

        dice = dice_coefficient(gt, pred)
        dice_scores.append(dice)

        acc = compute_acc(gt, pred)
        acc_scores.append(acc)
        
        ap = compute_ap(gt, score)
        ap_scores.append(ap)
        
    print(f'Dice Score is: {np.mean(dice_scores)} \n Acc is {np.mean(acc_scores)} \n Ap is {np.mean(ap_scores)}')
    

def post_eval_2d(predict_path, gt_path):
    pred = np.load(os.path.join(predict_path, 'predict.npy'), mmap_mode='r')
    score = np.load(os.path.join(predict_path, 'score.npy'), mmap_mode='r')
    gt = np.load(gt_path, mmap_mode='r')
    
    running_dice = []
    running_acc = []
    running_ap = []
    for i in tqdm(range(gt.shape[0])):
        dice = dice_coefficient(gt[i,:,:], pred[i,:,:])
        acc = compute_acc(gt[i,:,:], pred[i,:,:])
        ap = compute_ap(gt[i,:,:], score[i,:,:])
        running_dice.append(dice)
        running_acc.append(acc)
        running_ap.append(ap)
    
    print(f'Dice Score is: {np.mean(running_dice)} \n Acc is {np.mean(running_acc)} \n Ap is {np.mean(running_ap)}')

        
            
            
            
if __name__ == '__main__':
    predict_path = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice-512x512/predict'
    gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed/test/fault/faulttest.npy'
    post_eval_2d(predict_path, gt_path)
    
    