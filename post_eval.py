import h5py
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score
import cv2



def convert_gt(gt):
    h, w = gt.shape
    h_s, w_s = int(h/3), int(w/3)
    gt = gt.astype(np.float32)
    gt = cv2.resize(gt,(w_s,h_s))
    gt = gt>0.5
    gt = gt.astype(np.float32)
    return gt

def convert_score(score):
    h, w = score.shape
    h_s, w_s = int(h/3), int(w/3)
    score = cv2.resize(score,(w_s,h_s))
    return score

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
    
    


def post_eval(predict_path, gt_path):
    pred = np.load(os.path.join(predict_path, 'predict.npy'), mmap_mode='r')
    score = np.load(os.path.join(predict_path, 'score.npy'), mmap_mode='r')
    # f =  h5py.File(predict_path, 'r')
    # pred = f['predictions']
    # score = f['score']
    
    gt = np.load(gt_path, mmap_mode='r')
    
    running_dice = []
    running_acc = []
    running_ap = []
    for i in tqdm(range(0, gt.shape[0], 5)):
        gt_slice = convert_gt(gt[i,:,800:1300])
        pred_slice = convert_gt(pred[i,:,800:1300])
        score_slice = convert_score(score[i,:,800:1300])
        dice = dice_coefficient(gt_slice, pred_slice)
        acc = compute_acc(gt_slice, pred_slice)
        ap = compute_ap(gt_slice, score_slice)
        running_dice.append(dice)
        running_acc.append(acc)
        running_ap.append(ap)
    # f.close()
    print(f'Dice Score is: {np.mean(running_dice)} \n Acc is {np.mean(running_acc)} \n Ap is {np.mean(running_ap)}')
        
            
            
            
if __name__ == '__main__':
    # predict_path = '/home/zhangzr/FaultRecongnition/MIM-Med3D/output/Fault_Baseline/unetr_base_supbaseline_p16_public/test_pred/seistest.h5'
    # gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed/test/fault/faulttest.npy'
    # post_eval(predict_path, gt_path)
    predict_path = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512/predict'
    gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed/test/fault/faulttest.npy'
    post_eval(predict_path, gt_path)
    