import h5py
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import cv2


ths = [i / 100 for i in range(1, 100)]
print(f'ths is {ths}')

def convert_gt(gt):
    h, w = gt.shape
    h_s, w_s = int(h/3), int(w/3)
    # gt = cv2.resize(gt,(w_s,h_s), interpolation=cv2.INTER_NEAREST)
    gt = cv2.resize(gt,(w_s,h_s))
    gt = gt > 0.5
    gt = gt.astype(np.uint8)
    return gt

def convert_score(score):
    h, w = score.shape
    h_s, w_s = int(h/3), int(w/3)
    score = cv2.resize(score,(w_s,h_s))
    return score
    
    


def post_eval(predict_path, gt_path):
    # pred = np.load(os.path.join(predict_path, 'predict.npy'), mmap_mode='r')
    score = np.load(predict_path, mmap_mode='r')
    # f =  h5py.File(predict_path, 'r')
    # pred = f['predictions']
    # score = f['score']
    
    gt = np.load(gt_path, mmap_mode='r')
    image_f1_mat = np.zeros((len(range(0, gt.shape[0], 5)), len(ths)))
    image_precision_mat = np.zeros((len(range(0, gt.shape[0], 5)), len(ths)))
    k = 0
    for i in tqdm(range(0, gt.shape[0], 5)):
        gt_slice = convert_gt(gt[i,:,800:1300])
        # gt_slice = gt[i,:,800:1300]
        # pred_slice = convert_gt(pred[i,:,800:1300])
        score_slice = convert_score(score[i,:,800:1300])
        # score_slice = score[i,:,800:1300]
        y_true_f = gt_slice.flatten()
        y_score_f = score_slice.flatten()
        j = 0
        for th in ths:
            y_pred_f = (y_score_f > th).astype(np.uint8)
            image_precision_mat[k, j] = precision_score(y_true_f, y_pred_f)
            image_f1_mat[k, j] = f1_score(y_true_f, y_pred_f)
            j += 1
        k += 1
    ap = np.mean(image_precision_mat)
    OIS = np.mean(np.amax(image_f1_mat, axis=1))
    ODS = np.amax(np.mean(image_f1_mat, axis=0))
    print(f'Ap is {ap} \n OIS is {OIS}\n ODS is {ODS}')

def post_eval_val(predict_path, gt_path):
    item_lst = os.listdir(gt_path)
    running_score = []
    for item in tqdm(item_lst):
        with h5py.File(os.path.join(gt_path, item)) as f:
            gt = f['label'][:]
        with h5py.File(os.path.join(predict_path, item)) as f:
            pred = f['predictions'][:]
        f1 = f1_score(gt.flatten(), pred.flatten())
        # print(f'Current F1 is {f1}')
        running_score.append(f1)
    print(f'Average F1 is {np.mean(running_score)}')
            
            
            
if __name__ == '__main__':
    # predict_path = '/home/zhangzr/FaultRecongnition/MIM-Med3D/output/Fault_Baseline/unetr_base_supbaseline_p16_public/test_pred/seistest.h5'
    # gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed/test/fault/faulttest.npy'
    # post_eval(predict_path, gt_path)
    predict_path = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice-128x128/predict/score.npy'
    gt_path = '/home/zhangzr/Fault_Recong/Fault_data/public_data/precessed/test/fault/faulttest.npy'
    post_eval(predict_path, gt_path)
    # post_eval_val(predict_path, gt_path)
    