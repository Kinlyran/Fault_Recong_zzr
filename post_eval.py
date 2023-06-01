import h5py
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import cv2
import segyio


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
    
    


def post_eval_thebe(predict_path, gt_path):
    # pred = np.load(os.path.join(predict_path, 'predict.npy'), mmap_mode='r')
    score = np.load(predict_path, mmap_mode='r')
    print(f'score shape is {score.shape}')
    # f =  h5py.File(predict_path, 'r')
    # pred = f['predictions']
    # score = f['score']
    
    gt = np.load(gt_path, mmap_mode='r')
    print(f'gt shape is {gt.shape}')
    image_f1_mat = np.zeros((len(range(0, gt.shape[0], 5)), len(ths)))
    image_precision_mat = np.zeros((len(range(0, gt.shape[0], 5)), len(ths)))
    k = 0
    for i in tqdm(range(0, gt.shape[0], 5)):
        gt_slice = convert_gt(gt[i,:,800:1300])
        # gt_slice = gt[i,:,800:1300]å
        score_slice = convert_score(score[i,:,800:1300])
        # score_slice = convert_score(score[i,:,400:900])
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
            
def post_eval_2d_image(predict_path, gt_path):
    pred_image_name_lst = os.listdir(gt_path)
    image_f1_mat = np.zeros((len(pred_image_name_lst), len(ths)))
    image_precision_mat = np.zeros((len(pred_image_name_lst), len(ths)))
    k = 0
    for img_name in tqdm(pred_image_name_lst):
        gt_slice = cv2.imread(os.path.join(gt_path, img_name), cv2.IMREAD_UNCHANGED)
        score_slice = np.load(os.path.join(predict_path, img_name.replace('.png', '.npy')))
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



def post_eval_3d(predict_path, gt_path):
    # pred = np.load(os.path.join(predict_path, 'predict.npy'), mmap_mode='r')
    score = np.load(predict_path, mmap_mode='r')
    # f =  h5py.File(predict_path, 'r')
    # pred = f['predictions']
    # score = f['score']
    
    gt = segyio.tools.cube(gt_path)[373:,:,:]
    image_f1_mat = np.zeros((gt.shape[0], len(ths)))
    image_precision_mat = np.zeros((gt.shape[0], len(ths)))
    k = 0
    for i in tqdm(range(gt.shape[0])):
        gt_slice = convert_gt(gt[i,:,:])
        # gt_slice = gt[i,:,800:1300]
        # pred_slice = convert_gt(pred[i,:,800:1300])
        score_slice = convert_score(score[i,:,:])
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
            
if __name__ == '__main__':
    predict_path = '/home/zhangzr/FaultRecongnition/MIM-Med3D/output/Fault_Finetuning/swin_unetr_base_simmim_p16_real_labeled_crop_192-pos-weight-10-dilate-1/test_pred/mig_fill_score.npy'
    gt_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/origin_data/fault/label_fill.sgy'
    post_eval_3d(predict_path, gt_path)
    