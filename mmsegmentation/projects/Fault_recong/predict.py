import torch
import mmcv
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
import numpy as np
import segyio
from tqdm import tqdm
import cv2


def predict_3d(config_file, checkpoint_file, input_cube, save_path, device='cuda', force_3_chan=False, convert_25d=False, step=None):
    
    # init model
    model = init_model(config_file, checkpoint_file, device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    # load predict image cube
    if isinstance(input_cube, str):
        print(f'loading image: {input_cube}...')
        if '.npy' in input_cube:
            image = np.load(input_cube, mmap_mode='r')
        elif '.sgy' in input_cube:
            image = segyio.tools.cube(input_cube)
        else:
            raise TypeError
    else:
        image = input_cube
    
    print('start predict')
    predict = []
    prob = []
    for i in tqdm(range(image.shape[0])):
        image_slice = image[i, :, :]
        if force_3_chan:
            image_slice = np.stack([image_slice, image_slice, image_slice], axis=2)
        if convert_25d:
            image_prev = image[max(i - step, 0), :, :]
            image_future = image[min(i + step, image.shape[0]-1), :, :]
            image_slice = np.stack([image_prev, image_slice, image_future], axis=2)
        result = inference_model(model, image_slice.copy())
        predict.append(result.pred_sem_seg.data.detach().cpu().squeeze(0).numpy())
        prob.append(torch.sigmoid(result.seg_logits.data.detach().cpu().squeeze(0)).numpy())
    predict = np.stack(predict, axis=0)
    prob = np.stack(prob, axis=0)
    
    print('saving result....')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'predict.npy'), predict)
    np.save(os.path.join(save_path, 'score.npy'), prob)


def predict_2d(config_file, checkpoint_file, input_path, save_path, device='cuda', force_3_chan=False):
    
    # init model
    model = init_model(config_file, checkpoint_file, device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    # get predice image list
    pred_img_name_lst = os.listdir(input_path)
    
    # create missing dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
        
    print('start predict')
    for img_name in tqdm(pred_img_name_lst):
        if '.npy' in img_name:
            image_slice = np.load(os.path.join(input_path, img_name))
        elif '.png' in img_name:
            image_slice = cv2.imread(os.path.join(input_path, img_name), cv2.IMREAD_UNCHANGED)
        if force_3_chan:
            image_slice = np.stack([image_slice, image_slice, image_slice], axis=2)
        result = inference_model(model, image_slice.copy())
        np.save(os.path.join(save_path, img_name),torch.sigmoid(result.seg_logits.data.detach().cpu().squeeze(0)).numpy())
    
        
    
    
     
 
if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser(description='Using 2D segmentation to predict 2D/3D Fault')
    args.add_argument('--config', type=str, help='model config file path', default='./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512_per_image_normal_pos_weight_10/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512_per_image_normal_pos_weight_10.py')
    args.add_argument('--checkpoint', type=str, help='model checkpoint path', default='./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512_per_image_normal_pos_weight_10/Best.pth')
    args.add_argument('--input', type=str, help='input image/cube path', default='/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed/test/seis/seistest.npy')
    args.add_argument('--save_path', type=str, help='path to save predict result', default='./output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512_per_image_normal_pos_weight_10/predict')
    args.add_argument('--predict_type', type=str, help='predict 2d/3d image', default='3d')
    args.add_argument('--device', default='cuda:0')
    args.parse_args()
    
    
    config_file = args.config
    checkpoint_file = args.checkpoint
    input = args.input
    save_path = args.save_path
    if args.predict_type == '3d':
        predict_3d(config_file, checkpoint_file, input, save_path, device=args.device, convert_25d=True, step=5)
    elif args.predict_type == '2d':
        predict_2d(config_file, checkpoint_file, input, save_path, device=args.device, force_3_chan=True)