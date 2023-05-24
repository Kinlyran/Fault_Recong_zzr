import torch
import mmcv
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
import numpy as np
import segyio
from tqdm import tqdm


def main(config_file, checkpoint_file, input_cube, save_path, device='cuda', force_3_chan=False, convert_25d=False, step=None):
    
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


def main_2d(config_file, checkpoint_file, input_path, save_path, device='cuda', force_3_chan=False):
    
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
        image_slice = np.load(os.path.join(input_path, img_name))
        if force_3_chan:
            image_slice = np.stack([image_slice, image_slice, image_slice], axis=2)
        result = inference_model(model, image_slice.copy())
        np.save(os.path.join(save_path, img_name),torch.sigmoid(result.seg_logits.data.detach().cpu().squeeze(0)).numpy())
    
        
    
    
     
 
if __name__ == '__main__':
    config_file = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_real_labeled_slice_25d-256x256-dilate_ft_3/swin-base-patch4-window7_upernet_8xb2-160k_fault_real_labeled_slice_25d-256x256-dilate_ft.py'
    checkpoint_file = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_real_labeled_slice_25d-256x256-dilate_ft_3/Best_Dice_28.pth'
    input_cube = segyio.tools.cube('/home/zhangzr/Fault_Recong/Fault_data/real_labeled_data/origin_data/seis/mig_fill.sgy')[373:,:,:]
    save_path = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_real_labeled_slice_25d-256x256-dilate_ft_3/predict'
    main(config_file, checkpoint_file, input_cube, save_path, device='cuda:1', convert_25d=True, step=5)
    
    
    # config_file = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_2Dfault_0519_slice_split_force3chan-256x256/swin-base-patch4-window7_upernet_8xb2-160k_fault_2Dfault_0519_slice_split_force3chan-256x256.py'
    # checkpoint_file = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_2Dfault_0519_slice_split_force3chan-256x256/Best.pth'
    # input_path = '/home/zhangzr/Fault_Recong/Fault_data/2Dfault_0519_256/converted_slice_split/val/image'
    # save_path = '/home/zhangzr/Fault_Recong/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_2Dfault_0519_slice_split_force3chan-256x256/predict'
    # main_2d(config_file, checkpoint_file, input_path, save_path, device='cuda:1', force_3_chan=True)