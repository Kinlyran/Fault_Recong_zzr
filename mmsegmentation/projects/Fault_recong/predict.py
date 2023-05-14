import torch
import mmcv
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
import numpy as np
import segyio
from tqdm import tqdm


def main(config_file, checkpoint_file, input_cube_path, save_path, device='cuda', force_3_chan=False, convert_25d=False, step=None):
    
    # init model
    model = init_model(config_file, checkpoint_file, device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    # load predict image cube
    print(f'loading image: {input_cube_path}...')
    if '.npy' in input_cube_path:
        image = np.load(input_cube_path, mmap_mode='r')
    elif '.sgy' in input_cube_path:
        image = segyio.tools.cube(input_cube_path)
    else:
        raise TypeError
    
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
    
        
    
    
     
 
if __name__ == '__main__':
    config_file = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512.py'
    checkpoint_file = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512/Best_Dice65.pth'
    input_cube_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed/test/seis/seistest.npy'
    save_path = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice_25d-512x512/predict'
    main(config_file, checkpoint_file, input_cube_path, save_path, device='cuda', convert_25d=True, step=5)