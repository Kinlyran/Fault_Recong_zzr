import torch
import mmcv
import matplotlib.pyplot as plt
from mmengine.model.utils import revert_sync_batchnorm
from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
import numpy as np
import segyio
from tqdm import tqdm


def main(config_file, checkpoint_file, input_cube_path, save_path, device='cuda'):
    
    # init model
    model = init_model(config_file, checkpoint_file, device)
    if device == 'cpu':
        model = revert_sync_batchnorm(model)
    
    # load predict image cube
    print(f'loading image: {input_cube_path}...')
    if '.npy' in input_cube_path:
        image = np.load(input_cube_path)
    elif '.sgy' in input_cube_path:
        image = segyio.tools.cube(input_cube_path)
    else:
        raise TypeError
    
    print('start predict')
    predict = []
    prob = []
    for i in tqdm(range(image.shape[0])):
        result = inference_model(model, image[i, :, :])
        predict.append(result.pred_sem_seg.data.detach().cpu().squeeze(0).numpy())
        prob.append(torch.sigmoid(result.seg_logits.data.detach().cpu().squeeze(0)).numpy())
    predict = np.stack(predict, axis=0)
    prob = np.stack(prob, axis=0)
    
    print('saving result....')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'predict.npy'), predict)
    np.save(os.path.join(save_path, 'prob.npy'), prob)
    
        
    
    
     
 
if __name__ == '__main__':
    config_file = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice-128x128/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice-128x128.py'
    checkpoint_file = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice-128x128/Best_Dice_57.pth'
    input_cube_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed/seistest.npy'
    save_path = '/home/zhangzr/FaultRecongnition/mmsegmentation/output/swin-base-patch4-window7_upernet_8xb2-160k_fault_public_slice-128x128/predict'
    main(config_file, checkpoint_file, input_cube_path, save_path, device='cpu')