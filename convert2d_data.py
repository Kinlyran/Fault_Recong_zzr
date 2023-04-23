import h5py
import cv2
import os
from tqdm import tqdm
import numpy as np
import segyio

def main_v0():
    scr_root_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/crop'
    dst_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/2d_slices'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))

    
    # convert train data
    data_lst = os.listdir(os.path.join(scr_root_path, 'train'))
    for item in tqdm(data_lst):
        with h5py.File(os.path.join(scr_root_path, 'train', item), 'r') as f:
            image_cube = f['raw'][:]
            label = f['label'][:]
            # label = label.squeeze(0)
        num_id = int(item.split('.')[0])
        for i in range(128):
            label_slice = label[:,:,i]
            if np.sum(label_slice) > 0.03 * 128 * 128:
                image_slice = image_cube[:,:,i]
                # [0-1] scale
                image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
                image_slice = image_slice * 255
                
                cv2.imwrite(os.path.join(dst_path, 'train', 'image', f'cube_{num_id}_slice_{i}.png'), image_slice)
                cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'cube_{num_id}_slice_{i}.png'), label_slice)
            
            
    
    # convert val data
    data_lst = os.listdir(os.path.join(scr_root_path, 'val'))
    for item in tqdm(data_lst):
        with h5py.File(os.path.join(scr_root_path, 'val', item), 'r') as f:
            image_cube = f['raw'][:]
            label = f['label'][:]
            # label = label.squeeze(0)
        num_id = int(item.split('.')[0])
        for i in range(128):
            image_slice = image_cube[:,:,i]
            # [0-1] scale
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            image_slice = image_slice * 255
            label_slice = label[:,:,i]
            cv2.imwrite(os.path.join(dst_path, 'val', 'image', f'cube_{num_id}_slice_{i}.png'), image_slice)
            cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'cube_{num_id}_slice_{i}.png'), label_slice)
    
def main_v1():
    scr_root_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/precessed'
    dst_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/2d_slices'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))
    print('loading seis train data')
    seis_train = np.load(os.path.join(scr_root_path, 'seistrain.npy'))
    fault_train = np.load(os.path.join(scr_root_path, 'faulttrain.npy'))
    assert seis_train.shape == fault_train.shape
    for i in tqdm(range(seis_train.shape[0])):
        seis_slice = seis_train[i,:,:]
        seis_slice = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min())
        fault_slice = fault_train[i,:,:]
        # convert to gray
        seis_slice = 255 * seis_slice
        cv2.imwrite(os.path.join(dst_path, 'train', 'image', f'{i}.png'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'{i}.png'), fault_slice)
    del seis_train
    del fault_train
    
    print('loading seis val data')
    seis_val = np.load(os.path.join(scr_root_path,'seisval.npy'))
    fault_val = np.load(os.path.join(scr_root_path, 'faultval.npy'))
    assert seis_val.shape == fault_val.shape
    seis_val = (seis_val - seis_val.min()) / (seis_val.max() - seis_val.min())
    for i in tqdm(range(seis_val.shape[0])):
        seis_slice = seis_val[i,:,:]
        seis_slice = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min())
        fault_slice = fault_val[i,:,:]
        # convert to gray
        seis_slice = 255 * seis_slice
        cv2.imwrite(os.path.join(dst_path, 'val', 'image', f'{i}.png'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'{i}.png'), fault_slice)
    del seis_val
    del fault_val

def main_v2():
    data_root = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data'
    dst_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/2d_slices'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))
    seis_data = segyio.tools.cube(os.path.join(data_root, 'mig_fill.sgy'))
    # precess missing value
    # seis_data[seis_data==-912300] = seis_data[seis_data!=-912300].mean()
    # seis_data[seis_data==0.0] = seis_data[seis_data!=0.0].mean()
    fault = segyio.tools.cube(os.path.join(data_root, 'label_fill.sgy'))
    fault = fault.astype(np.uint8)
    for i in range(673):
        seis_slice = seis_data[:,:,i]
        seis_slice = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min())
        fault_slice = fault[:,:,i]
        # convert to gray
        seis_slice = 255 * seis_slice
        cv2.imwrite(os.path.join(dst_path, 'train', 'image', f'{i}.png'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'{i}.png'), fault_slice)
    for i in range(673,801):
        seis_slice = seis_data[:,:,i]
        seis_slice = (seis_slice - seis_slice.min()) / (seis_slice.max() - seis_slice.min())
        fault_slice = fault[:,:,i]
        # convert to gray
        seis_slice = 255 * seis_slice
        cv2.imwrite(os.path.join(dst_path, 'val', 'image', f'{i}.png'), seis_slice)
        cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'{i}.png'), fault_slice)
        
    

if __name__ == '__main__':
    main_v2()