import h5py
import cv2
import os
from tqdm import tqdm
import segyio


def main():
    ratio = 0.8
    scr_root_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data'
    dst_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/2d_slices'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))

    
    # convert data
    seis_data = segyio.tools.cube(os.path.join(scr_root_path, 'mig_fill.sgy'))
    label = segyio.tools.cube(os.path.join(scr_root_path, 'label_fill.sgy'))
    
    assert seis_data.shape == label.shape
    h, w, d = seis_data.shape
    valid_slices = 775
    for i in range(valid_slices):
        image_slice = seis_data[:,i,:]
        # [0-1] scale
        image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        image_slice = image_slice * 255
        label_slice = label[:,i,:]
        if i <= valid_slices * ratio:
            cv2.imwrite(os.path.join(dst_path, 'train', 'image', f'slice_{i}.png'), image_slice)
            cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'slice_{i}.png'), label_slice)
        else:
            cv2.imwrite(os.path.join(dst_path, 'val', 'image', f'slice_{i}.png'), image_slice)
            cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'slice_{i}.png'), label_slice)
    
            
            
            



if __name__ == '__main__':
    main()