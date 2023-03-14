import h5py
import cv2
import os
from tqdm import tqdm

def main():
    scr_root_path = '/mnt/disk1/zhangzr/dataset/Fault_data/hdf5'
    dst_path = '/mnt/disk1/zhangzr/dataset/Fault_data/2d-simulate-data/'
    if not os.path.exists(dst_path):
        os.makedirs(os.path.join(dst_path, 'train', 'image'))
        os.makedirs(os.path.join(dst_path, 'train', 'ann'))
        os.makedirs(os.path.join(dst_path, 'val', 'image'))
        os.makedirs(os.path.join(dst_path, 'val', 'ann'))

    
    # convert train data
    data_lst = os.listdir(os.path.join(scr_root_path, 'train'))
    for k, item in tqdm(enumerate(data_lst)):
        with h5py.File(os.path.join(scr_root_path, 'train', item), 'r') as f:
            image_cube = f['raw'][:]
            label = f['label'][:]
            label = label.squeeze(0)
        for i in range(128):
            image_slice = image_cube[:,:,i]
            # [0-1] scale
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            image_slice = image_slice * 255
            label_slice = label[:,:,i]
            cv2.imwrite(os.path.join(dst_path, 'train', 'image', f'cube_{k}_slice_{i}.png'), image_slice)
            cv2.imwrite(os.path.join(dst_path, 'train', 'ann', f'cube_{k}_slice_{i}.png'), label_slice)
            
            
    
    # convert val data
    data_lst = os.listdir(os.path.join(scr_root_path, 'val'))
    for k, item in tqdm(enumerate(data_lst)):
        with h5py.File(os.path.join(scr_root_path, 'val', item), 'r') as f:
            image_cube = f['raw'][:]
            label = f['label'][:]
            label = label.squeeze(0)
        for i in range(128):
            image_slice = image_cube[:,:,i]
            # [0-1] scale
            image_slice = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
            image_slice = image_slice * 255
            label_slice = label[:,:,i]
            cv2.imwrite(os.path.join(dst_path, 'val', 'image', f'cube_{k}_slice_{i}.png'), image_slice)
            cv2.imwrite(os.path.join(dst_path, 'val', 'ann', f'cube_{k}_slice_{i}.png'), label_slice)
    



if __name__ == '__main__':
    main()