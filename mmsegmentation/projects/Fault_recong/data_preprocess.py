import os
import numpy as np
import glob
import cv2
import random
import shutil

def random_sample_val(root_dir, save_dir):
    image_lst = os.listdir(os.path.join(root_dir, 'val', 'image'))
    sampled = random.sample(image_lst, 5000)
    os.makedirs(os.path.join(save_dir, "mini_val", "image"))
    os.makedirs(os.path.join(save_dir, "mini_val", "ann"))
    for img_name in sampled:
        shutil.copy(os.path.join(root_dir, 'val', 'image', img_name), os.path.join(save_dir, 'mini_val', 'image', img_name))
        shutil.copy(os.path.join(root_dir, 'val', 'ann', img_name), os.path.join(save_dir, 'mini_val', 'ann', img_name))
        
        
        
    

def main(root_dir, save_dir):
    # create missing dir
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, "train", "image"))
        os.makedirs(os.path.join(save_dir, "train", "ann"))
        os.makedirs(os.path.join(save_dir, "val", "image"))
        os.makedirs(os.path.join(save_dir, "val", "ann"))
    
    
    # split data
    ratio = 0.9
    slice_1700_all_data = [i for i in range(215)]
    slice_1700_train_lst = random.sample(slice_1700_all_data, int(ratio * len(slice_1700_all_data)))
    slice_1700_val_lst = [item for item in slice_1700_all_data if item not in slice_1700_train_lst]
    
    slice_1744_all_data = [i for i in range(234)]
    slice_1744_train_lst = random.sample(slice_1744_all_data, int(ratio * len(slice_1744_all_data)))
    slice_1744_val_lst = [item for item in slice_1744_all_data if item not in slice_1744_train_lst]
    
    slice_1828_all_data = [i for i in range(264)]
    slice_1828_train_lst = random.sample(slice_1828_all_data, int(ratio * len(slice_1828_all_data)))
    slice_1828_val_lst = [item for item in slice_1828_all_data if item not in slice_1828_train_lst]
    
    slice_1956_all_data = [i for i in range(278)]
    slice_1956_train_lst = random.sample(slice_1956_all_data, int(ratio * len(slice_1956_all_data)))
    slice_1956_val_lst = [item for item in slice_1956_all_data if item not in slice_1956_train_lst]
    
    for idx in range(len(slice_1700_all_data)):
        image = np.fromfile(os.path.join(root_dir, '0519-1700-256', f'GYX3D2018-PSDM-VTI-CG1203-400Km2-DP-50_Feature{idx}.bin'), dtype=np.double).reshape(256, 256)
        label = np.fromfile(os.path.join(root_dir, '0519-1700-256', f'Label{idx}.bin'), dtype=np.double).reshape(256, 256)
        if idx in slice_1700_train_lst:
            image_save_path = os.path.join(save_dir, "train", "image", '1700_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "train", "ann", '1700_' + str(idx) + '.png' )
        if idx in slice_1700_val_lst:
            image_save_path = os.path.join(save_dir, "val", "image", '1700_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "val", "ann", '1700_' + str(idx) + '.png' )
        np.save(image_save_path, image)
        cv2.imwrite(label_save_path, label)
    
    for idx in range(len(slice_1744_all_data)):
        image = np.fromfile(os.path.join(root_dir, '0519-1744-256', f'GYX3D2018-PSDM-VTI-CG1203-400Km2-DP-50_Feature{idx}.bin'), dtype=np.double).reshape(256, 256)
        label = np.fromfile(os.path.join(root_dir, '0519-1744-256', f'Label{idx}.bin'), dtype=np.double).reshape(256, 256)
        if idx in slice_1744_train_lst:
            image_save_path = os.path.join(save_dir, "train", "image", '1744_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "train", "ann", '1744_' + str(idx) + '.png' )
        if idx in slice_1744_val_lst:
            image_save_path = os.path.join(save_dir, "val", "image", '1744_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "val", "ann", '1744_' + str(idx) + '.png' )
        np.save(image_save_path, image)
        cv2.imwrite(label_save_path, label)
        
    for idx in range(len(slice_1828_all_data)):
        image = np.fromfile(os.path.join(root_dir, '0519-1828-256', f'GYX3D2018-PSDM-VTI-CG1203-400Km2-DP-50_Feature{idx}.bin'), dtype=np.double).reshape(256, 256)
        label = np.fromfile(os.path.join(root_dir, '0519-1828-256', f'Label{idx}.bin'), dtype=np.double).reshape(256, 256)
        if idx in slice_1828_train_lst:
            image_save_path = os.path.join(save_dir, "train", "image", '1828_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "train", "ann", '1828_' + str(idx) + '.png' )
        if idx in slice_1828_val_lst:
            image_save_path = os.path.join(save_dir, "val", "image", '1828_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "val", "ann", '1828_' + str(idx) + '.png' )
        np.save(image_save_path, image)
        cv2.imwrite(label_save_path, label)
    
    for idx in range(len(slice_1956_all_data)):
        image = np.fromfile(os.path.join(root_dir, '0519-1956-256', f'GYX3D2018-PSDM-VTI-CG1203-400Km2-DP-50_Feature{idx}.bin'), dtype=np.double).reshape(256, 256)
        label = np.fromfile(os.path.join(root_dir, '0519-1956-256', f'Label{idx}.bin'), dtype=np.double).reshape(256, 256)
        if idx in slice_1956_train_lst:
            image_save_path = os.path.join(save_dir, "train", "image", '1956_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "train", "ann", '1956_' + str(idx) + '.png' )
        if idx in slice_1956_val_lst:
            image_save_path = os.path.join(save_dir, "val", "image", '1956_' + str(idx) + '.npy')
            label_save_path = os.path.join(save_dir, "val", "ann", '1956_' + str(idx) + '.png' )
        np.save(image_save_path, image)
        cv2.imwrite(label_save_path, label)
            
        
        
    

if __name__ == '__main__':
    root_dir = '/home/zhangzr/Fault_Recong/Fault_data/2Dfault_0519_256/0519_4_256/0519'
    save_dir = '/home/zhangzr/Fault_Recong/Fault_data/2Dfault_0519_256/converted'
    main(root_dir, save_dir)
    # random_sample_val(root_dir, save_dir)