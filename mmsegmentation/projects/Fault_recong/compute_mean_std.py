import numpy as np
import cv2
import os
from tqdm import tqdm

def main(root_path):
    data_lst = os.listdir(os.path.join(root_path, 'train', 'image'))
    running_mean = []
    runing_std = []
    for image_name in tqdm(data_lst):
        img = np.load(os.path.join(root_path, 'train', 'image', image_name))
        running_mean.append(np.mean(img))
        runing_std.append(np.std(img))
    print(f'mean is {np.mean(running_mean)}')
    print(f'std is {np.mean(runing_std)}')

if __name__ == '__main__':
    root_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/2d_slices'
    main(root_path)