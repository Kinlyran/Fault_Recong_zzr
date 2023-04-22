import numpy as np
import cv2
import os

def main(root_path):
    data_lst = os.listdir(os.path.join(root_path, 'train', 'image'))
    imgs = []
    for image_name in data_lst:
        imgs.append(cv2.imread(os.path.join(root_path, 'train', 'image', image_name), cv2.IMREAD_UNCHANGED))
    imgs = np.stack(imgs)
    print(f'mean is {np.mean(imgs)}')
    print(f'std is {np.std(imgs)}')

if __name__ == '__main__':
    root_path = '/home/zhangzr/FaultRecongnition/Fault_data/public_data/2d_slices'
    main(root_path)