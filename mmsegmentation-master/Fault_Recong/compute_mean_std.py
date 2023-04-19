import cv2
import numpy as np
import os

def main(root_dir):
    # read train images
    img_lst = os.listdir(os.path.join(root_dir, 'train', 'image'))
    imgs = []
    for img_name in img_lst:
        imgs.append(cv2.imread(os.path.join(root_dir, 'train', 'image', img_name), cv2.IMREAD_UNCHANGED))
    imgs = np.stack(imgs)
    print(f'mean is {np.mean(imgs)}')
    print(f'std is {np.std(imgs)}')

if __name__ == '__main__':
    root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/2d_slices'
    main(root_dir)