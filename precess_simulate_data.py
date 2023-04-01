import numpy as np
import h5py
import os


def process_data(root_dir, split):
    seis_data_lst = os.listdir(os.path.join(root_dir, split, 'seis'))
    fault_data_lst = os.listdir(os.path.join(root_dir, split, 'fault'))
    for name in seis_data_lst:
        seis = np.fromfile(os.path.join(root_dir, split, 'seis', name), dtype=np.single)
        fault = np.fromfile(os.path.join(root_dir, split, 'fault', name), dtype=np.single)
        seis = seis.reshape((128, 128, 128))
        fault = fault.reshape((128, 128, 128))
        print(seis.min(), seis.max())


def main():
    root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/simulate_data'
    process_data(root_dir, 'train')

if __name__ == '__main__':
    main()
    