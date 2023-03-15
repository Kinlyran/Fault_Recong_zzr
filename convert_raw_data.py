import h5py
import segyio
import numpy as np
import os
from tqdm import tqdm


'''
data = np.fromfile('/home/zhangzr/FaultRecongnition/Fault_data/real_data/seis.dat')
print(data.shape)

with h5py.File('Fault_data/hdf5/train/1.h5', 'r') as f:
    data = f['raw'][:]
print(data.min(), data.max())

root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/hdf5/train'
all_files = os.listdir(root_dir)

all_data = []
for item in all_files:
    with h5py.File(os.path.join(root_dir, item), 'r') as f:
        data = np.array(f['raw'][:])
    all_data.append(data)
all_data = np.concatenate(all_data)
print(f'min is {all_data.min()}, max is {all_data.max()}')
'''

def dat2h5():
    data_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_data/seis.dat'
    

    cube = np.fromfile(data_path, dtype=np.single)
    cube = cube.reshape(101,1201,2751)
    # standard
    mean = np.mean(cube)
    std = np.std(cube)
    cube = (cube - mean) / std

    f = h5py.File('/mnt/disk1/zhangzr/dataset/Fault_data_test.h5','w') 
    f['raw'] = cube
    # f['label'] = np.zeros((x_size, y_size, z_size))
    f.close() 
    # np.save('', cube)
if __name__ == '__main__':
    dat2h5()

    




