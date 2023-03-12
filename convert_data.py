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

def sgy2h5():
    data_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_data/fault.sgy'
    with segyio.open(data_path, ignore_geometry=True) as segyfile:

        # Memory map file for faster reading (especially if file is big...)
        segyfile.mmap()
        # print(f'CDP_X Range is {segyfile.attributes(segyio.TraceField.CDP_X)[:].min(), segyfile.attributes(segyio.TraceField.CDP_X)[:].max()}')
        # print(f'CDP_Y Range is {segyfile.attributes(segyio.TraceField.CDP_Y)[:].min(), segyfile.attributes(segyio.TraceField.CDP_Y)[:].max()}')
        # print(f'Unique X Range({np.unique(segyfile.attributes(segyio.TraceField.CDP_X)[:])})')
        # print(f'Unique Y Range({np.unique(segyfile.attributes(segyio.TraceField.CDP_Y)[:])})')
        # print(len(np.unique(segyfile.attributes(segyio.TraceField.CDP_X)[:])) * len(np.unique(segyfile.attributes(segyio.TraceField.CDP_Y)[:])))
        x_size = len(np.unique(segyfile.attributes(segyio.TraceField.CDP_X)[:]))
        y_size = len(np.unique(segyfile.attributes(segyio.TraceField.CDP_Y)[:]))
        z_size = len(segyfile.samples)
        cube = np.zeros((x_size, y_size, z_size), dtype=np.float64)
        x_start = segyfile.attributes(segyio.TraceField.CDP_X)[:].min()
        y_start = segyfile.attributes(segyio.TraceField.CDP_Y)[:].min()
        step = 20


        for trace_id in tqdm(range(segyfile.tracecount)):
            trace = segyfile.trace[trace_id]
            ori_x = segyfile.attributes(segyio.TraceField.CDP_X)[trace_id][0]
            ori_y = segyfile.attributes(segyio.TraceField.CDP_Y)[trace_id][0]
            cord_x = (ori_x - x_start) // step
            cord_y = (ori_y - y_start) // step
            cube[cord_x, cord_y, :] = trace

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
    sgy2h5()

    




