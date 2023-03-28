import numpy as np
import os
import h5py

save_path = '/home/zhangzr/FaultRecongnition/Fault_data/simulate_data/hdf5'
base_path = '/home/zhangzr/FaultRecongnition/Fault_data/simulate_data'
train_root_path = os.path.join(base_path, 'train')
val_root_path = os.path.join(base_path, 'validation')
dim=(128,128,128)
for i in range(200):
    gx  = np.fromfile(os.path.join(train_root_path, 'seis',f'{i}.dat'),dtype=np.single)
    fx  = np.fromfile(os.path.join(train_root_path, 'fault',f'{i}.dat'),dtype=np.single)
    gx = np.reshape(gx,dim)
    fx = np.reshape(fx,dim)
    gx = np.transpose(gx)
    fx = np.transpose(fx)

    # xm = np.mean(gx)
    # xs = np.std(gx)
    # gx = gx - xm
    # gx = gx / xs
    X = np.zeros(dim, dtype=np.single)
    Y = np.zeros(dim, dtype=np.single)
    X = np.reshape(gx, dim)
    Y = np.reshape(fx, dim)
    f = h5py.File(os.path.join(save_path, 'train',f'{i}.h5'),'w') 
    f['raw'] = X 
    f['label'] = Y 
    f.close() 


for i in range(20):
    gx  = np.fromfile(os.path.join(val_root_path, 'seis',f'{i}.dat'),dtype=np.single)
    fx  = np.fromfile(os.path.join(val_root_path, 'fault',f'{i}.dat'),dtype=np.single)
    gx = np.reshape(gx,dim)
    fx = np.reshape(fx,dim)
    gx = np.transpose(gx)
    fx = np.transpose(fx)

    X = np.zeros(dim, dtype=np.single)
    Y = np.zeros(dim, dtype=np.single)
    X = np.reshape(gx, dim)
    Y = np.reshape(fx, dim)
    f = h5py.File(os.path.join(save_path, 'val',f'{i}.h5'),'w') 
    f['raw'] = X 
    f['label'] = Y 
    f.close() 
        
        

