import numpy as np
import os
import h5py
from tqdm import tqdm


def main(root_dir):
    # create converted dir
    converted_dir = os.path.join(root_dir, 'precessed')
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)
    # get data
    print('loading fault_train')
    fault_train = np.concatenate([np.load(os.path.join(root_dir, 'ori_data', 'npzfiles', 'fault', f'faulttrain{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 10))], axis=0).astype(np.uint8)
    np.save(os.path.join(converted_dir, 'faulttrain.npy'), fault_train)
    del fault_train
    print('loading fault_val')
    fault_val = np.concatenate([np.load(os.path.join(root_dir, 'ori_data', 'npzfiles', 'fault', f'faultval{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 3))], axis=0).astype(np.uint8)
    np.save(os.path.join(converted_dir, 'faultval.npy'), fault_val)
    del fault_val
    print('loading fault_test')
    fault_test = np.concatenate([np.load(os.path.join(root_dir,  'ori_data', 'npzfiles', 'fault', f'faulttest{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 8))], axis=0).astype(np.uint8)
    np.save(os.path.join(converted_dir, 'faulttest.npy'), fault_test)
    del fault_test
    print('loading seis_train')
    seis_train = np.concatenate([np.load(os.path.join(root_dir,  'ori_data' ,'npzfiles', 'seis', f'seistrain{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 10))], axis=0)
    np.save(os.path.join(converted_dir, 'seistrain.npy'), seis_train)
    del seis_train
    print('loading seis_val')
    seis_val = np.concatenate([np.load(os.path.join(root_dir,  'ori_data', 'npzfiles', 'seis', f'seisval{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 3))], axis=0)
    np.save(os.path.join(converted_dir, 'seisval.npy'), seis_val)
    del seis_val
    print('loading seis_test')
    seis_test = np.concatenate([np.load(os.path.join(root_dir,  'ori_data', 'npzfiles', 'seis', f'seistest{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 8))], axis=0)
    np.save(os.path.join(converted_dir, 'seistest.npy'), seis_test)
    del seis_test
    
    # save result
    
    
    
    
    
    
    
    # get_slice(seis_train, fault_train, converted_dir, 'train')
    # get_slice(seis_val, fault_val, converted_dir, 'val')
    # get_slice(seis_test, fault_test, converted_dir, 'test')
        
    
if __name__ == '__main__':
    root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/pulic_data'
    main(root_dir)