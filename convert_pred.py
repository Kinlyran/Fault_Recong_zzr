import h5py
import numpy as np

with h5py.File('/home/zhangzr/FaultRecongnition/Fault_data/Predict-pdo/Fault_data_test_predictions.h5','r') as f:
    data = f['predictions'][:]
data = data.squeeze(0)
data[data >= 0.5] = 1
data[data < 0.5] = 0
np.save('/home/zhangzr/FaultRecongnition/Fault_data/Predict-pdo/unet-3d-s4-test_predict.npy', data)