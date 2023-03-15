import h5py
import numpy as np

with h5py.File('/home/zhangzr/FaultRecongnition/pytorch3dunet/3dunet-pdo/Predict-pdo/real/Fault_data_test_predictions.h5','r') as f:
    data = f['predictions'][:]
data = data.squeeze(0)
data[data >= 0.5] = 1
data[data < 0.5] = 0
data_flatten = data.flatten()
data_flatten.tofile('/home/zhangzr/FaultRecongnition/pytorch3dunet/3dunet-pdo/Predict-pdo/real/s4_unet_3d_predict.bin')
# np.save('/home/zhangzr/FaultRecongnition/pytorch3dunet/3dunet/3DUnet-Predict/real/unet_3d_predict.bin', data)
# data_read = np.fromfile('/home/zhangzr/FaultRecongnition/pytorch3dunet/3dunet-pdo/Predict-pdo/real/s4_unet_3d_predict.bin', dtype=np.single)
# data_read = data_read.reshape(101,1201,2751)
# print(data==data_read)