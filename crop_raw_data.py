import h5py
import segyio
import numpy as np
import os
from tqdm import tqdm
import random


def get_slice(seis, fault, save_path, patch_shape, stride_shape):
    slice_builder = SliceBuilder(raw_dataset=seis,
                                 label_dataset=None,
                                 weight_dataset=None,
                                 patch_shape=patch_shape,
                                 stride_shape=stride_shape
                                 # patch_shape=(256, 256, 256),
                                 # stride_shape=(128, 128, 128)
                                 )
    crop_cubes_pos = slice_builder.raw_slices
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    i = 0
    for pos in tqdm(crop_cubes_pos):
        x_range = pos[0]
        y_range = pos[1]
        z_range = pos[2]
        seis_cube_crop = seis[x_range, y_range, z_range]
        label_cube_crop = fault[x_range, y_range, z_range]
        # if np.sum(label_cube_crop) > 0.0:
        f = h5py.File(os.path.join(save_path, f'{i}.h5'),'w') 
        f['raw'] = seis_cube_crop
        f['label'] = label_cube_crop
        f.close() 
        i += 1
        


def dat2h5():
    """
    data_path = '/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/'
    # 501, 501, 801
    seis_data = segyio.tools.cube(os.path.join(data_path, 'origin_data', 'seis', 'mig_fill.sgy'))
    # precess missing value
    # seis_data[seis_data==-912300] = seis_data[seis_data!=-912300].mean()
    # seis_data[seis_data==0.0] = seis_data[seis_data!=0.0].mean()
    label = segyio.tools.cube(os.path.join(data_path, 'origin_data', 'fault', 'label_fill.sgy'))
    label = label.astype(np.uint8)
    get_slice(seis=seis_data[:373,:,:], fault=label[:373,:,:],save_path=os.path.join(data_path, 'crop_192x384x384', 'train'), patch_shape=(192, 384, 384), stride_shape=(96, 192, 192))
    get_slice(seis=seis_data[373:,:,:], fault=label[373:,:,:],save_path=os.path.join(data_path, 'crop_192x384x384', 'val'), patch_shape=(128, 384, 384), stride_shape=(128, 192, 192))
    """
    data_path = '/gpfs/share/home/2001110054/ondemand/code/Fault_Recong/Fault_data/public_data'
    
    print('loading seis train data')
    seis_train = np.load(os.path.join(data_path, 'precessed', 'train', 'seis', 'seistrain.npy'), mmap_mode='r')
    fault_train = np.load(os.path.join(data_path, 'precessed', 'train', 'fault', 'faulttrain.npy'), mmap_mode='r')
    get_slice(seis=seis_train, fault=fault_train, save_path=os.path.join(data_path, 'crop_256', 'train'), patch_shape=(256, 256, 256), stride_shape=(128, 128, 128))
    del seis_train
    del fault_train
    
    print('loading seis val data')
    seis_val = np.load(os.path.join(data_path, 'precessed','val', 'seis', 'seisval.npy'), mmap_mode='r')
    fault_val = np.load(os.path.join(data_path, 'precessed', 'val', 'fault', 'faultval.npy'), mmap_mode='r')
    get_slice(seis=seis_val, fault=fault_val, save_path=os.path.join(data_path, 'crop_256', 'val'), patch_shape=(200, 256, 256), stride_shape=(10, 128, 128))
    del seis_val
    del fault_val
    
    
    
    '''
    print('loading seis test data')
    seis_test = np.load(os.path.join(data_path, 'precessed','test', 'seis', 'seistest.npy'), mmap_mode='r')
    fault_test = np.load(os.path.join(data_path, 'precessed', 'test', 'fault', 'faulttest.npy'), mmap_mode='r')
    get_slice(seis=seis_test, fault=fault_test, save_path=os.path.join(data_path, 'crop', 'test'))
    del seis_test
    del fault_test
    '''
    
    
    

class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the the patch and stride shape
    """

    def __init__(self, raw_dataset, label_dataset, weight_dataset, patch_shape, stride_shape, **kwargs):
        """
        :param raw_dataset: ndarray of raw data
        :param label_dataset: ndarray of ground truth labels
        :param weight_dataset: ndarray of weights for the labels
        :param patch_shape: the shape of the patch DxHxW
        :param stride_shape: the shape of the stride DxHxW
        :param kwargs: additional metadata
        """

        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            # take the first element in the label_dataset to build slices
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)
        if weight_dataset is None:
            self._weight_slices = None
        else:
            self._weight_slices = self._build_slices(weight_dataset, patch_shape, stride_shape)
            assert len(self.raw_slices) == len(self._weight_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @property
    def weight_slices(self):
        return self._weight_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 64 and patch_shape[2] >= 64, 'Height and Width must be greater or equal 64'
    
if __name__ == '__main__':
    dat2h5()

    




