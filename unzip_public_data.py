import numpy as np
import os
import h5py
from tqdm import tqdm

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
    

def get_slice(seis, fault, converted_dir, split):
    slice_builder = SliceBuilder(raw_dataset=seis,
                                 label_dataset=fault,
                                 weight_dataset=None,
                                 patch_shape=(128, 128, 128),
                                 stride_shape=(128, 128, 128))
    crop_cubes_pos = slice_builder.raw_slices
    save_path = os.path.join(converted_dir, split)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, pos in enumerate(crop_cubes_pos):
        x_range = pos[0]
        y_range = pos[1]
        z_range = pos[2]
        print(f'Processing num {i} slice')
        seis_cube_crop = seis[x_range, y_range, z_range]
        label_cube_crop = fault[x_range, y_range, z_range]
        f = h5py.File(os.path.join(save_path, f'{i}.h5'),'w')
        f['raw'] = seis_cube_crop
        f['label'] = label_cube_crop
        f.close()

def main(root_dir):
    # create converted dir
    converted_dir = os.path.join(root_dir, 'precessed')
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)
    # get data
    print('loading fault_train')
    fault_train = np.concatenate([np.load(os.path.join(root_dir, 'npzfiles', 'fault', f'faulttrain{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 10))], axis=0).astype(np.uint8)
    np.save(os.path.join(converted_dir, 'faulttrain.npy'), fault_train)
    del fault_train
    print('loading fault_val')
    fault_val = np.concatenate([np.load(os.path.join(root_dir, 'npzfiles', 'fault', f'faultval{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 3))], axis=0).astype(np.uint8)
    np.save(os.path.join(converted_dir, 'faultval.npy'), fault_val)
    del fault_val
    print('loading fault_test')
    fault_test = np.concatenate([np.load(os.path.join(root_dir, 'npzfiles', 'fault', f'faulttest{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 8))], axis=0).astype(np.uint8)
    np.save(os.path.join(converted_dir, 'faulttest.npy'), fault_test)
    del fault_test
    print('loading seis_train')
    seis_train = np.concatenate([np.load(os.path.join(root_dir, 'npzfiles', 'seis', f'seistrain{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 10))], axis=0)
    np.save(os.path.join(converted_dir, 'seistrain.npy'), seis_train)
    del seis_train
    print('loading seis_val')
    seis_val = np.concatenate([np.load(os.path.join(root_dir, 'npzfiles', 'seis', f'seisval{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 3))], axis=0)
    np.save(os.path.join(converted_dir, 'seisval.npy'), seis_val)
    del seis_val
    print('loading seis_test')
    seis_test = np.concatenate([np.load(os.path.join(root_dir, 'npzfiles', 'seis', f'seistest{str(i)}.npz'))['arr_0'] for i in tqdm(range(1, 8))], axis=0)
    np.save(os.path.join(converted_dir, 'seistest.npy'), seis_test)
    del seis_test
    
    # save result
    
    
    
    
    
    
    
    # get_slice(seis_train, fault_train, converted_dir, 'train')
    # get_slice(seis_val, fault_val, converted_dir, 'val')
    # get_slice(seis_test, fault_test, converted_dir, 'test')
        
    
if __name__ == '__main__':
    root_dir = '/home/zhangzr/FaultRecongnition/Fault_data/public_data'
    main(root_dir)