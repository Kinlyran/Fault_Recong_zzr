# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile

from mmpretrain.registry import TRANSFORMS
# from mmpretrain.utils import datafrombytes
import os
import cv2

@TRANSFORMS.register_module()
class PerImageNormalization(BaseTransform):
    """Rerange the image pixel value.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, ignore_zoro=False):
        self.ignore_zoro = ignore_zoro
        
    def transform(self, results: dict) -> dict:
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        if self.ignore_zoro:
            mean = np.mean(img[img != 0.0])
            std = np.std(img[img != 0.0])
            img[img != 0.0] = (img[img != 0.0] - mean) / std
        else:
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / std
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ignore_zoro={self.ignore_zoro})'
        return repr_str

@TRANSFORMS.register_module()
class Rerange(BaseTransform):
    """Rerange the image pixel value.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def transform(self, results: dict) -> dict:
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@TRANSFORMS.register_module()
class LoadImageFromNpy(BaseTransform):
    """
    Load Image from .npy file, Similiar to LoadImageFromFile, different from LoadImageFromNDArray
    """
    def __init__(self,
                 to_float32: bool = False,
                 decode_backend: str = 'numpy',
                 force_3_channel: bool = False,
                 convert_25d: bool = False,
                 step: int = None, # only used when convert_25d == True
                 max_slice_id: int = None, # only used when convert_25d == True
                 backend_args: Optional[dict] = None) -> None:
        self.to_float32 = to_float32
        self.decode_backend = decode_backend
        self.backend_args = backend_args.copy() if backend_args else None
        self.force_3_channel = force_3_channel
        self.convert_25d = convert_25d
        self.step = step
        self.max_slice_id = max_slice_id

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        img = np.load(results['img_path'])
        if self.to_float32:
            img = img.astype(np.float32)
        if self.force_3_channel:
            img = np.stack([img, img, img], axis=2)
        if self.convert_25d:
            cur_slice = int(results['img_path'].split('/')[-1].replace('.npy', ''))
            cur_path_lst = results['img_path'].split('/')[:-1]
            cur_path_lst[0] = '/' + cur_path_lst[0]
            prev_path_lst = cur_path_lst + [str(max(0, cur_slice - self.step)) + '.npy']
            future_path_lst = cur_path_lst + [str(min(self.max_slice_id, cur_slice + self.step)) + '.npy']
            prev_slice = os.path.join(*prev_path_lst)
            future_slice = os.path.join(*future_path_lst)
            img_prev = np.load(prev_slice)
            img_future = np.load(future_slice)
            if self.to_float32:
                img_prev = img_prev.astype(np.float32)
                img_future = img_future.astype(np.float32)
            img = np.stack([img_prev, img, img_future], axis=2)
            
            
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f"decode_backend='{self.decode_backend}', "
                    f'to_float32={self.to_float32}, '
                    f'backend_args={self.backend_args})')
        return repr_str