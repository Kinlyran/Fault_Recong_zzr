from typing import Optional, Sequence, Union
import os
import torch
from torch.utils.data import Dataset, ConcatDataset
import torch.distributed as ptdist
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import h5py
import segyio
import cv2
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    RandShiftIntensityd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    RandRotated,
    NormalizeIntensityd,
    ToTensord,
    Resized,
    Zoomd
)


class Fault(Dataset):
    def __init__(self, 
                root_dir: str, 
                split: str = 'train',
                is_ssl=False,
                mean=None,
                std=None,
                zoom=False,
                zoom_scale=None,
                dilate=False):
        self.root_dir = root_dir
        self.split = split
        self.is_ssl = is_ssl
        self.dilate = dilate
        if self.dilate:
            self.dilate_kernel = np.ones((3,3), dtype=np.uint8)
        if zoom:
            self.train_transform = Compose([Zoomd(keys=["image"], zoom=zoom_scale, mode='area', keep_size=False),
                                    Zoomd(keys=["label"], zoom=zoom_scale, mode='nearest', keep_size=False),
                                    # RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                                    # RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                                    # RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                                    # RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3, spatial_axes=(0, 1)),
                                    RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
                                    NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std, nonzero=True, channel_wise=False), # nonzero = False
                                    ])
            self.val_transform = Compose([Zoomd(keys=["image"], zoom=zoom_scale, mode='area', keep_size=False),
                                        Zoomd(keys=["label"], zoom=zoom_scale, mode='nearest', keep_size=False),
                                        NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std, nonzero=True, channel_wise=False)]) # nonzero = False])
        else:
            self.train_transform = Compose([
                                    # RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                                    # RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                                    # RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                                    # RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3, spatial_axes=(0, 1)),
                                    RandSpatialCropd(keys=["image", "label"], roi_size=(128, 128, 128), random_size=False),
                                    NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std, nonzero=True, channel_wise=False) # nonzero = False
                                    ])
            self.val_transform = NormalizeIntensityd(keys=["image"], subtrahend=mean, divisor=std, nonzero=True, channel_wise=False) # nonzero = False
        # self.convert_size = convert_size
        if self.split == 'train':
            self.data_lst = os.listdir(os.path.join(self.root_dir, 'train'))
        elif self.split == 'val':
            self.data_lst = os.listdir(os.path.join(self.root_dir, 'val'))
        else:
            raise ValueError('Only support split = train/val')
        
    
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        f = h5py.File(os.path.join(self.root_dir, self.split, self.data_lst[index]),'r') 
        image = f['raw'][:]
        if 'label' in f.keys():
            mask = f['label'][:]
            mask = mask.astype(np.float32)
            if self.dilate:
                for idx in range(mask.shape[0]):
                    mask[idx, :, :] = cv2.dilate(mask[idx, :, :], kernel=self.dilate_kernel, iterations=1) # iterations = 1, 3, 5
        else:
            mask = None
        # mask = np.squeeze(mask,0)
        f.close()
        if mask is None:
            return self.val_transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'image_name': self.data_lst[index]})
        elif self.split == 'train' and not self.is_ssl:
            return self.train_transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'label': torch.from_numpy(mask).unsqueeze(0),
                    'image_name': self.data_lst[index]})
        elif self.split == 'train' and self.is_ssl:
            return self.val_transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'image_name': self.data_lst[index]})

        elif self.split == 'val':
            return self.val_transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'label': torch.from_numpy(mask).unsqueeze(0),
                    'image_name': self.data_lst[index]})


class Fault_Simple(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_lst = os.listdir(self.root_dir)
        self.transform = NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=False) # nonzero = False
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        if '.sgy' in self.data_lst[index]:
            image = segyio.tools.cube(os.path.join(self.root_dir, self.data_lst[index]))
        elif '.npy' in self.data_lst[index]:
            image = np.load(os.path.join(self.root_dir, self.data_lst[index]))
        return self.transform({'image': torch.from_numpy(image).unsqueeze(0),
                                'image_name': self.data_lst[index]})




class FaultDataset(pl.LightningDataModule):
    def __init__(
        self,
        is_ssl=False,
        zoom=False,
        zoom_scale=None,
        dilate=False,
        unlabeled_data_root_dir_lst=None,
        labeled_data_root_dir_lst=None,
        test_data_root_dir=None,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
    ):
        super().__init__()
        self.is_ssl = is_ssl
        self.zoom = zoom
        self.zoom_scale = zoom_scale
        self.dilate = dilate
        self.test_data_root_dir = test_data_root_dir
        self.unlabeled_data_root_dir_lst = unlabeled_data_root_dir_lst
        self.labeled_data_root_dir_lst = labeled_data_root_dir_lst
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist



    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            train_ds = []
            valid_ds = []
            if self.unlabeled_data_root_dir_lst is not None:
                for data_root_dir in self.unlabeled_data_root_dir_lst:
                    train_ds.append(Fault(root_dir=data_root_dir, split='train', is_ssl=self.is_ssl, zoom=self.zoom, zoom_scale=self.zoom_scale, dilate=self.dilate))
                    valid_ds.append(Fault(root_dir=data_root_dir, split='val', is_ssl=self.is_ssl, zoom=self.zoom, zoom_scale=self.zoom_scale, dilate=self.dilate))
            if self.labeled_data_root_dir_lst is not None:
                for data_root_dir in self.labeled_data_root_dir_lst:
                    train_ds.append(Fault(root_dir=data_root_dir, split='train', is_ssl=self.is_ssl, zoom=self.zoom, zoom_scale=self.zoom_scale, dilate=self.dilate))
                    valid_ds.append(Fault(root_dir=data_root_dir, split='val', is_ssl=self.is_ssl, zoom=self.zoom, zoom_scale=self.zoom_scale, dilate=self.dilate))
                
                
            self.train_ds = ConcatDataset(train_ds)
            self.valid_ds = ConcatDataset(valid_ds)
          

        if stage in [None, "test"]:
            if self.test_data_root_dir is not None:
                self.test_ds = Fault_Simple(root_dir=self.test_data_root_dir)
            else:
                if self.labeled_data_root_dir_lst is not None:
                    test_ds = []
                    for data_root_dir in self.labeled_data_root_dir_lst:
                        test_ds.append(Fault(root_dir=data_root_dir, split='val', is_ssl=self.is_ssl, zoom=self.zoom, zoom_scale=self.zoom_scale, dilate=self.dilate))
                    self.test_ds = ConcatDataset(test_ds)

    def train_dataloader(self):
        if self.dist:
            dataloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(self.train_ds),
            drop_last=False,
        )
        else:
            dataloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
            
        return dataloader

    def val_dataloader(self):
        if self.dist:
            dataloader = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            # num_workers=0,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            sampler=DistributedSampler(self.valid_ds, shuffle=False, drop_last=False),
            drop_last=False,
        )
        else:
            dataloader = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            # num_workers=0,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
        return dataloader

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )