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

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
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
)


class Fault_Simulate(Dataset):
    def __init__(self,
                 root_dir,
                 split,):
        self.root_dir = root_dir
        self.split = split
        self.train_transform = Compose([RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                                RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3, spatial_axes=(0, 1)),
                                NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True)
                                    # RandRotated(keys=["image", "label"], prob=0.10, )
                                    ])
        self.val_transform = NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True)
        self.data_lst = os.listdir(os.path.join(root_dir, self.split, 'seis'))

    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        name = self.data_lst[index]
        seis = np.fromfile(os.path.join(self.root_dir, self.split, 'seis', name), dtype=np.single)
        fault = np.fromfile(os.path.join(self.root_dir, self.split, 'fault', name), dtype=np.single)
        # reshape into 128 * 128 * 128
        seis = seis.reshape((128, 128, 128))
        fault = fault.reshape((128, 128, 128))
        output = {'image': torch.from_numpy(seis).unsqueeze(0),
                    'label': torch.from_numpy(fault).unsqueeze(0),
                    'image_name': self.data_lst[index]}
        if self.split == 'train':
            return self.train_transform(output)
        elif self.split == 'val':
            return self.val_transform(output)


class Fault(Dataset):
    def __init__(self, 
                root_dir: str, 
                split: str = 'train',
                 ):
        self.root_dir = root_dir
        self.split = split
        self.train_transform = Compose([RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                                RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                                RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                                RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3, spatial_axes=(0, 1)),
                                NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True)
                                    # RandRotated(keys=["image", "label"], prob=0.10, )
                                    ])
        self.val_transform = NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True)
        # self.convert_size = convert_size
        if self.split == 'train':
            self.data_lst = os.listdir(os.path.join(self.root_dir, 'train'))
        elif self.split == 'val':
            self.data_lst = os.listdir(os.path.join(self.root_dir, 'val'))
        else:
            raise ValueError('Only support split = train or val')
        
    
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        f = h5py.File(os.path.join(self.root_dir, self.split, self.data_lst[index]),'r') 
        image = f['raw'][:]
        if 'label' in f.keys():
            mask = f['label'][:]
        else:
            mask = None
        # mask = np.squeeze(mask,0)
        f.close()
        if mask is None:
            return self.val_transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'image_name': self.data_lst[index]})
        elif self.split == 'train':
            return self.train_transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'label': torch.from_numpy(mask).unsqueeze(0),
                    'image_name': self.data_lst[index]})
        elif self.split == 'val':
            return self.val_transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'label': torch.from_numpy(mask).unsqueeze(0),
                    'image_name': self.data_lst[index]})

class Fault_Simple(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_lst = os.listdir(self.root_dir)
        self.transform = NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True)
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        image = segyio.tools.cube(os.path.join(self.root_dir, self.data_lst[index]))
        return self.transform({'image': torch.from_numpy(image).unsqueeze(0),
                                'image_name': self.data_lst[index]})




class FaultDataset(pl.LightningDataModule):
    def __init__(
        self,
        real_data_root_dir: str,
        simulate_data_root_dir=None,
        test_data_root_dir=None,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
    ):
        super().__init__()
        self.real_data_root_dir = real_data_root_dir
        self.simulate_data_root_dir = simulate_data_root_dir
        self.test_data_root_dir = test_data_root_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist



    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            if self.simulate_data_root_dir is not None:
                self.train_ds = ConcatDataset(
                    [Fault(root_dir=self.real_data_root_dir, split='train'),
                    Fault_Simulate(root_dir=self.simulate_data_root_dir, split='train')]
                    )
            else:
                self.train_ds = Fault(root_dir=self.real_data_root_dir, split='train')
            self.valid_ds = Fault(root_dir=self.real_data_root_dir, split='val')
          

        if stage in [None, "test"]:
            if self.test_data_root_dir is not None:
                self.test_ds = Fault_Simple(root_dir=self.test_data_root_dir)
            else:
                self.test_ds = Fault(root_dir=self.real_data_root_dir, split='val')

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
            pin_memory=True,
            shuffle=False,
            sampler=DistributedSampler(self.valid_ds, shuffle=False, drop_last=False),
            drop_last=False,
        )
        else:
            dataloader = torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
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


        
if __name__ == '__main__':
    data = FaultDataset(real_data_root_dir='/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/crop',
        simulate_data_root_dir=None,
        test_data_root_dir=None,
        batch_size=4,
        val_batch_size=1,
        num_workers= 4,
        dist=False,)
    data.setup()
    for item in data.train_dataloader():
        print(item['image'])
        break