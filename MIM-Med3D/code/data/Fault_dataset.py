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
    ToTensord,
)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data in a fixed range of [-1, 1].
    """

    def __init__(self, min_value, max_value, **kwargs):
        assert max_value > min_value
        self.min_value = min_value
        self.value_range = max_value - min_value

    def __call__(self, m):
        norm_0_1 = (m - self.min_value) / self.value_range
        return np.clip(2 * norm_0_1 - 1, -1, 1)


class Fault_Simulate(Dataset):
    def __init__(self,
                 root_dir,
                 split,
                 is_ssl=False):
        self.root_dir = root_dir
        self.split = split
        self.is_ssl = is_ssl
        self.base_transform = Normalize(min_value=-7, max_value=7)
        if not is_ssl:
            self.transform = Compose([RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                                        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                                        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                                        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3, spatial_axes=(0, 1)),
                                        # RandRotated(keys=["image", "label"], prob=0.10, )
                                        ])
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
        seis = self.base_transform(seis)
        output = {'image': torch.from_numpy(seis).unsqueeze(0),
                    'label': torch.from_numpy(fault).unsqueeze(0),
                    'image_name': self.data_lst[index]}
        if self.split == 'train' and not self.is_ssl:
            return self.transform(output)
        elif self.split == 'train' and self.is_ssl:
            return output
        elif self.split == 'val':
            return output


class Fault(Dataset):
    def __init__(self, 
                root_dir: str, 
                split: str = 'train',
                is_ssl=False
                 ):
        self.root_dir = root_dir
        self.split = split
        self.is_ssl = is_ssl
        self.base_transform = Normalize(min_value=-46924.76953125, max_value=55077.2109375)
        if not is_ssl:
            self.transform = Compose([RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10,),
                                        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10,),
                                        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10,),
                                        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3, spatial_axes=(0, 1))
                                        ])
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
        # apply base transform
        image = self.base_transform(image)
        if 'label' in f.keys():
            mask = f['label'][:]
        else:
            mask = None
        # mask = np.squeeze(mask,0)
        f.close()
        if mask is None:
            return {'image': torch.from_numpy(image).unsqueeze(0),
                    'image_name': self.data_lst[index]}
        elif self.split == 'train' and not self.is_ssl:
            return self.transform({'image': torch.from_numpy(image).unsqueeze(0),
                    'label': torch.from_numpy(mask).unsqueeze(0),
                    'image_name': self.data_lst[index]})
        elif self.split == 'train' and self.is_ssl:
            return {'image': torch.from_numpy(image).unsqueeze(0),
                    'label': torch.from_numpy(mask).unsqueeze(0),
                    'image_name': self.data_lst[index]}
        elif self.split == 'val':
            return {'image': torch.from_numpy(image).unsqueeze(0),
                    'label': torch.from_numpy(mask).unsqueeze(0),
                    'image_name': self.data_lst[index]}

class Fault_Simple(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_lst = os.listdir(self.root_dir)
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, index):
        image = segyio.tools.cube(os.path.join(self.root_dir, self.data_lst[index]))
        base_transform = Normalize(min_value=image.min(), max_value=image.max())
        image = base_transform(image)
        return {'image': torch.from_numpy(image).unsqueeze(0),
                'image_name': self.data_lst[index]}




class FaultDataset(pl.LightningDataModule):
    def __init__(
        self,
        real_data_root_dir: str,
        simulate_data_root_dir: str,
        test_data_root_dir: None,
        is_ssl,
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
        json_path = None,
        downsample_ratio=None
    ):
        super().__init__()
        self.real_data_root_dir = real_data_root_dir
        self.simulate_data_root_dir = simulate_data_root_dir
        self.test_data_root_dir = test_data_root_dir
        self.is_ssl = is_ssl
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist
        self.json_path = json_path
        self.downsample_ratio = downsample_ratio



    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            self.train_ds = ConcatDataset(
                [Fault(root_dir=self.real_data_root_dir, split='train', is_ssl=self.is_ssl),
                 Fault_Simulate(root_dir=self.simulate_data_root_dir, split='train', is_ssl=self.is_ssl)]
                )
            self.valid_ds = Fault(root_dir=self.real_data_root_dir, split='val', is_ssl=self.is_ssl)
          

        if stage in [None, "test"]:
            if self.test_data_root_dir is not None:
                self.test_ds = Fault_Simple(root_dir=self.test_data_root_dir)
            else:
                self.test_ds = Fault(root_dir=self.real_data_root_dir, split='val', is_ssl=self.is_ssl)

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
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )
        else:
            dataloader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            # sampler=DistributedSampler(self.train_ds),
            drop_last=False,
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )
            
        return dataloader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # sampler=DistributedSampler(self.valid_ds),
            drop_last=False,
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            # sampler=DistributedSampler(self.test_ds),
            drop_last=False,
            # collate_fn=pad_list_data_collate,
            # prefetch_factor=4,
        )


        
if __name__ == '__main__':
    data = FaultDataset(root_dir='/home/zhangzr/FaultRecongnition/Fault_data/real_labeled_data/crop',
                        batch_size=4)
    data.setup()
    for item in data.train_dataloader():
        print(item)
        # break