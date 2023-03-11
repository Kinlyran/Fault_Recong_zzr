from typing import Optional, Sequence, Union
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as ptdist
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from monai.data import MetaTensor
import h5py


class Standardize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, eps=1e-10, mean=None, std=None, channelwise=False, **kwargs):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.eps = eps
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=self.eps, a_max=None)

class Fault(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train',
                 # convert_size=(96,96,96)
                 ):
        self.root_dir = root_dir
        self.split = split
        self.transform = Standardize()
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
        # apply transform
        image = self.transform(image)
        mask = f['label'][:]
        # mask = np.squeeze(mask,0)
        f.close()
        return {'image': MetaTensor(image).unsqueeze(0),
                'label': MetaTensor(mask)}
        


class FaultDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        # convert_size: tuple = (96, 96, 96),
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
        json_path = None,
        downsample_ratio=None
    ):
        super().__init__()
        self.root_dir = root_dir
        # self.convert_size = convert_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist
        self.json_path = json_path
        self.downsample_ratio = downsample_ratio



    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            self.train_ds = Fault(root_dir=self.root_dir, split='train')
            self.valid_ds = Fault(root_dir=self.root_dir, split='val')
          

        if stage in [None, "test"]:
            self.test_ds = Fault(root_dir=self.root_dir, split='val')

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


        
