from typing import Optional, Sequence, Union
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as ptdist
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import h5py
import numpy as np
from monai.data import MetaTensor

class ScanObjectNN(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'main_split',
                 convert_size: tuple = (96, 96, 96),
                 is_train: bool = True
                 ):
        self.root_dir = root_dir
        self.split = split
        self.convert_size = convert_size
        
        # size of point cloud is num_data * 2048 * 3
        if is_train:
            f = h5py.File(os.path.join(self.root_dir, self.split, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.pointClouds = f['data'][:] 
            f.close()
        else:
            f = h5py.File(os.path.join(self.root_dir, self.split, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.pointClouds = np.array(f['data'][:]).astype(np.float64)
            f.close()
            
    
    def __len__(self):
        return self.pointClouds.shape[0]
    
    def __getitem__(self, index):
        pointCloud = self.pointClouds[index, :, :]
        cubeImage = self.convert(pointCloud, self.convert_size)
        return {'image': cubeImage}
    
    def convert(self, pointCloud, size):
        cubeImage = torch.zeros((size))
        normalPointCloud = (pointCloud - pointCloud.min()) / (pointCloud.max() - pointCloud.min())
        scalePointCloud = (normalPointCloud * (size[0]-1)).astype(np.uint8)
        for i in range(scalePointCloud.shape[0]):
            cord = scalePointCloud[i,:]
            cubeImage[cord[0], cord[1], cord[2]] = 1.0
        # add channel: (96, 96, 96) -> (1, 96, 96, 96)
        cubeImage = cubeImage.unsqueeze(0)
        return MetaTensor(cubeImage)
        


class ScanObjDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        split: str = 'main_split',
        convert_size: tuple = (96, 96, 96),
        batch_size: int = 1,
        val_batch_size: int = 1,
        num_workers: int = 4,
        dist: bool = False,
        json_path = None,
        downsample_ratio=None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.convert_size = convert_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.dist = dist
        self.json_path = json_path
        self.downsample_ratio = downsample_ratio



    def setup(self, stage: Optional[str] = None):
        # Assign Train split(s) for use in Dataloaders
        if stage in [None, "fit"]:
            self.train_ds = ScanObjectNN(root_dir=self.root_dir, split=self.split, convert_size=self.convert_size, is_train=True)
            self.valid_ds = ScanObjectNN(root_dir=self.root_dir, split=self.split, convert_size=self.convert_size, is_train=False)
          

        if stage in [None, "test"]:
            self.test_ds = ScanObjectNN(root_dir=self.root_dir, split=self.split, convert_size=self.convert_size, is_train=False)

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
