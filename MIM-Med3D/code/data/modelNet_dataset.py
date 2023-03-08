from typing import Optional, Sequence, Union
import os
import torch
from torch.utils.data import Dataset
import torch.distributed as ptdist
import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from monai.data import MetaTensor

class ModelNet(Dataset):
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train',
                 convert_size: tuple = (96, 96, 96),
                 ):
        self.root_dir = root_dir
        self.split = split
        self.convert_size = convert_size
        
        # size of point cloud is num_sample_point * 3
        self.point_lst = []
        if self.split == 'train':
            with open(os.path.join(self.root_dir, 'modelnet40_train.txt'), 'r') as f:
                raw_point_lst = f.readlines()
            for point_cloud in raw_point_lst:
                point_cloud = point_cloud.strip('\n') + '.txt'
                class_name = point_cloud[:-9]
                self.point_lst.append(os.path.join(self.root_dir, class_name, point_cloud))
        elif self.split == 'test':
            with open(os.path.join(self.root_dir, 'modelnet40_test.txt'), 'r') as f:
                raw_point_lst = f.readlines()
            for point_cloud in raw_point_lst:
                point_cloud = point_cloud.strip('\n') + '.txt'
                class_name = point_cloud[:-9]
                self.point_lst.append(os.path.join(self.root_dir, class_name, point_cloud))
        else:
            raise ValueError('Only support split = train or test!')  
    
    def __len__(self):
        return len(self.point_lst)
    
    def __getitem__(self, index):
        with open(self.point_lst[index], 'r') as f:
            raw_pointCloud = f.readlines() # x, y, z, dx, dy, dz
        pointCloud = []
        for point in raw_pointCloud:
            point_lst = point.split(',')
            pointCloud.append(np.array([float(point_lst[0]), float(point_lst[1]), float(point_lst[2])]))
        pointCloud = np.stack(pointCloud, axis=0)
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
        


class ModelNetDataset(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
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
            self.train_ds = ModelNet(root_dir=self.root_dir, split='train', convert_size=self.convert_size)
            self.valid_ds = ModelNet(root_dir=self.root_dir, split='test', convert_size=self.convert_size)
          

        if stage in [None, "test"]:
            self.test_ds = ModelNet(root_dir=self.root_dir, split='test', convert_size=self.convert_size)

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
