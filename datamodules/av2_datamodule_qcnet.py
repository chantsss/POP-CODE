from typing import Callable, Optional

import pytorch_lightning as plfrom torch_geometric.loader import DataLoader
from datasets.av2_dataset_qcnet import ArgoverseV2DatasetQCNetfrom transforms import TargetBuilder, ReconsTargetBuilder


class Av2DataModuleQCNet(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 test: bool = False,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 data_subset: float = 1,
                 **kwargs) -> None:
        super(Av2DataModuleQCNet, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.test_transform = test_transform
        self.test = test
        self.data_subset = data_subset

        self.train_transform = TargetBuilder(50, 60)
        self.val_transform = TargetBuilder(50, 60)
        # else:
        #     self.train_transform = ReconsTargetBuilder(50, 50)
        #     self.val_transform = ReconsTargetBuilder(50, 50)

    def prepare_data(self) -> None:
        ArgoverseV2DatasetQCNet(self.root, 'train', self.train_processed_dir, self.train_transform)
        ArgoverseV2DatasetQCNet(self.root, 'val', self.val_processed_dir, self.val_transform)
        ArgoverseV2DatasetQCNet(self.root, 'test', self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = ArgoverseV2DatasetQCNet(self.root, 'train', self.train_processed_dir,
                                                    transform=self.train_transform, data_subset=self.data_subset)
            self.val_dataset = ArgoverseV2DatasetQCNet(self.root, 'val', self.val_processed_dir,
                                              transform=self.val_transform)
        else:
            self.test_dataset = ArgoverseV2DatasetQCNet(self.root, 'test', self.test_processed_dir,
                                               transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

