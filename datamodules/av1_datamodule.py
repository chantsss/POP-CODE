
#
from typing import Callable, Optionalfrom pytorch_lightning import LightningDataModulefrom torch_geometric.data import DataLoaderfrom aug.aug_base import AugBasefrom datasets.av1_dataset import Av1dataset


class Av1DataModule(LightningDataModule):

    def __init__(self,
                 root: str,
                 model_name: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test: bool = False,
                 shuffle: bool = True,
                 aug: AugBase = None,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 test_transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 data_subset=1.0) -> None:
        super(Av1DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.local_radius = local_radius
        self.aug = aug
        self.model_name = model_name
        self.data_subset = data_subset
        self.test = test 
    def prepare_data(self) -> None:
        if not self.test:
            Av1dataset(self.root, self.model_name, 'train', self.train_transform, self.local_radius)
            Av1dataset(self.root, self.model_name, 'val', self.val_transform, self.local_radius)
        else:
            Av1dataset(self.root, self.model_name, 'test', self.test_transform, self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = Av1dataset(root=self.root, model = self.model_name, split='train', aug=self.aug, \
                                        data_subset=self.data_subset, transform=self.train_transform, local_radius=self.local_radius)
            self.val_dataset = Av1dataset(root=self.root, model = self.model_name, split='val', aug=None, \
                                        data_subset=1., transform=self.val_transform, local_radius=self.local_radius)

        else:
            self.test_dataset = Av1dataset(root=self.root, model = self.model_name, split='test', aug=None, \
                                        data_subset=1., transform=self.val_transform, local_radius=self.local_radius)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)
