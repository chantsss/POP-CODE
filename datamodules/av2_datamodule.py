from importlib import import_module
# import torchfrom pytorch_lightning import LightningDataModule
# from torch_geometric.data import DataLoader
# from torch_geometric.loader import DataLoader as PyGDataLoader
from aug.aug_base import AugBasefrom datasets.av2_dataset import Av2dataset
# from datasets.av2_pyg_dataset import Av2PyGDatasetfrom typing import Callable, Optionalfrom torch.utils.data import DataLoader as TorchDataLoader

class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        root: str,
        model_name: str,
        modes: list = [],
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        aug: AugBase = None,
        persistent_workers: bool = True,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        radius: int = 100,
        test: bool = False,
        expand_trainval: bool = False,
        collate_fn: str = None,
        data_subset=1.0,
        custom_split: str = None,
    ):
        super(Av2DataModule, self).__init__()
        self.root = root
        self.batch_size = train_batch_size  # for lightning's batch_size finder
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.model_name = model_name
        self.radius = radius
        self.aug = aug
        self.test = test
        self.expand_trainval = expand_trainval
        self.data_subset = data_subset
        self.custom_split = custom_split
        self.Dataset = Av2dataset 
        self.DataLoader = TorchDataLoader 
        self.modes = modes

        if collate_fn is not None:
            module = import_module(f"datamodules.collate.{collate_fn}")
            self.collate_fn = getattr(module, "collate_fn")
        else:
            self.collate_fn = None

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            self.train_dataset = self.Dataset(
                data_root=self.root + '/hivt',
                cached_split="train",
                extend_trainval=self.expand_trainval,
                aug=self.aug,
                data_subset=self.data_subset,
                custom_split=self.custom_split,
            )
            self.val_dataset = self.Dataset(
                data_root=self.root + '/hivt',
                cached_split="val" ,
                aug=None,
                custom_split=self.custom_split,
            )
        else:
            self.test_dataset = self.Dataset(
                data_root=self.root + '/hivt',
                cached_split="test",
                aug=None,
            )

    def train_dataloader(self):
        return self.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return self.DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,

        )

    def test_dataloader(self):
        return self.DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
    

