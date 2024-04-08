import picklefrom pathlib import Path
import torchfrom torch.utils.data import Dataset
from aug.aug_base import AugBase
# from extractors.extractor import Extractorfrom typing import Callable, Dict, List, Optional, Tuple, Unionfrom tqdm import tqdmfrom datasets.av2_common import load_av2_df, OBJECT_TYPE_MAPfrom datasets.av2_extractor_hivt import Extractor

class Av2dataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        cached_split: str = None,
        extend_trainval: bool = False,
        aug: AugBase = None,
        data_subset: float = 1.0,
        custom_split: str = None,
        model: str = 'hivt',
        extractor: Extractor = None, 
        ):

        if model != 'hivt':
            print('Attention, this is not hivt data extractor')

        if custom_split is not None:
            self.pickle = False
            self.load = True
            split_path = Path(data_root) / (cached_split + "_" + custom_split + ".pkl")
            with open(split_path, "rb") as f:
                self.file_list = pickle.load(f)
            self.file_list = sorted(
                [Path(data_root) / cached_split / f for f in self.file_list]
            )
            print(f"load {len(self.file_list)} files from {split_path}")

        elif cached_split is not None:
            self.pickle = True
            self.data_folder = Path(data_root) / cached_split/ "processed"
            self.file_list = sorted(list(self.data_folder.glob("*.pkl")))
            if len(self.file_list) == 0:
                self.file_list = sorted(list(self.data_folder.glob("*.pt")))
                self.pickle = False
            self.load = True

            if extend_trainval:
                file_type = "pkl" if self.pickle else "pt"
                self.file_list.extend(
                    sorted(list((Path(data_root) / "val").glob(f"*.{file_type}")))
                )

        elif extractor is not None:
            self.extractor = extractor
            self.data_folder = Path(data_root)
            print(f"Extracting data from {self.data_folder}")
            self.file_list = list(self.data_folder.rglob("*.parquet"))
            self.load = False

            if extend_trainval:
                val_root = Path(data_root).parent.parent
                self.file_list.extend(list((val_root / "val" / "data").glob("*.csv")))
        else:
            raise ValueError("Either data_folder or extractor must be specified")

        print(
            f"data root: {data_root}/{cached_split}, total number of files: {len(self.file_list)}, use trainval: {extend_trainval}"
        )

        self.aug = aug

        if data_subset < 1.0:
            print(
                f"Using only {data_subset * 100}% of the data, {len(self.file_list)} files"
            )
            self.file_list = self.file_list[: int(len(self.file_list) * data_subset)]
            print(f"New number of files: {len(self.file_list)}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, index: int):
        if self.load:
            if self.pickle:
                with open(self.file_list[index], "rb") as f:
                    data = pickle.load(f)
            else:
                data = torch.load(self.file_list[index])
        else:
            data = self.extractor.get_data(self.file_list[index])

        if self.aug is not None:
            data = self.aug.augment(data)

        return data

    def process(self) -> None:
        for raw_path in tqdm(self.file_list):
            self.extractor.save(raw_path)
