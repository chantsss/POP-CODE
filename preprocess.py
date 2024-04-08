
#from argparse import ArgumentParser

import pytorch_lightning as plfrom pytorch_lightning.callbacks import ModelCheckpointfrom typing import List
import ray
import hydra
from datamodules.av1_datamodule import Av1DataModulefrom datamodules.av2_datamodule import Av2DataModulefrom models.hivt import HiVTfrom pathlib import Pathfrom tqdm import tqdmfrom datasets.av2_extractor_hivt import Av2ExtractorHiVTfrom datasets.av2_extractor_hivt import Extractorfrom utils.ray_utils import ActorHandle, ProgressBar

def glob_files(data_root: Path, mode: str):
    if data_root.stem == "av1":
        file_root = data_root / mode / "data"
        scenario_files = list(file_root.rglob("*.csv"))
    elif data_root.stem == "av2":
        file_root = data_root / mode
        scenario_files = list(file_root.rglob("*.parquet"))
    else:
        raise ValueError(f"Unsupported dataset: {data_root.stem}")

    return scenario_files

@ray.remote
def preprocess_batch(extractor: Extractor, file_list: List[Path], pb: ActorHandle):
    for file in file_list:
        extractor.save(file)
        pb.update.remote(1)

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--dataset', type=str, default='av1')
    parser.add_argument('--model_name', type=str, default='hivt')
    parser.add_argument('--modes', type=str, default=[])
    parser.add_argument('--save_folder', type=str, default='')
    parser.add_argument('--parallel', action="store_true")
    # parser = HiVT.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.dataset == 'av1':
        if args.model_name == 'hivt':
            print(f"modes to be preprocessed: {args.modes}")
            modes_list = args.modes.split(',')
            modes_list = [elem.strip() for elem in modes_list if elem.strip()]
            datamodule = Av1DataModule.from_argparse_args(args)
            datamodule.prepare_data()
        

    elif args.dataset == 'av2':

        if args.model_name == 'hivt':
            args.save_folder = 'hivt'
            batch = 50
            data_root = Path(args.root)

            print(f"modes to be preprocessed: {args.modes}")
            modes_list = args.modes.split(',')
            modes_list = [elem.strip() for elem in modes_list if elem.strip()]



            for mode in modes_list:
                save_path = data_root / args.save_folder / mode / 'processed'
                print(f"preprocessing {mode}set, save to {save_path}")
                extractor = Av2ExtractorHiVT(save_path=save_path, mode=mode)
                print(f"extractor: {extractor}")
                scenario_files = glob_files(data_root, mode)

                if args.parallel:
                    print("parallel mode")
                    pb = ProgressBar(len(scenario_files), f"preprocess {mode}-set")
                    pb_actor = pb.actor

                    for i in range(0, len(scenario_files), batch):
                        preprocess_batch.remote(
                            extractor, scenario_files[i : i + batch], pb_actor
                        )

                    pb.print_until_done()

                else:
                    for file in tqdm(scenario_files):
                        extractor.save(file)



