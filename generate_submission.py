
#from argparse import ArgumentParser

import pytorch_lightning as plfrom torch_geometric.data import DataLoader
from datasets.av1_dataset import Av1datasetfrom models.hivt import HiVTfrom models.hivt_lite import HiVTLitefrom models.hivt_plus import HiVTPlusfrom models.hivt_lite_recons import HiVTLiteRconsfrom datamodules.av1_datamodule import Av1DataModulefrom datamodules.av2_datamodule import Av2DataModule, Av2DataModuleQCNetfrom submission.submission_av2 import SubmissionAv2
import os

if __name__ == '__main__':
    pl.seed_everything(2022)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--valid_observation_length', type=int, default=20)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--reduce_his_length', action="store_true")
    parser.add_argument('--random_his_length', action="store_true")
    parser.add_argument('--random_interpolate_zeros', action="store_true")
    parser.add_argument('--dataset', type=str, default='av1')
    parser.add_argument('--model_name', type=str, default='hivt')
    parser.add_argument('--collate_fn', type=str, default='av2_hivt')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    args = parser.parse_args()

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        max_epochs=1,
        # limit_test_batches=conf.limit_test_batches,
    )
    
    checkpoint = args.ckpt_path
    print(checkpoint)
    assert os.path.exists(checkpoint), f"Checkpoint {checkpoint} does not exist"

    sub_save_dir = os.path.dirname(checkpoint)
    submission_handler = SubmissionAv2(save_dir=sub_save_dir, filename=args.dataset+'submission')

    if args.model_name == 'hivt':
        model = HiVT.load_from_checkpoint(checkpoint_path=checkpoint, submission_handler=submission_handler,pretrained_weights=None)

    elif args.model_name == 'hivt_lite':
        model = HiVTLite.load_from_checkpoint(checkpoint_path=checkpoint, submission_handler=submission_handler,pretrained_weights=None)

    elif args.model_name == 'hivt_plus':
        model = HiVTPlus.load_from_checkpoint(checkpoint_path=checkpoint, submission_handler=submission_handler,pretrained_weights=None)

    elif args.model_name == 'hivt_recons':
        model = HiVTLiteRcons.load_from_checkpoint(checkpoint_path=checkpoint, submission_handler=submission_handler,pretrained_weights=None)


    if args.dataset == 'av1':
        datamodule = Av1DataModule.from_argparse_args(args, test=True)
    if args.dataset == 'av2':
        datamodule = Av2DataModule.from_argparse_args(args, test=True)
    if args.dataset == 'av2qcnet':
        datamodule = Av2DataModuleQCNet.from_argparse_args(args, test=True)

    trainer.test(model, datamodule)
