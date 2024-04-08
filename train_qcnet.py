
from argparse import ArgumentParser
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import importlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from copy import copy
from datamodules import Av1DataModule
from datamodules import Av2DataModule
from datamodules import Av2DataModuleQCNet
from models.hivt import HiVT
from models.hivt_lite import HiVTLite
from models.hivt_plus import HiVTPlus
from models.hivt_lite_recons import HiVTLiteRcons
from models.hivt_lite_distill import HiVTLiteDistill
from models.hivt_lite_mtask import HiVTLiteMTask
from models.qcnet.qcnet import QCNet
from models.qcnet_lite.qcnet_lite import QCNetLite
from models.qcnet_distill.qcnet_distill import QCNetDistill
from models.qcnet_recons.qcnet_recons import QCNetRecons
from pytorch_lightning.strategies.ddp import DDPStrategy

import yaml
import random
import numpy as np
import torch

@hydra.main(config_path=".", config_name="config")

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # pl.seed_everything(seed)
    # # Set random seed

class EmptyCacheCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def on_test_epoch_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

def parse_arguments():
    parser = ArgumentParser(description='Example Argument Parser')

    # Add command line arguments
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=48)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--max_epochs', type=int, default=64)
    parser.add_argument('--monitor', type=str, default='val_minADE', choices=['val_minADE_recons', 'val_minFDE_refine', \
                        'val_minADE_refine', 'val_minADE', 'val_minFDE', 'val_minMR'])
    parser.add_argument('--save_top_k', type=int, default=1000)
    parser.add_argument('--reduce_his_length', type=bool, default=False)
    parser.add_argument('--random_his_length', type=bool, default=False)
    parser.add_argument('--random_interpolate_zeros', type=bool, default=False)
    parser.add_argument('--valid_observation_length', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--recons_model_path', type=str, default='')
    parser.add_argument('--init_model_path', type=str, default='')
    parser.add_argument('--file_name', type=str, default='')
    parser.add_argument('--dataset', type=str, default='av1')
    parser.add_argument('--model_name', type=str, default='hivt', help='hivt, hivt_lite, hivt_plus')
    parser.add_argument('--collate_fn', type=str, default='av2_hivt')
    parser.add_argument('--model_config', type=str, default='', help='Path to YAML configuration file')
    parser.add_argument('--data_module_config', type=str, default='', help='Path to YAML configuration file')
    parser.add_argument('--data_subset', type=float, default=1, help='Using only a part of the full data')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='if you want to do validation every n epoch')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='if you want to do backward every n batches')
    parser.add_argument('--teacher_ckpt_path', type=str, default='', help='if you want to build distill framework')
    parser.add_argument('--build_recons_target', type=bool, default='', help='if you want to build recons target')
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)

    args = parser.parse_args()

    return args


def load_model_config(args):

    print('loading config from ', args.model_config)
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    config = copy(model_config)
    for key in ['lr', 'weight_decay']:
        model_config[key] = float(model_config[key])
   
    if 'submission' in model_config:
        config = OmegaConf.create(model_config['submission'])
        del model_config['submission']

    return model_config, config


if __name__ == '__main__':

    seed = 2023
    setup_seed(seed)    
    args = parse_arguments()
    model_config, config = load_model_config(args)
    # data_module_config = load_datamodule_config(args)
    if 'submission_handler' in config:
        module_name, class_name = config['submission_handler'].rsplit('.', 1)
        module = importlib.import_module(module_name)
        submission_handler = getattr(module, class_name)
        submission_handler = submission_handler(save_dir='./', filename=args.model_name)
    
    if args.model_name == 'hivt':
        model = HiVT(**model_config)
    
    elif args.model_name == 'hivt_lite':
        model = HiVTLite(**model_config)

    elif args.model_name == 'hivt_plus':
        model = HiVTPlus(**model_config)

    elif args.model_name == 'hivt_recons':
        model = HiVTLiteRcons(**model_config, submission_handler=submission_handler)

    elif args.model_name == 'hivt_distill':
        model = HiVTLiteDistill(**model_config)

    elif args.model_name == 'hivt_mtask':
        model = HiVTLiteMTask(**model_config)

    elif args.model_name == 'qcnet':
        model = QCNet(**model_config)

    elif args.model_name == 'qcnet_lite':
        model = QCNetLite(**model_config)

    elif args.model_name == 'qcnet_distill':
        model = QCNetDistill(**model_config)

    elif args.model_name == 'qcnet_recons':
        model = QCNetRecons(**model_config)

    if args.recons_model_path != '':
        # Load pre-trained weights
        pretrained_dict = torch.load(args.recons_model_path)['state_dict']
        model_dict = model.state_dict()

        # Filter out unnecessary parts from pre-trained weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Update model weights
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('--------------loaded recons_model checkpoint from--------------', args.recons_model_path)

    if args.init_model_path != '':
        # Load pre-trained weights
        pretrained_dict = torch.load(args.init_model_path)['state_dict']
        model_dict = model.state_dict()

        # Filter out unnecessary parts from pre-trained weights
        pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k not in model_dict}

        if pretrained_dict_ != {}:
            print('Note init_model has unmatched para, replacing decoder to decoder_init')
            for k,v in pretrained_dict_.items():
                new_k = k.replace('decoder', 'decoder_init')
                pretrained_dict[new_k] = pretrained_dict.pop(k)

        pretrained_dict_in = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # Update model weights
        model_dict.update(pretrained_dict_in)
        model.load_state_dict(model_dict)
        print('loaded init_model checkpoint from', args.init_model_path)
        if len(pretrained_dict_in) != len(pretrained_dict):
            print('Note after replacement init_model still has unmatched para')

    if args.model_path != '' and (args.resume != True or args.test != True):
        # Load pre-trained weights
        model_dict_ = torch.load(args.model_path)
        pretrained_dict = model_dict_['state_dict']
        model_dict = model.state_dict()

        # Filter out unnecessary parts from pre-trained weights
        pretrained_dict_ = {k: v for k, v in pretrained_dict.items() if k not in model_dict}

        pretrained_dict_in = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        # Update model weights
        model_dict.update(pretrained_dict_in)
        model.load_state_dict(model_dict)
        print('loaded model checkpoint from', args.model_path)
        del model_dict_
        del model_dict
        if len(pretrained_dict_in) != len(pretrained_dict):
            print('Note after replacement model still has unmatched para')
        

    if args.model_path != '' and args.file_name != '':
        model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, \
                                           mode='min', dirpath=args.model_path, filename=args.file_name)
    else:
        model_checkpoint = ModelCheckpoint(monitor=args.monitor, save_top_k=args.save_top_k, mode='min', save_last=True)
    
    empty_cache_callback = EmptyCacheCallback()

    # model.recons = False
    # model.distill = False
    if args.resume:
        model.distill = False
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint, empty_cache_callback], accelerator='gpu',
                                                strategy = DDPStrategy(find_unused_parameters=False),
                                                resume_from_checkpoint=args.model_path, precision=16)
    else:
        trainer = pl.Trainer.from_argparse_args(args, callbacks=[model_checkpoint, empty_cache_callback], accelerator='gpu',
                                                strategy = DDPStrategy(find_unused_parameters=True), precision=16)

    # trainer.current_epoch = model_dict_['epoch']
    # trainer.global_step = model_dict_['global_step']
    # trainer.lr_scheduler_configs = model_dict_['lr_schedulers']

    if args.dataset == 'av1':
        datamodule = Av1DataModule.from_argparse_args(args, test=args.test)
    if args.dataset == 'av2':
        datamodule = Av2DataModule.from_argparse_args(args, test=args.test)
    if args.dataset == 'av2qcnet':
        datamodule = Av2DataModuleQCNet.from_argparse_args(args, test=args.test)

    # trainer.validate(model, datamodule)
    if args.test or args.eval:
        if hasattr(model, 'freeze_recons'):
            model.freeze_recons = True
        if hasattr(model, 'distillation'):
            model.distillation = False
        if hasattr(model, 'distill'):
            model.distillation = False
        
        if hasattr(model, 'reduce_his_length'):
            model.reduce_his_length = args.reduce_his_length
            if model.reduce_his_length:
                print('Reduce_his_length')

        if hasattr(model, 'reduce_his_length'): 
            model.random_his_length = args.random_his_length

        if hasattr(model, 'random_interpolate_zeros'): 
            model.random_interpolate_zeros = args.random_interpolate_zeros
            if model.random_his_length:
                print('Random_his_length')

        if hasattr(model, 'valid_observation_length'): 
            model.valid_observation_length = args.valid_observation_length
            print('Fix valid_observation_length at ', model.valid_observation_length)
       
        if hasattr(model, 'random_interpolate_zeros'): 
            if model.random_interpolate_zeros:
                print('Random_interpolate_zeros')

        if args.test:
            trainer.test(model, datamodule, ckpt_path=args.model_path)
        else:
            trainer.validate(model, datamodule, ckpt_path=args.model_path)        
        # trainer.validate(model, datamodule)
    else:
        # if args.model_path != '':
        #     trainer.fit(model, datamodule)
        # else:
        # model.recons = False
        # model.distill = True
        # model.load_teacher()
        trainer.fit(model, datamodule)
