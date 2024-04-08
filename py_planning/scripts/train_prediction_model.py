import argparse
import yamlfrom train_eval.trajectory_prediction.train_eval import TrainAndEvaluatorfrom train_eval.trajectory_prediction.train_eval_hivt import TrainAndEvaluatorHiVTfrom torch.utils.tensorboard import SummaryWriter
import os
import torch
import numpy as np
import random

import envs.configfrom utils.time import get_date_str

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model_path(dataset_name: str, model_name: str):
  '''
  Return the path to store the processed data of the corresponding dataset
  :param dataset_name: the type of dataset to store
  :param process_mode: a arbitary string to identify the mode 
  '''
  folder_name = 'models/' + model_name+'_{}'.format(get_date_str())
  path = envs.config.get_dataset_exp_folder(dataset_name, folder_name)

  return path

setup_seed(2023)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Config file with dataset parameters", required=True)
parser.add_argument("-m", "--model_name", help="name of model for identification", required=True)
parser.add_argument("-n", "--num_epochs", help="Number of epochs to run training for", required=True)
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=False)
parser.add_argument("-j", "--just_weights", help="Just load weights", action='store_true')
parser.add_argument("-l", "--ld", help="Load for Distill or not", action='store_true')
args = parser.parse_args()

# Load config
output_dir = None
with open(args.config, 'r') as yaml_file:
  cfg = yaml.safe_load(yaml_file)
  output_dir = get_model_path(cfg['dataset'], args.model_name)

  # Make directories
  if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
  if not os.path.isdir(os.path.join(output_dir, 'checkpoints')):
    os.mkdir(os.path.join(output_dir, 'checkpoints'))
  if not os.path.isdir(os.path.join(output_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(output_dir, 'tensorboard_logs'))

  # Save config file to model path
  with open(os.path.join(output_dir, 'config.yaml'), 'w') as wfile:
    documents = yaml.dump(cfg, wfile)

# Initialize tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard_logs'))

# Train
if 'hivt' in args.model_name:
  teval_tool = TrainAndEvaluatorHiVT(cfg, checkpoint_path=args.checkpoint, writer=writer, just_weights=args.just_weights, ld=args.ld)
else:
  teval_tool = TrainAndEvaluator(cfg, checkpoint_path=args.checkpoint, writer=writer)
# 

teval_tool.train(num_epochs=int(args.num_epochs), output_dir=output_dir)

# Close tensorboard writer
writer.close()
