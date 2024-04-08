import osfrom typing import Dict
import torch
import torch.utils.data as torch_data
import random

import envs.configfrom utils.file_io import extract_folder_file_list, read_dict_from_bin

class PriorityDataset(torch_data.Dataset):
  def __init__(self, version: str, args: Dict):
    '''
    Interface to read priority prediction dataset data
    :param version: value belongs to ['train', 'eval']
    :param args: dict of args to process the dataset
    '''
    assert version in ['train', 'eval'], "Error, version should be inside ['train', 'eval']"

    EXP_ROOT = {
      'interaction_dataset': envs.config.INTERACTION_EXP_ROOT,
      'commonroad': envs.config.COMMONROAD_EXP_ROOT,
    }

    self.data_folder = os.path.join(EXP_ROOT[args['dataset']], 'priority_prediction')
    file_list = extract_folder_file_list(self.data_folder)

    file_list = sorted(file_list, key=lambda p: int(p.split('_')[0]))

    total_file_num :int= len(file_list)
    train_prop :float= args['dataset_config']['train_proportion']
    eval_prop :float= args['dataset_config']['eval_proportion']
    assert (0.0 <= train_prop) and (train_prop <= 1.0),\
      "Parameter error, train_prop should be inside [0.0, 1.0]"

    self.train_file_num :int = round(total_file_num * train_prop)
    self.eval_file_num :int = round(total_file_num * eval_prop)
    
    if version == 'train':
      self.file_list = file_list[0:self.train_file_num]
    else:
      # version == 'eval':
      self.file_list = file_list[(total_file_num - self.eval_file_num):-1]

  def __len__(self) -> int:
    return len(self.file_list)

  def __getitem__(self, idx):
    '''
    :param idx: index of map.csv file
    '''
    filepath = os.path.join(self.data_folder, self.file_list[idx])

    return read_dict_from_bin(filepath, verbose=False)
