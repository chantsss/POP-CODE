import os
import abcfrom typing import Dict, Any
import torch
import torch.utils.data as torch_data
from envs.config import INTERACTION_DATA_PLAN_ROOT, INTERACTION_DATA_PREDICTION_ROOT, INTERACTION_EXP_ROOT
import envs.interaction_dataset.prediction_io as intersect_dataset_reader_challenge

class InteractionDataset(torch_data.Dataset):
  def __init__(self, branch:str = 'prediction',
                     data_path: str='train',
                     config_path: str=None,
                     save_path: str=None,
                     file_data_num: int=100):
    '''
    Interface to read interaction dataset data
    :param branch: interaction dataset has two formats of data for planning and prediction respectively,
                   here we use branch = 'prediction' or 'plan' to indicates which format of data is to
                   read.
    :param data_path: read the train or val data_path in dataset. {'train', 'val'}.
    :param config_path: path to read config file
    :param save_path: path to save results
    :param file_data_num: batch number to process samples data
    @note, for interaction dataset, data len corresponds to the csv file number
           but not the sample number
    '''    
    map2branch_folder: Dict[str, str] = {
      'prediction': os.path.join(INTERACTION_DATA_PREDICTION_ROOT, 'INTERACTION-Dataset-DR-multi-v1_2'),
    }
    map2file_end_str: Dict[str, str] = {
      'train': '_train.csv',
      'val': '_val.csv',
    }

    self.branch = branch
    self.maps_folder = os.path.join(map2branch_folder[branch], "maps")    
    self.tracks_folder = os.path.join(map2branch_folder[branch], data_path)

    self.file_end_str = map2file_end_str[data_path]
    self.split_str = '_' + data_path

    # each map.csv has cases of data
    self.csv_files = os.listdir(self.tracks_folder)

    self.save_path = save_path
    self.file_data_num = file_data_num

  def __len__(self) -> int:
    return len(self.csv_files)

  def __getitem__(self, idx):
    '''
    :param idx: index of map.csv file
    '''
    filename = self.csv_files[idx]
    # split the file name to get the scenario name
    split_strs = filename.split(self.split_str)

    get_data = None
    if len(split_strs) > 0:
      scenario_name = split_strs[0]
      csv_file = os.path.join(
        self.tracks_folder, scenario_name + self.file_end_str)

      if self.branch == 'prediction':
        cases_dict = intersect_dataset_reader_challenge.read_cases(csv_file)
        get_data = self.process_data(
          idx, scenario_name, self.file_data_num, cases_dict=cases_dict)
      elif self.branch == 'plan':
        raise NotImplementedError()
      else:
        raise ValueError("Unexpected branch={}.".format(self.branch))

    # default not return data for training
    if get_data == None:
      return 0
    else:
      return 0 # TODO

  @abc.abstractmethod
  def process_data(self, idx: int, scenario_name: str,
                         file_data_num: int, **kwargs):
    '''
    Process data given index of map.csv
    :param idx: index of map.csv in dataset folder
    '''
    raise NotImplementedError()
