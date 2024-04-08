#!/usr/bin/env python
import abc
import collections
import osfrom typing import Dict, List, Any
import warnings
import yaml
import pickle
from utils.file_io import write_dict2bin, extract_folder_file_list

class DataCollector:
    '''
    Class to collect common datas from different dataset
    '''
    def __init__(self, key_mode: str, cache_dir: str, cache_batch_num: int,
                       config_path: str=None, save_dir: str=None):
      '''
      :param key_mode: which mode to performance
      :param cache_dir: directory to cache temporary data
      :param cache_batch_num: cache data data number of one file
      :param config_path: path to open config
      :param save_dir: directory to save results
      '''
      self.reinit()

      self.key_mode = key_mode
      self.cache_dir = cache_dir
      self.cache_file_name = None
      self.cache_batch_num = cache_batch_num

      # conf
      self.args_dict = {}
      if config_path:
        with open(config_path) as config_file:
          self.args_dict = yaml.safe_load(config_file)

      self.save_path = save_dir

      # grid parameters
      self._speed_reso = 0.5
      self._speed_reso_2 = self._speed_reso * 0.5

    def reinit(self) -> None:
      '''
      Reinit the class
      '''
      self.data_num: int = 0        # number of data being collected
      self.dict_data: Dict = {}     # shared collecting data

      self.write_file_id: int = 0   # index of file of cache data
      self.write_data_num: int = 0  # number of data being writed

    def exists_at_cache_dir(self, cache_file_name: str):
      '''
      check whether file named 'cache_file_name' exists in the self.cache_dir folder
      '''
      if self.cache_dir == None:
        return False
      if cache_file_name == None:
        return False

      flag = False
      file_list = extract_folder_file_list(self.cache_dir)
      for fname in file_list:
        if cache_file_name in fname:
          flag = True
          break

      if flag:
        print("Cache file exists at {}.".format(self.cache_dir))

      return flag

    @abc.abstractclassmethod
    def add_data(self, input_data: Dict) -> bool:
      '''
      Add data to collector for analysis
      Return False if there is not need to add data
      '''
      raise NotImplementedError()

    @abc.abstractclassmethod
    def final_processing_data(self) -> None:
      '''
      Final step / process data after add_data() operations / see detials in analyze_processed_dataset.py
      '''
      raise NotImplementedError()

    @abc.abstractclassmethod
    def plot_data(self, ordered_dict_data: Dict, is_ordered: bool):
      '''
      Last step to rviz data
      '''
      raise NotImplementedError()

    def try_write_cache_data(self, file_name:str, 
                                   forcibly_write:bool =False) -> None:
      '''
      Check if enable write the data with 'file_name' to self.cache_dir folder
      :param forcibly_write: if == true, forcibly enable to write. Else, this function will
        enable writing if data_num being writed large than self.cache_batch_num
      '''
      if self.cache_dir:
        data_num2write = (self.data_num - self.write_data_num)
        enable_write: bool = data_num2write >= self.cache_batch_num
        enable_write = enable_write or (forcibly_write and (data_num2write > 0))
        if enable_write:
          self.write_data_num = self.data_num
          fname = os.path.join(self.cache_dir, "{}_{}.bin".format(file_name, self.write_file_id))
          write_dict2bin(self.dict_data, fname, verbose=False)
          
          print("\nWith {} number of data writed.".format(self.write_data_num))
          self.dict_data.clear() # clear data
          self.write_file_id += 1
      else:
        warnings.warn("cache_dir is None, fails to write cache data.")

    def read_cache_data_list(self, file_name:str) -> List:
      '''
      Return list of file names exsisting in self.cache_dir
      '''
      if self.cache_dir:
        return extract_folder_file_list(self.cache_dir)
      return []

    def save_data(self, filename: str, data: Dict):
      '''
      Save data to save_path with file name = filename
      '''
      filepath = os.path.join(self.save_path, filename)
      with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename: str) -> Dict:
      """
      Function to load saved data from save_path.
      :param idx: data index
      :return data: Dictionary with saved data
      """
      filepath = os.path.join(self.save_path, filename)
      if not os.path.isfile(filepath):
        raise Exception('Could not find data. Please run the dataset in extract_data mode')

      with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
      return data

    def plot(self, is_ordered: bool=True):
      '''
      Port function to plot the data
      '''
      plot_dict_data = self.dict_data
      if is_ordered:
        plot_dict_data = collections.OrderedDict(sorted(self.dict_data.items()))

      self.plot_data(plot_dict_data, is_ordered)

    def debug_print(self, front_extra_str: str) -> None:
      '''
      Debug print out.
      '''
      print("\rDataCollector: {} , with key num={}, data num={}.".format(
          front_extra_str, len(self.dict_data.keys()), self.data_num), end="")
