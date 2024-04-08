#!/usr/bin/env python
import abc
import collections
import math
import copyfrom typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os
from analysis.collector import DataCollectorfrom analysis.utils import normalize_trajectory
import type_utils.state_trajectory as state_traj
import paper_plot.utils as plot_utilsfrom utils.kmeans_torch import MyKmeansPytorch
from utils.file_io import write_dict2bin, read_dict_from_bin

class TrajectoryCollectorWithKmeansProcessor(DataCollector):
    '''
    Class to collect trajectories from different dataset, and process the kmeans algorihtm
    '''
    def __init__(self, key_mode: str=None,
                       cache_dir: str=None,
                       cache_batch_num: int=10000,
                       dt2sample_next_traj: float=3.0,
                       sample_duration: float=6.0,
                       sample_interval: float=0.5,
                       sparse_reso: float=0.5,
                       config_path: str=None,
                       save_dir: str=None):
      '''
      :param key_mode: key string for function design
      '''
      super().__init__(key_mode=key_mode, cache_dir=cache_dir, 
                       cache_batch_num=cache_batch_num,
                       config_path=config_path,
                       save_dir=save_dir)

      self.dt2sample_next_traj = dt2sample_next_traj
      self.sample_duration = sample_duration
      self.sample_interval = sample_interval
      self.sparse_reso = sparse_reso
      self.sparse_reso_2 = sparse_reso * 0.5

      self.traj_exists_keys = {}

      self.cache_file_name = 'epsilon'
      self.cache_exists = self.exists_at_cache_dir(self.cache_file_name)

      self.mode_functions = {
        'plot_elements': self.plot_trajs_list,
      }

    def enable_add_this_trajectory(self, traj: np.ndarray) -> bool:
      grid_traj = np.floor((traj + self.sparse_reso_2) / self.sparse_reso)
      grid_traj = np.array(grid_traj, dtype=np.int)
      tuple_list = tuple(grid_traj.tolist())

      grid_keys = (tuple(tp) for tp in tuple_list)
      grid_keys = tuple(grid_keys)

      flag = False
      if not grid_keys in self.traj_exists_keys:
        self.traj_exists_keys[grid_keys] = True
        flag = True

      return flag

    def add_data(self, input_data: Dict) -> bool:
      '''
      Add data to collector for analysis
      Return False if there is not need to add data
      '''
      local_trajs = input_data['local_traj']

      if not 'local_traj' in self.dict_data:
        self.dict_data['local_traj'] = []

      for traj_list in local_trajs:
        traj = state_traj.StateTrajectory()
        traj.set_trajectory_list(traj_list)
        array_traj = traj.numpy_trajecotry()

        traj_info = traj.get_info()

        frame0_s = traj_info.first_time_stamp_s

        traj_len = array_traj.shape[0]
        add_frame: int = int(self.dt2sample_next_traj / traj_info.time_interval_s)
        state_len: int = int(self.sample_duration / traj_info.time_interval_s)
        interval: int = int(self.sample_interval / traj_info.time_interval_s)

        frame0_list = range(1, array_traj.shape[0], add_frame)
        for frame0 in frame0_list:
          frame1 = frame0 + state_len
          if frame1 <= traj_len:
            seg_traj = array_traj[frame0:frame1:interval, :]
            seg_traj = normalize_trajectory(seg_traj)
            if self.enable_add_this_trajectory(seg_traj):
              # reduce from 12869 > 11137 when reso=0.2
              self.dict_data['local_traj'].append(seg_traj.tolist())
              self.data_num += 1

      return True

    def final_processing_data(self):
      self.dict_data['local_traj'] = np.array(self.dict_data['local_traj'])

      if self.cache_exists == True:
        print("Skip to process data, manually delete the cache data first.")
        return
      print('\nExtracted trajectory shape= {}'.format(self.dict_data['local_traj'].shape))

      initial_k: int = 2
      cluster_num_list: List[int] = [2048, 1024, 512, 256, 128, 50]
      trajectory_set = self.dict_data['local_traj'][:, :, :2] # (num, 12, 3) > (num, 12, 2)

      for cluster_num in cluster_num_list:
        if cluster_num >= trajectory_set.shape[0]:
          continue

        print("Process cluster_num={}.".format(cluster_num))
        # cluster_trajs, center_trajs, cluster_metric = [], [], {'max_se': 10.0}
        cluster_trajs, center_trajs, cluster_metric =\
          MyKmeansPytorch.run_kmeans(trajectory_set, cluster_num, device='cpu')

        write_data = {
          'cluster_trajs': cluster_trajs,
          'center_trajs': center_trajs,
          'cluster_metric': cluster_metric,
        }

        epsilon = cluster_metric['max_se']
        filename = "{}_{:.2f}_[{}].kmeans.bin".format(
          self.cache_file_name, epsilon, cluster_num)
        filepath = os.path.join(self.cache_dir, filename)
        
        write_dict2bin(write_data, filepath)

    def plot_data(self, dict_data: Dict, is_ordered: bool):
      plot_func = self.mode_functions[self.key_mode]
      plot_func(dict_data, is_ordered)

    def plot_trajs_list(self, dict_data: Dict, is_ordered: bool):
      file_list = self.read_cache_data_list(self.cache_file_name)

      for fname in file_list:
        fpath = os.path.join(self.cache_dir, fname)
        read_data:Dict = read_dict_from_bin(fpath, verbose=False)

        fig = plt.figure()
        plot_utils.fig_reset()
        for traj in dict_data['local_traj']:
          plt.plot(traj[:,0], traj[:,1], 'b')
        
        traj_num = 0 
        for traj in read_data['center_trajs']:
          plt.plot(traj[:,0], traj[:,1], 'y')
          traj_num += 1
        
        print("{} with traj num={}.".format(fname, traj_num))
        plt.show()

