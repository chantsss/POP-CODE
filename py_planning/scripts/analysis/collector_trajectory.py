#!/usr/bin/env python
import abc
import collections
import math
import copyfrom typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import os
from analysis.collector import DataCollector
import type_utils.agent as agent_utils
import type_utils.state_trajectory as state_traj
import paper_plot.utils as plot_utils

class TrajectoryCollector(DataCollector):
    '''
    Class to collect trajectories from different dataset
    '''
    def __init__(self, key_mode: str=None,
                       filter_agent_type_ids: List[int]=[],
                       filter_v0_lowerbound: float=0.0,
                       filter_v0_upperbound: float=1e+3,
                       cache_dir: str=None,
                       cache_batch_num: int=10000,
                       config_path: str=None,
                       save_dir: str=None):
      '''
      :param key_mode: key string for function design
      :param filter_agent_type_ids: remove agent ids in this list
      :param filter_v0_lowerbound: remove agent v0 < this value
      :param filter_v0_upperbound: remove agent v0 > this value
      '''
      super().__init__(key_mode=key_mode, cache_dir=cache_dir, 
                       cache_batch_num=cache_batch_num,
                       config_path=config_path, 
                       save_dir=save_dir)
      
      self.filter_agent_type_ids = filter_agent_type_ids
      self.filter_v0_lowerbound = filter_v0_lowerbound
      self.filter_v0_upperbound = filter_v0_upperbound

      self.mode_functions = {
        'plot_elements': self.plot_trajs_list,
      }

    def add_data(self, input_data: Dict) -> bool:
      '''
      Add data to collector for analysis
      Return False if there is not need to add data
      '''
      local_trajs = input_data['local_traj']
      global_trajs = input_data['global_traj']

      for traj_list in local_trajs:
        traj = state_traj.StateTrajectory()
        traj.set_trajectory_list(traj_list)
        array_traj = traj.numpy_trajecotry()

        agent_id = traj.get_info().agent_type
        start_v = traj.state_value(0, 'velocity')
        if agent_id in self.filter_agent_type_ids:
          continue
        if start_v < self.filter_v0_lowerbound or start_v >= self.filter_v0_upperbound:
          continue

        if self.key_mode == 'plot_elements':
          key_v = math.floor((start_v + self._speed_reso_2) / self._speed_reso) * self._speed_reso        
          if not key_v in self.dict_data:
            self.dict_data[key_v] = []
          self.dict_data[key_v].append(array_traj)
          self.data_num += 1
        else:
          raise NotImplementedError()

      return True

    def final_processing_data(self):
      pass

    def plot_data(self, dict_data: Dict, is_ordered: bool):
      plot_func = self.mode_functions[self.key_mode]
      plot_func(dict_data, is_ordered)

    def plot_trajs_list(self, dict_data: Dict, is_ordered: bool):
      # print("plot_trajs_list", is_ordered, dict_data.keys())
      fig = plt.figure()
      plot_utils.fig_reset()
      row_num = math.ceil(math.sqrt(float(len(dict_data.keys()))))
      col_num = row_num
      num_figure = row_num * col_num

      axess = []
      fid = 0
      for key_v, traj_list in dict_data.items():
        fid = fid + 1
        if fid > (row_num * col_num):
          print("{} out of plot figure range [{},{}], break.".format(fid, row_num, col_num))
          break
        print("subplot figure index={}/{}".format(fid, num_figure), 
              "; key v0={}m/s".format(key_v),
              "; traj_num={}.".format(len(traj_list))
        )
        axess.append(fig.add_subplot(row_num,col_num,fid))
        
        plot_utils.subfig_reset()
        axess[fid-1].set_title("v0=" + str(key_v))
        plot_utils.axis_set_xticks(axess[fid-1], [])
        for traj in traj_list:            
          plt.plot(traj[1:, 0], traj[1:, 1], '-')

      plt.legend()
      plt.show()

    # TODO: move functions in trajectory_analysis.py to here

