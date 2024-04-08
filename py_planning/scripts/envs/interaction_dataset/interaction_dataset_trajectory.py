from typing import List, Dict, Any
import collections
import copy
import math
import numpy as np
from envs.format_trajectory import DatasetTrajecotryIO
from envs.interaction_dataset.interface import InteractionDataset
import type_utils.agent as agent_utils
import type_utils.state_trajectory as state_trajfrom utils.transform import XYYawTransform

class InteractionDatasetTrajectoryExtractor(InteractionDataset):
  def __init__(self, data_path: str='train',
                     config_path: str=None,
                     save_path: str=None,
                     file_data_num: int=100):
    '''
    Interface to read interaction dataset data
    :param data_path: read the train or val folder in dataset. {'train', 'val'}.
    :param config_path: path to read config file
    :param save_path: path to save results
    :param file_data_num: how much cases of trajectories being stored in one file
    @note, for interaction dataset, data len corresponds to the csv file number
           but not the sample number
    '''
    super().__init__(branch='prediction', 
                     data_path=data_path, 
                     config_path=config_path,
                     save_path=save_path, file_data_num=file_data_num)

  def process_data(self, idx: int, scenario_name: str, 
                         file_data_num: int, **kwargs):
    '''
    Process data given index of map.csv
    :param idx: index of map.csv in dataset folder
    :param file_data_num: useless in this case
    '''
    cases_dict = kwargs['cases_dict']
    cases_num = len(cases_dict)
    agent_type_id = agent_utils.string2agent_type('vehicle')
  
    extractor = DatasetTrajecotryIO()
    
    extractor.reinit_batch_cases_data()
    for sample_idx, case in cases_dict.items():
      vehicle_track_dict = case[0]
      pedestrian_track_dict = case[1]

      self.append_case_data(extractor, idx, sample_idx, scenario_name, 
                            vehicle_track_dict, agent_type_id)
      extractor.set_case_data2batch_cases_data(sample_idx)

      extractor.try_write_batch_cases_data(self.save_path, idx, sample_idx, file_data_num)
    
    print("Process data: scene={}; case_amount={};".format(idx, len(cases_dict.keys())))
    extractor.try_write_batch_cases_data(
      self.save_path, idx, sample_idx, file_data_num, forcibly_write=True)

    return None

  def append_case_data(self, 
                       extractor: DatasetTrajecotryIO,
                       idx: int, sample_idx: int,
                       scenario_name: str,
                       agents_track_dict: Dict,
                       agent_type_id: int) -> None:
    '''
    Extract data from dataset
    '''
    extractor.reinit_case_data()

    for key, agent in agents_track_dict.items():
      # agent.track_id = id
      # agent.agent_type = None
      # agent.length = None
      # agent.width = None
      # agent.time_stamp_ms_first = None
      # agent.time_stamp_ms_last = None
      # agent.motion_states = dict()
      #       MotionState: {'time_stamp_ms': 100, 'x': 1062.882, 'y': 1003.611, 
      #                     'vx': 9.174, 'vy': 1.568, 'psi_rad': 0.169}
      od_motions = collections.OrderedDict(sorted(agent.motion_states.items()))
      state0 = od_motions[agent.time_stamp_ms_first]
      vx = state0.vx
      vy = state0.vy
      v = math.sqrt(vx*vx + vy*vy)

      # trajs records, # set filter conditions
      xyyaw0 = XYYawTransform(x=state0.x, y=state0.y, yaw_radian=state0.psi_rad)
      inv_xyyaw0 = copy.copy(xyyaw0)
      inv_xyyaw0.inverse()
      
      inv_last_xyyaw = copy.copy(inv_xyyaw0)

      # <format>: first unit with, [scene_id, length, width, agent_type, first_time_stamp_s]
      info = state_traj.TrajectoryInfo(
        scene_id=sample_idx,
        agent_type=agent_type_id,
        length=agent.length,
        width=agent.width,
        first_time_stamp_s=float(agent.time_stamp_ms_first) * 0.001,
        time_interval_s=0.1
      )

      local_traj = state_traj.StateTrajectory(info=info)
      global_traj = state_traj.StateTrajectory(info=info)

      is_2nd_frame: int = 2
      for key_time_ms, state in od_motions.items():
        xyyaw = XYYawTransform(x=state.x, y=state.y, yaw_radian=state.psi_rad)
        start2xyyaw = inv_xyyaw0.multiply_from_right(xyyaw)
        abs_v = math.sqrt(state.vx*state.vx + state.vy*state.vy)
        key_time_s = float(key_time_ms) * 0.001 # not need to add frame0_s

        last2this_xyyaw = inv_last_xyyaw.multiply_from_right(xyyaw)
        sign_v = abs_v
        if last2this_xyyaw._x <= -1e-9:
          sign_v = -abs_v

        if is_2nd_frame == 0:
          # correct initial frame v, which inline with 2nd frame
          abs_v0 = math.fabs(local_traj.state_value(0, 'velocity'))
          new_v0 = abs_v0 if sign_v >= 0.0 else -abs_v0
          local_traj.set_state_value(0, 'velocity', new_v0)

        local_traj.append_state(start2xyyaw._x, start2xyyaw._y, start2xyyaw._yaw, sign_v, key_time_s)
        global_traj.append_state(xyyaw._x, xyyaw._y, xyyaw._yaw, sign_v, key_time_s)

        inv_last_xyyaw = copy.copy(xyyaw)
        inv_last_xyyaw.inverse()
        is_2nd_frame -= 1

      extractor.append_case_data(
        local_traj=local_traj,
        global_traj=global_traj,
        correct_yaw=agent_utils.is_human(agent_type_id)
      )

