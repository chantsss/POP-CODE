#!/usr/bin/env python
import os
import abc
import collections
import math
import copyfrom typing import Dict, List
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatches
import random

import envs.configfrom envs.config import get_root2folderfrom analysis.collector import DataCollector
import type_utils.agent as agent_utils
import type_utils.state_trajectory as state_trajfrom utils.file_io import read_dict_from_binfrom utils.boostrap_sample import boostrap_sampling

import paper_plot.utils as plot_utils
import paper_plot.functions as plot_func

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

def rviz_boxplot(axs, tag_str: str, data_array: np.ndarray, 
                      data0_index: int, data1_index: int, data0_reso: float,
                      count_postive_bound: bool,
                      protect_range: List= []): 
  _, _, boxvalues = plot_func.plot_boxplot(
    axs, data_array[:, data0_index], data0_reso, data_array[:, data1_index])

  xi_values = boxvalues[:, 0]
  x_values = boxvalues[:, 1]

  if count_postive_bound:
    _2sigma_datas = np.vstack((x_values, boxvalues[:, 2] + 2.0*boxvalues[:, 3])).transpose()
    _1sigma_datas = np.vstack((x_values, boxvalues[:, 2] + 1.0*boxvalues[:, 3])).transpose()
    _05sigma_datas = np.vstack((x_values, boxvalues[:, 2] + 0.5*boxvalues[:, 3])).transpose()
  else:
    _2sigma_datas = np.vstack((x_values, boxvalues[:, 2] - 2.0*boxvalues[:, 3])).transpose()
    _1sigma_datas = np.vstack((x_values, boxvalues[:, 2] - 1.0*boxvalues[:, 3])).transpose()
    _05sigma_datas = np.vstack((x_values, boxvalues[:, 2] - 0.5*boxvalues[:, 3])).transpose()
  
  # axs.plot(x_values, _2sigma_datas[:, 1], 'o', label='$2\sigma$ acc. value')
  # axs.legend()

  poly_2sigma = np.polyfit(_2sigma_datas[:, 0], _2sigma_datas[:, 1], 4)
  poly_2sigma_bounds = np.polyval(poly_2sigma, x_values)
  poly_1sigma = np.polyfit(_1sigma_datas[:, 0], _1sigma_datas[:, 1], 4)
  poly_1sigma_bounds = np.polyval(poly_1sigma, x_values)
  poly_05sigma = np.polyfit(_05sigma_datas[:, 0], _05sigma_datas[:, 1], 4)
  poly_05sigma_bounds = np.polyval(poly_05sigma, x_values)

  if len(protect_range) == 2:
    # inquiry values
    protect_range = np.array(protect_range)
    fixed_values_2sigma = np.polyval(poly_2sigma, protect_range)
    fixed_values_1sigma = np.polyval(poly_1sigma, protect_range)
    fixed_values_05sigma = np.polyval(poly_05sigma, protect_range)

    # change values of sample points
    left_part_locs = x_values <= protect_range[0]
    right_part_locs = x_values >= protect_range[1]
    print("protect_range= ", protect_range)
    print("fixed_values_2sigma= ", fixed_values_2sigma)
    print("fixed_values_1sigma= ", fixed_values_1sigma)
    print("fixed_values_05sigma= ", fixed_values_05sigma)

    # plot the original values
    axs.plot(x_values, poly_2sigma_bounds, '-.', color='orange')
    axs.plot(x_values, poly_1sigma_bounds, '-.', color='g')
    axs.plot(x_values, poly_05sigma_bounds, '-.', color='b')

    # reset values
    poly_2sigma_bounds[left_part_locs] = fixed_values_2sigma[0]
    poly_2sigma_bounds[right_part_locs] = fixed_values_2sigma[1]
    poly_1sigma_bounds[left_part_locs] = fixed_values_1sigma[0]
    poly_1sigma_bounds[right_part_locs] = fixed_values_1sigma[1]
    poly_05sigma_bounds[left_part_locs] = fixed_values_05sigma[0]
    poly_05sigma_bounds[right_part_locs] = fixed_values_05sigma[1]
  else:
    print("protect_range values invlaid, len should be 2")

  axs.plot(x_values, poly_2sigma_bounds, '-', color='orange', label='poly. $2\sigma$ acc. bound')
  axs.legend()
  axs.plot(x_values, poly_1sigma_bounds, '-', color='g', label='poly. $1\sigma$ acc. bound')
  axs.legend()
  axs.plot(x_values, poly_05sigma_bounds, '-', color='b', label='poly. $0.5\sigma$ acc. bound')
  axs.legend() 

  print("**"*10)
  print("{}::debug()".format(tag_str))
  print("poly 2sigma=", poly_2sigma)
  print("poly 1sigma=", poly_1sigma)
  print("poly 0.5sigma=", poly_05sigma)

  print("test_values")
  print(np.polyval(poly_2sigma, np.array([-10.0, 0.0, 10.0])))

class InteractionCollector(DataCollector):
    '''
    Class to collect common datas from different dataset
    '''
    def __init__(self, key_mode: str=None,
                       cache_dir: str=None,
                       cache_batch_num: int=10000,
                       config_path: str=None,
                       save_dir: str=None):
      '''
      :param key_mode: key string for function design
      '''
      super().__init__(key_mode=key_mode, cache_dir=cache_dir,
                       cache_batch_num=cache_batch_num,
                       config_path=config_path,
                       save_dir=save_dir)

      self.cache_file_name = 'case_overlap_trajs'
      self.cache_exists = self.exists_at_cache_dir(self.cache_file_name)

      self.mode_functions = {
        'plot_elements': self.plot_interactions,
        'analyze_patterns': self.analyze_patterns,
      }

      self.enable_plot_patterns = {
        'analyze_patterns': True,
      }

      # reaction parameters
      self.agent_point_width :float= self.args_dict['reactions']['agent_point_width']
      self.react_fut_cond_t :float= self.args_dict['reactions']['future_cond_t']
      self.react_past_dur_t :float= self.args_dict['reactions']['past_dur_t']
      self.priority_input_dt :float= self.args_dict['reactions']['input_interval_t']
      self.priority_input_traj_len :int= int(self.args_dict['reactions']['input_max_traj_node_num'])

      self.train_vector_file_idx: int= 0

    def add_data(self, input_data: Dict):
      '''
      Add data to collector for analysis
      Return False if there is not need to add data
      '''
      if self.cache_exists:
        return False

      for case_str, dt in input_data.items():
        global_trajs = dt['global_traj']
        # print(case_str, dt.keys(), len(global_trajs))
        
        if not case_str in self.dict_data:
          self.dict_data[case_str] = []
  
        # trajboxes_dict: Dict = {}
        # def get_traj_polys(traj_idx: int, traj: state_traj.StateTrajectory, interval: int):
        #   if not traj_idx in trajboxes_dict:
        #     trajboxes_dict[traj_idx] = traj.get_merged_polys(
        #       0, traj.len(), interval, 0.0, 0.0)
        #   return trajboxes_dict[traj_idx]

        flag_exits = {}
        for traji_idx, traji_list in enumerate(global_trajs):
          traji = state_traj.StateTrajectory()
          traji.set_trajectory_list(traji_list)
          # interval = int(merged_dur_s / traji.get_info().time_interval_s)
          # traji_polys = get_traj_polys(traji_idx, traji, interval)

          for trajj_idx, trajj_list in enumerate(global_trajs):
            if (traji_idx == trajj_idx) or ((traji_idx, trajj_idx) in flag_exits):
              # print("avoid duplication", traji_idx, trajj_idx)
              continue
            flag_exits[(traji_idx, trajj_idx)] = True
            flag_exits[(trajj_idx, traji_idx)] = True

            trajj = state_traj.StateTrajectory()
            trajj.set_trajectory_list(trajj_list)
            # trajj_polys = get_traj_polys(trajj_idx, trajj, interval)

            # print("trajs len=", traji.len(), trajj.len())
            iinfos = traji.check_space_interaction(compare_traj=trajj, 
              point_width=self.agent_point_width)

            # build local trajs
            origin = traji.numpy_xyyaw_array()[0, :]
            local_traji = traji.get_local_frame_trajectory()
            local_trajj = trajj.get_local_frame_trajectory(origin)

            if (iinfos['has_overlap'] == True):
              self.dict_data[case_str].append({
                'traj_i': traji,
                'traj_j': trajj,
                'local_trajs': [local_traji, local_trajj],
                'interactions': iinfos
              })
              self.data_num += 1

        self.try_write_cache_data(self.cache_file_name)
      return True

    def final_processing_data(self):
      if not self.cache_exists:
        self.try_write_cache_data(self.cache_file_name, forcibly_write=True)

    def plot_data(self, dict_data: Dict, is_ordered: bool):
      # Read from cache files and plot
      fig = plt.figure()
      plot_utils.fig_reset()
      self.init_fig_ax = False
      # fig.set_size_inches(3.5, 2.163) # golden ratio= 0.618
      if self.key_mode == 'plot_elements':
        plot_utils.set_axis_equal()

      mode_func = self.mode_functions[self.key_mode]
      self.anal_results = {
        'statistics': {
          'total_num': 0,
          'near_situ_num': 0,
          'outlier_num': 0,
          'max_path_dist': 0.,
        },
        'overtake_theta2dt': [],
        'giveway_theta2dt': [],
        'react_dccs': [],
        'interaction_statistics': [],
        'conditional_statistics': [],
        'preempt_accs': [],
        'yield_accs': [],
      }

      file_list = self.read_cache_data_list(self.cache_file_name)
      len_file_list = len(file_list)
      for fidx, fname in enumerate(file_list):
        fpath = os.path.join(self.cache_dir, fname)
        dict_data:Dict = read_dict_from_bin(fpath, verbose=False)
        for case_name, list_dt in dict_data.items():
          # happens inside one case: case_name
          num = len(list_dt)
          for idx, dt in enumerate(list_dt):
            # print(case_name, idx, dt.keys(), num)
            print("\rProcess file {}/{}.".format(fidx, len_file_list), end="")
            mode_func(fig, dt)

      if self.key_mode in self.enable_plot_patterns:
        self.plot_patterns(fig)

    def plot_interactions(self, fig, dict_data: Dict, video_pause: float=0.01):
      if self.init_fig_ax == False:
        self.fig_ax = fig.add_subplot(1,1,1)
        plot_utils.subfig_reset()
        self.init_fig_ax = True

      traj_i = dict_data['traj_i']
      traj_j = dict_data['traj_j']

      traj_i_info = traj_i.get_info()
      traj_j_info = traj_j.get_info()
      interval_s = traj_i_info.time_interval_s

      cond_dur_s = 2.0 # minimum duration to show
      if (traj_i.duration_s() < cond_dur_s) or\
         (traj_j.duration_s() < cond_dur_s):
        return

      # traj_i is with 'g' color, traj_j is with 'b' color
      # traj1 is the one first emerge
      traji_color = 'g'
      trajj_color = 'b'
      traj1 = [traj_i, traj_i_info]
      traj2 = [traj_j, traj_j_info]
      if traj_i_info.first_time_stamp_s > traj_j_info.first_time_stamp_s:
        traj1 = [traj_j, traj_j_info]
        traj2 = [traj_i, traj_i_info]
        traji_color = 'b'
        trajj_color = 'g'
      bias_index = int(
        (traj2[1].first_time_stamp_s - traj1[1].first_time_stamp_s) / interval_s)
      traj1_len = traj1[0].len()
      traj2_len = traj2[0].len()
      array1 = traj1[0].numpy_trajecotry()
      array2 = traj2[0].numpy_trajecotry()

      for tidx1 in range(traj1[0].len()):
        tidx2 = tidx1 - bias_index
        plt.ion(), plt.cla(), plt.grid()

        plt.plot(array1[1:(tidx1+1), 0], array1[1:(tidx1+1), 1], traji_color+'.')
        self.fig_ax.add_patch(
          pltpatches.Circle(array1[tidx1+1, 0:2], 
            radius=traj1[1].width * 0.5, fc='None', ec=traji_color))

        if (0 <= tidx2) and (tidx2 < traj2_len):
          plt.plot(array2[1:(tidx2+1), 0], array2[1:(tidx2+1), 1], trajj_color+'.')
          self.fig_ax.add_patch(
            pltpatches.Circle(array2[tidx2+1, 0:2], 
              radius=traj2[1].width * 0.5, fc='None', ec=trajj_color))

        plt.pause(video_pause), plt.ioff()

    def analyze_patterns(self, fig, dict_data: Dict):
      config_cond_dur_s = 3.0 # minimum duration to analyze

      iinfos = dict_data['interactions']
      traj_i :state_traj.StateTrajectory= dict_data['traj_i']
      traj_j :state_traj.StateTrajectory= dict_data['traj_j']
      local_traj_i :state_traj.StateTrajectory= dict_data['local_trajs'][0]
      local_traj_j :state_traj.StateTrajectory= dict_data['local_trajs'][1]

      traj_i_info = traj_i.get_info()
      traj_j_info = traj_j.get_info()
      interval_s = traj_i_info.time_interval_s
      if (traj_i.duration_s() < config_cond_dur_s) or\
         (traj_j.duration_s() < config_cond_dur_s):
        return

      time_interval_s = traj_i_info.time_interval_s

      ## Analysis
      self.anal_results['statistics']['total_num'] += 1

      piecewise_dists = iinfos['piecewise_dists']
      point_interactions = iinfos['point_interaction']
      space_interactions = iinfos['space_interaction']

      # point_interaction = {
      #   'ipoint_index': [iid, jid2], 
      #   'arrive_t_s': [tpi_t, tpj_t],
      #   'arrive_dt_s': tpi_t - tpj_t,
      #   'directs': [tpi_yaw, tpj_yaw],
      #   'overlap_angle': get_normalized_angle(tpi_yaw - tpj_yaw),
      # }
      pindexs = point_interactions['ipoint_index']
      arrive_t_s = point_interactions['arrive_t_s']
      arrive_dt_s = point_interactions['arrive_dt_s']
      overlap_radian = point_interactions['overlap_angle']
      overlap_angle = overlap_radian * TO_DEGREE

      # skip when dt is too near or too large
      if math.fabs(arrive_dt_s) <= 1e-3:
        self.anal_results['statistics']['near_situ_num'] += 1
        return
      if math.fabs(arrive_dt_s) > 4.0:
        return
      # skip when it is outlier
      has_both_overtake_and_giveway = space_interactions['has_both_overtake_and_giveway']
      if has_both_overtake_and_giveway:
        self.anal_results['statistics']['outlier_num'] += 1
        return

      traj_i_piecewise_dists = piecewise_dists[0]
      traj_j_piecewise_dists = piecewise_dists[1]
      traj_i_path_length = np.sum(traj_i_piecewise_dists)
      traj_j_path_length = np.sum(traj_j_piecewise_dists)
      trajs_path_length = [traj_i_path_length, traj_j_path_length]

      is_overtake: bool = space_interactions['has_overtake']

      gway_idx: int = 0 if (arrive_dt_s >= 0.0) else 1
      ovtake_idx: int = 1-gway_idx
      traj_giveway = traj_i if (arrive_dt_s >= 0.0) else traj_j
      traj_ovtake = traj_j if (arrive_dt_s >= 0.0) else traj_i
      traj_giveway_array = traj_giveway.numpy_trajecotry()
      traj_ovtake_array = traj_ovtake.numpy_trajecotry()

      ovtake_agent_t0_s = traj_ovtake.first_time_stamp_s()
      ovtake_agent_te_s = traj_ovtake.end_time_stamp_s()
      giveway_agent_t0_s = traj_giveway.first_time_stamp_s()
      giveway_agent_te_s = traj_giveway.end_time_stamp_s()

      # _t1: when overtaking agent arrive the interaction point
      # _t2: when giving way agent arrive the interaction point 
      _t1 = arrive_t_s[ovtake_idx] # when agent_overtake arrive the interaction point
      _t2 = arrive_t_s[gway_idx]  # when agent_giveway arrive the interaction point

      ## collect theta 2 dts
      if arrive_dt_s <= 0.:
        self.anal_results['overtake_theta2dt'].append([overlap_angle, arrive_dt_s])
      else:
        self.anal_results['giveway_theta2dt'].append([overlap_angle, arrive_dt_s])

      ## clculate giveway features
      traj_i_t0_s = traj_i.first_time_stamp_s()
      traj_i_te_s = traj_i.end_time_stamp_s()
      traj_j_t0_s = traj_j.first_time_stamp_s()
      traj_j_te_s = traj_j.end_time_stamp_s()
      bias_t0_s = round(max(traj_i_t0_s, traj_j_t0_s), 1)
      end_t1_s = round(min(
          local_traj_i.state_value(pindexs[0], 'timestamp'), 
          local_traj_j.state_value(pindexs[1], 'timestamp')), 
        1)

      if (traj_i_t0_s < traj_j_te_s) and (traj_j_t0_s < traj_i_te_s):
        # time window has overlaps
        i0_index = local_traj_i.frame_index(local_traj_i.numpy_trajecotry(), bias_t0_s)
        j0_index = local_traj_j.frame_index(local_traj_j.numpy_trajecotry(), bias_t0_s)
        i1_index = local_traj_i.frame_index(local_traj_i.numpy_trajecotry(), end_t1_s)
        j1_index = local_traj_j.frame_index(local_traj_j.numpy_trajecotry(), end_t1_s)

        traj_i_idur = local_traj_i.state_value(pindexs[0], 'timestamp') - local_traj_i.state_value(i0_index, 'timestamp')
        traj_j_idur = local_traj_j.state_value(pindexs[1], 'timestamp') - local_traj_j.state_value(j0_index, 'timestamp')
        interaction_dur = round(min(traj_i_idur, traj_j_idur), 1)

        if math.fabs(arrive_dt_s) > 1e-1:
          traj_i_dist2ipoint = np.sum(traj_i_piecewise_dists[i0_index:(pindexs[0]-1)])
          traj_j_dist2ipoint = np.sum(traj_j_piecewise_dists[j0_index:(pindexs[1]-1)])
          
          traj_i_dist2ipoint -= (traj_i.info.length * 0.5)
          traj_i_dist2ipoint = max(0.0, traj_i_dist2ipoint)
          traj_j_dist2ipoint -= (traj_j.info.length * 0.5)
          traj_j_dist2ipoint = max(0.0, traj_j_dist2ipoint)

          traj_i_dist2ipoint_situ2 = np.sum(traj_i_piecewise_dists[i1_index:(pindexs[0]-1)])
          traj_j_dist2ipoint_situ2 = np.sum(traj_j_piecewise_dists[j1_index:(pindexs[1]-1)])

          traj_i_dist2ipoint_situ2 -= (traj_i.info.length * 0.5)
          traj_i_dist2ipoint_situ2 = max(0.0, traj_i_dist2ipoint_situ2)
          traj_j_dist2ipoint_situ2 -= (traj_j.info.length * 0.5)
          traj_j_dist2ipoint_situ2 = max(0.0, traj_j_dist2ipoint_situ2)

          tidx :int= traj_i.state_format['timestamp']
          vidx :int= traj_i.state_format['velocity']

          traj_dict = {
            'traj_i': {},
            'traj_j': {},
          }

          i_area = traj_i.info.get_shape_area()
          j_area = traj_j.info.get_shape_area()

          traj_dict['traj_i']['index0'] = i0_index # when interaction begin
          traj_dict['traj_i']['index1'] = i1_index # when interaction finished: one of participants arrive the interaction point
          traj_dict['traj_i']['indexe'] = pindexs[0] # when this agent arrive the interaction point
          traj_dict['traj_i']['dist2ipoint'] = traj_i_dist2ipoint
          traj_dict['traj_i']['dur2ipoint'] = arrive_t_s[0] - bias_t0_s

          traj_dict['traj_j']['index0'] = j0_index
          traj_dict['traj_j']['index1'] = j1_index
          traj_dict['traj_j']['indexe'] = pindexs[1]
          traj_dict['traj_j']['dist2ipoint'] = traj_j_dist2ipoint
          traj_dict['traj_j']['dur2ipoint']= arrive_t_s[1] - bias_t0_s

          if (traj_dict['traj_i']['dur2ipoint'] > 1e-1) and\
             (traj_dict['traj_j']['dur2ipoint'] > 1e-1) and\
             (traj_dict['traj_i']['dist2ipoint'] > 1e-1) and\
             (traj_dict['traj_j']['dist2ipoint'] > 1e-1):
            time_interval_s = traj_i_info.time_interval_s
            traj_dict['traj_i']['v0'] =\
              traj_i.state_value(traj_dict['traj_i']['index0'], 'velocity')
            traj_dict['traj_j']['v0'] =\
              traj_j.state_value(traj_dict['traj_j']['index0'], 'velocity')
            traj_dict['traj_i']['v1'] =\
              traj_i.state_value(traj_dict['traj_i']['index1'], 'velocity')
            traj_dict['traj_j']['v1'] =\
              traj_j.state_value(traj_dict['traj_j']['index1'], 'velocity')
            traj_dict['traj_i']['array_v'] =\
              traj_i.numpy_trajecotry()[(1+traj_dict['traj_i']['index0']):(1+pindexs[0]), vidx]
            traj_dict['traj_j']['array_v'] =\
              traj_j.numpy_trajecotry()[(1+traj_dict['traj_j']['index0']):(1+pindexs[1]), vidx]
            traj_dict['traj_i']['01_array_v'] =\
              traj_i.numpy_trajecotry()[(1+i0_index):(1+i1_index+10), vidx]
            traj_dict['traj_j']['01_array_v'] =\
              traj_j.numpy_trajecotry()[(1+j0_index):(1+j1_index+10), vidx]
            traj_dict['traj_i']['cvm_reach_t'] = min(10.0, 
              traj_dict['traj_i']['dist2ipoint'] / max(traj_dict['traj_i']['v0'], 0.2))
            traj_dict['traj_j']['cvm_reach_t'] = min(10.0, 
              traj_dict['traj_j']['dist2ipoint'] / max(traj_dict['traj_j']['v0'], 0.2))
            traj_dict['traj_i']['cvm_reach_t_situ2'] = min(10.0, 
              traj_i_dist2ipoint_situ2 / max(traj_dict['traj_i']['v1'], 0.2))
            traj_dict['traj_j']['cvm_reach_t_situ2'] = min(10.0, 
              traj_j_dist2ipoint_situ2 / max(traj_dict['traj_j']['v1'], 0.2))

            traj_dict['traj_i']['avg_acc'] = 0.0
            array_v = traj_dict['traj_i']['array_v']
            if array_v.shape[0] > 2:
              # s = v0 * t + 0.5 * a * t**2
              # >> a = 2.0 * (s - v0 * t) / (t**2)
              traj_dict['traj_i']['array_acc'] = (array_v[1:] - array_v[:-1]) / time_interval_s
              traj_dict['traj_i']['avg_acc'] = np.mean(traj_dict['traj_i']['array_acc'])
              # traj_dict['traj_i']['avg_acc'] = 2.0 *\
              #   (traj_dict['traj_i']['dist2ipoint'] - traj_dict['traj_i']['v0'] * traj_dict['traj_i']['dur2ipoint']) / (traj_dict['traj_i']['dur2ipoint'] **2)

            traj_dict['traj_j']['avg_acc'] = 0.0
            array_v = traj_dict['traj_j']['array_v']
            if array_v.shape[0] > 2:
              traj_dict['traj_j']['array_acc'] = (array_v[1:] - array_v[:-1]) / time_interval_s
              traj_dict['traj_j']['avg_acc'] = np.mean(traj_dict['traj_j']['array_acc'])
              # traj_dict['traj_j']['avg_acc'] = 2.0 *\
              #   (traj_dict['traj_j']['dist2ipoint'] - traj_dict['traj_j']['v0'] * traj_dict['traj_j']['dur2ipoint']) / (traj_dict['traj_j']['dur2ipoint'] **2)

            traj_dict['traj_i']['01_mean_acc'] = 0.0
            traj_dict['traj_j']['01_mean_acc'] = 0.0
            array_v = traj_dict['traj_i']['01_array_v']
            if array_v.shape[0] >= 2:
              traj_dict['traj_i']['01_mean_acc'] = np.mean((array_v[1:] - array_v[:-1]) / time_interval_s)
            array_v = traj_dict['traj_j']['01_array_v']
            if array_v.shape[0] >= 2:
              traj_dict['traj_j']['01_mean_acc'] = np.mean((array_v[1:] - array_v[:-1]) / time_interval_s)

            agent_i_high_priority = (arrive_dt_s < 0.0)
            self.anal_results['interaction_statistics'].append([
              traj_dict['traj_i']['v0'], # [0]
              traj_dict['traj_j']['v0'],
              traj_dict['traj_i']['cvm_reach_t'], # [2]
              traj_dict['traj_j']['cvm_reach_t'],
              traj_dict['traj_i']['dist2ipoint'], # [4]
              traj_dict['traj_j']['dist2ipoint'],
              traj_dict['traj_i']['avg_acc'], # [6]
              traj_dict['traj_j']['avg_acc'],
              agent_i_high_priority * 1.0, # [8]
              interaction_dur,
              traj_dict['traj_i']['01_mean_acc'], # [10] 01_mean_acc
              traj_dict['traj_j']['01_mean_acc'],
            ])
            # print("sttttttt", traj_i_dist2ipoint_situ2, traj_j_dist2ipoint_situ2)

            giveway_key = 'traj_i'
            ovtk_key = 'traj_j'
            if agent_i_high_priority == True:
              giveway_key = 'traj_j'
              ovtk_key = 'traj_i'

            giveway_traj = traj_dict[giveway_key]
            ovtk_traj = traj_dict[ovtk_key]

            if (giveway_traj['dist2ipoint'] <= 100.0) and\
               (giveway_traj['avg_acc'] >= -5.0) and (giveway_traj['avg_acc'] <= 5.0):
              giveway_agent_cvm_THW = giveway_traj['cvm_reach_t_situ2']

              self.anal_results['conditional_statistics'].append([
                giveway_traj['v0'], # [0]
                ovtk_traj['v0'],
                giveway_traj['cvm_reach_t'], # [2]
                ovtk_traj['cvm_reach_t'],
                giveway_traj['dist2ipoint'], # [4]
                ovtk_traj['dist2ipoint'],
                giveway_traj['avg_acc'],  # [6]
                ovtk_traj['avg_acc'],
                interaction_dur,  # [8]
                giveway_agent_cvm_THW, 
              ])

              if (ovtk_traj['cvm_reach_t'] > giveway_traj['cvm_reach_t']):
                self.anal_results['preempt_accs'].append(
                  [
                    ovtk_traj['array_acc'],
                    giveway_traj['array_acc'],
                  ]
                )

      ## collect interactions
      # print("t1/t2 = {:.2f}, {:.2f}.".format(_t1, _t2))
      # print("with overtake agent {:.2f}, {:.2f}".format(ovtake_agent_t0_s, ovtake_agent_te_s))
      # print("with giveway agent {:.2f}, {:.2f}".format(giveway_agent_t0_s, giveway_agent_te_s))
      cache_v = self.anal_results['statistics']['max_path_dist']
      self.anal_results['statistics']['max_path_dist'] = max(
        cache_v, max(trajs_path_length[ovtake_idx], trajs_path_length[gway_idx]))

      ## collect reaction relevant values
      # react dcc values
      if ((_t2 - _t1) <= self.react_fut_cond_t) and\
         ((_t1 - giveway_agent_t0_s) >= self.react_past_dur_t):
        _t0 = _t1 - self.react_past_dur_t
        index_t0 = traj_giveway.frame_index(traj_giveway_array, _t0)
        index_t1 = traj_giveway.frame_index(traj_giveway_array, _t1)
        # print(index_t0, index_t1, traj_giveway_array.shape)

        tidx :int= traj_giveway.state_format['timestamp']
        vidx :int= traj_giveway.state_format['velocity']
        array_t = traj_giveway_array[(1+index_t0):(2+index_t1), tidx]
        array_v = traj_giveway_array[(1+index_t0):(2+index_t1), vidx]

        if array_t.shape[0] >= 2:
          array_acc = (array_v[1:] - array_v[:-1]) / (array_t[1:] - array_t[:-1])
          array_dcc = array_acc[array_acc <= 0.]
          self.anal_results['react_dccs'] += array_dcc.tolist()

      ## collect inputs/outputs for priority prediction model
      traj_feat_size :int= 2
      point_feat_size :int= 2

      # update input_trajs_array[0, :, :]
      if (traj_i_t0_s < traj_j_te_s) and (traj_j_t0_s < traj_i_te_s):
        # time window has overlaps
        i0_index = local_traj_i.frame_index(local_traj_i.numpy_trajecotry(), bias_t0_s)
        j0_index = local_traj_j.frame_index(local_traj_j.numpy_trajecotry(), bias_t0_s)

        # inputs
        input_trajs_array = np.zeros(
          (traj_feat_size, self.priority_input_traj_len, point_feat_size))

        traj_i_path_length =\
          np.sum(traj_i_piecewise_dists[i0_index:(pindexs[0]-1)])
        traj_j_path_length =\
          np.sum(traj_j_piecewise_dists[j0_index:(pindexs[1]-1)])

        input_path_features = np.array([
          ## interaction angle in radian
          overlap_radian,
          ## i: s, des_x, des_y, des_yaw
          traj_i_path_length,
          local_traj_i.state_value(pindexs[0], 'pos_x'), 
          local_traj_i.state_value(pindexs[0], 'pos_y'),
          local_traj_i.state_value(pindexs[0], 'pos_yaw'),
          ## j: s, des_x, des_y, des_yaw
          traj_j_path_length,
          local_traj_j.state_value(pindexs[1], 'pos_x'), 
          local_traj_j.state_value(pindexs[1], 'pos_y'),
          local_traj_j.state_value(pindexs[1], 'pos_yaw'),
        ])

        input_v0_array = np.array(
          [local_traj_i.state_value(i0_index, 'velocity'), 
           local_traj_j.state_value(j0_index, 'velocity')])

        # outputs
        output_priority_array = np.array([0.0, 1.0])
        if is_overtake:
          output_priority_array = np.array([1.0, 0.0])

        ltraji_array = local_traj_i.numpy_trajecotry()[1:, :]
        ltrajj_array = local_traj_j.numpy_trajecotry()[1:, :]

        xidx :int= traj_i.state_format['pos_x']
        yidx :int= traj_i.state_format['pos_y']
        tidx :int= traj_i.state_format['timestamp']

        fill_ti :int = 0
        for node in ltraji_array:
          node_t :float= node[tidx]
          add_flag :bool= (node_t-bias_t0_s - (fill_ti * self.priority_input_dt - 1e-3)) >= 1e-4

          if add_flag:
            cond_dt = node_t-bias_t0_s - float(fill_ti) * self.priority_input_dt
            update_flag = (cond_dt < 9e-2)
            if update_flag:
              input_trajs_array[0, fill_ti, :] = node[[xidx, yidx]]
            fill_ti += 1

            if fill_ti >= self.priority_input_traj_len:
              break

        # update input_trajs_array[1, :, :]
        fill_ti = 0
        for node in ltrajj_array:
          node_t :float= node[tidx]
          add_flag :bool= (node_t-bias_t0_s - (fill_ti * self.priority_input_dt - 1e-3)) >= 1e-4
          
          if add_flag:
            cond_dt = node_t-bias_t0_s - float(fill_ti) * self.priority_input_dt
            update_flag = (cond_dt < 9e-2)
            if update_flag:
              input_trajs_array[1, fill_ti, :] = node[[xidx, yidx]]
            fill_ti += 1

            if fill_ti >= self.priority_input_traj_len:
              break

        # save vectors for trainning
        save2file_name :str= '{}_vectors.bin'.format(self.train_vector_file_idx)
        self.save_data(
          filename=save2file_name, 
          data = {
            'inputs': {
              'i_trajs': input_trajs_array,
              'i_path_feats': input_path_features,
              'i_dur': input_v0_array,
            },
            'ground_truth': output_priority_array,
          }
        )
        self.train_vector_file_idx += 1

    def plot_patterns(self, fig):
      self.anal_results['react_dccs'] = np.array(self.anal_results['react_dccs'])
      print("Interaction statistics results= {}.".format(self.anal_results['statistics']))
      print("Total reaction amount=", self.anal_results['react_dccs'].shape[0])

      feature1 = np.array(self.anal_results['giveway_theta2dt'])
      feature2 = np.array(self.anal_results['overtake_theta2dt'])
      feature3 = self.anal_results['react_dccs']

      feature4 = np.array(self.anal_results['interaction_statistics'])
      feature5 = np.array(self.anal_results['conditional_statistics'])
      iaccs_list = self.anal_results['preempt_accs']
    
      angle_reso = 10.0
      v_reso = 1.0
      acc_reso = 0.25
      dist_reso = 5.0
      t_reso = 0.5

      grid_cond_num :int= 10
      blank_fill_value = math.nan

      i_key_v0 :int= 0
      j_key_v0 :int= 1
      i_key_cvm_t :int= 2
      j_key_cvm_t :int= 3
      i_key_dist :int= 4
      j_key_dist :int= 5
      i_key_avg_acc :int= 6
      j_key_avg_acc :int= 7
      i_high_pri :int= 8
      f4_idur :int= 9
      f4_i_01_mean_acc :int= 10
      f4_j_01_mean_acc :int= 11

      f5_idur :int= 8
      f5_cvm_thw :int= 9

      # #####################################################################
      ref_acc = 0.5
      m_t_coef = 0.1
      protect_t = 0.5

      # feature5 = feature5[feature5[:, f5_idur] > 2.0] # only consider idur > 2.0 parts
      hp_data = []
      lp_data = []
      for hp, v0, cvm_dt, idur, avg_acc in zip(
        feature4[:, i_high_pri],
        feature4[:, i_key_v0],
        feature4[:, i_key_cvm_t] - feature4[:, j_key_cvm_t],
        feature4[:, f4_idur], 
        feature4[:, i_key_avg_acc]):
        if (hp > 0.5):
          hp_data.append([idur, v0, cvm_dt, avg_acc])
        else:
          lp_data.append([idur, v0, cvm_dt, avg_acc])
      hp_data = np.array(hp_data)
      lp_data = np.array(lp_data)

      g_idur = feature5[:, f5_idur]
      g_i_v0 = feature5[:, i_key_v0]
      g_i_move_s = feature5[:, i_key_dist]
      g_j_cvm_t = feature5[:, j_key_cvm_t]
      # 0.5 * a * t**2 + v0 * t - s = 0.0
      # >> t = (-v0 + sqrt(v0**2 + 2.0 * a * s)) / a
      g_i_acc_t = (-g_i_v0 + np.sqrt(np.square(g_i_v0) + 2.0*ref_acc*g_i_move_s)) / ref_acc
      m_ij_t = g_i_acc_t / (m_t_coef + g_j_cvm_t)

      # [content1]
      g_i_dur2mt = np.vstack((g_idur, m_ij_t)).transpose()

      g_j_v0 = feature5[:, j_key_v0]
      g_j_move_s = feature5[:, j_key_dist]
      g_i_cvm_t = feature5[:, i_key_cvm_t]
      # 0.5 * a * t**2 + v0 * t - s = 0.0
      # >> t = (-v0 + sqrt(v0**2 + 2.0 * a * s)) / a
      g_j_acc_t = (-g_j_v0 + np.sqrt(np.square(g_j_v0) + 2.0*ref_acc*g_j_move_s)) / ref_acc
      m_ji_t = g_j_acc_t / (m_t_coef + g_i_cvm_t)

      # [content1]
      g_j_dur2mt = np.vstack((g_idur, m_ji_t)).transpose()

      # [content2]
      g_j_preempt_acc = 2.0 * (g_j_move_s - g_j_v0 * (g_i_cvm_t - protect_t)) / (1e-3 + np.square(g_i_cvm_t - protect_t))
      g_j_yield_acc = 2.0 * (g_j_move_s - g_j_v0 * (g_i_cvm_t + protect_t)) / np.square(g_i_cvm_t + protect_t)

      g_i_preempt_acc = 2.0 * (g_i_move_s - g_i_v0 * (g_j_cvm_t - protect_t)) / (1e-3 + np.square(g_j_cvm_t - protect_t))
      g_i_yield_acc = 2.0 * (g_i_move_s - g_i_v0 * (g_j_cvm_t + protect_t)) / np.square(g_j_cvm_t + protect_t)

      protect_acc = 10.0
      g_j_preempt_acc[(g_i_cvm_t - protect_t) <= 0.0] = protect_acc
      g_i_preempt_acc[(g_j_cvm_t - protect_t) <= 0.0] = protect_acc

      g_j_preempt_acc[g_j_preempt_acc >= protect_acc] = protect_acc
      g_j_preempt_acc[g_j_preempt_acc <= -protect_acc] = -protect_acc
      g_i_preempt_acc[g_i_preempt_acc >= protect_acc] = protect_acc
      g_i_preempt_acc[g_i_preempt_acc <= -protect_acc] = -protect_acc
      g_j_yield_acc[g_j_yield_acc >= protect_acc] = protect_acc
      g_j_yield_acc[g_j_yield_acc <= -protect_acc] = -protect_acc
      g_i_yield_acc[g_i_yield_acc >= protect_acc] = protect_acc
      g_i_yield_acc[g_i_yield_acc <= -protect_acc] = -protect_acc

      # [content3]
      g_i_dur2cvm_dts = np.vstack((g_idur, (g_i_cvm_t - g_j_cvm_t))).transpose()
      g_j_dur2cvm_dts = np.vstack((g_idur, (g_j_cvm_t - g_i_cvm_t))).transpose()

      # [content5]
      i_avg_acc = feature5[:, i_key_avg_acc]
      j_avg_acc = feature5[:, j_key_avg_acc]
      idur = feature5[:, f5_idur]

      ### rviz ax1
      dur_max = np.max(g_idur) + 0.1

      fig.set_size_inches(3.5 * 2, 2.163)
      
      fig_ax1 = fig.add_subplot(111)
      plot_utils.subfig_reset()
      plot_utils.axis_set_title(fig_ax1, 'dur-m_t', loc_y=1.0)
      plot_utils.axis_set_xlabel(fig_ax1, 'dur. (s)')
      plot_utils.axis_set_ylabel(fig_ax1, '$M^T$')
      
      fig_ax1.plot(g_j_dur2mt[:, 0], g_j_dur2mt[:, 1], '.', color='orange', label='preempt', markersize=1)
      fig_ax1.plot(g_i_dur2mt[:, 0], g_i_dur2mt[:, 1], '.', color='grey', label='yield', markersize=1)

      fig_ax1.set_xlim(-0.1, dur_max)

      # Create the subwindow that zooms in on a certain area
      subax = fig_ax1.inset_axes([0.55, 0.55, 0.4, 0.4])
      subax.plot(g_j_dur2mt[:, 0], g_j_dur2mt[:, 1], '.', color='orange', markersize=1)
      subax.plot(g_i_dur2mt[:, 0], g_i_dur2mt[:, 1], '.', color='grey', markersize=1)
      # subax.set_xlim(0, 10)
      subax.set_yticks([0.0, 1.0, 2.0])
      subax.grid(True)
      subax.set_yticks([0.0, 1.0, 2.0, 4.0, 6.0])
      subax.set_ylim(-0.5, 6.0)

      plt.legend(loc='upper left')
      plot_utils.save_fig(os.path.expanduser('~') + "/dur2mt_test", vdpi=600)
      # plt.show()

      ### rviz ax2
      fig.clf()
      fig.set_size_inches(3.5 * 2, 2.163)

      fig_ax2 = fig.add_subplot(111)
      plot_utils.subfig_reset()
      plot_utils.axis_set_title(fig_ax2, 'dur-acc', loc_y=1.0)
      plot_utils.axis_set_xlabel(fig_ax2, 'dur. (s)')
      plot_utils.axis_set_ylabel(fig_ax2, 'acc. (m/$s^2$)')

      fig_ax2.plot(g_idur, g_i_yield_acc, ',r', label='preempt', markersize=1)
      fig_ax2.plot(g_idur, g_j_yield_acc, ',b', label='yield', markersize=1)
      # fig_ax2.plot(-g_idur, i_avg_acc, '.r', label='preempt', markersize=1)
      # fig_ax2.plot(g_idur, j_avg_acc, '.b', label='yield', markersize=1)
      fig_ax2.set_xlim(-0.1, dur_max)

      # Create the subwindow that zooms in on a certain area
      subax = fig_ax2.inset_axes([0.55, 0.55, 0.4, 0.4])
      subax.plot(g_idur, g_i_yield_acc, ',r', markersize=1)
      subax.plot(g_idur, g_j_yield_acc, ',b', markersize=1)
      # subax.plot(-g_idur, i_avg_acc, '.r', markersize=1)
      # subax.plot(g_idur, j_avg_acc, '.b', markersize=1)
      subax.grid(True)
      subax.set_yticks([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
      subax.set_ylim(-4.0, 6.0)
      
      plt.legend(loc='upper left')
      plot_utils.save_fig(os.path.expanduser('~') + "/dur2acc_test", vdpi=600)
      # plt.show()

      # ### rviz ax5
      fig.clf()
      fig.set_size_inches(3.5 * 2, 2.163)

      fig_ax5 = fig.add_subplot(111) # projection='3d'
      plot_utils.subfig_reset()
      plot_utils.axis_set_title(fig_ax5, 'Avg. acc. ', loc_y=1.0)
      plot_utils.axis_set_xlabel(fig_ax5, '$\Delta$t (s)')
      plot_utils.axis_set_ylabel(fig_ax5, 'acc. ($m/s^2$)')

      cvm_ji_dt = g_j_cvm_t - g_i_cvm_t
      cvm_ij_dt = g_i_cvm_t - g_j_cvm_t
      fig_ax5.plot(cvm_ji_dt, g_i_yield_acc, ',r', label='preempt', markersize=1)
      fig_ax5.plot(cvm_ij_dt, g_j_yield_acc, ',b', label='yield', markersize=1)
      # fig_ax5.plot(cvm_ji_dt, j_avg_acc, ',r', label='preempt', markersize=1)
      # fig_ax5.plot(cvm_ij_dt, i_avg_acc, ',b', label='yield', markersize=1)
      fig_ax5.grid(True)
      # fig_ax5.set_ylim([-5.0, 5.0])

      # Create the subwindow that zooms in on a certain area
      subax = fig_ax5.inset_axes([0.65, 0.65, 0.3, 0.3])
      subax.plot(cvm_ij_dt, g_j_yield_acc, ',b', markersize=1)
      subax.plot(cvm_ji_dt, g_i_yield_acc, ',r', markersize=1)
      subax.set_xticks([-10.0, -5.0, 0.0, 1.0, 5.0, 10.0])
      subax.set_ylim(-4.0, 4.0)
      subax.set_yticks([-4.0, -1.0, 0.0, 1.0, 4.0])
      subax.grid(True)

      plt.legend(loc='upper left')
      plot_utils.save_fig(os.path.expanduser('~') + "/dt2acc_test", vdpi=600)
      # plt.show()

      ### rviz ax3
      fig.clf()
      fig.set_size_inches(3.5 * 2, 2.163)

      fig_ax3 = fig.add_subplot(111)
      plot_utils.subfig_reset()
      plot_utils.axis_set_title(fig_ax3, 'acc-acc', loc_y=1.0)
      plot_utils.axis_set_xlabel(fig_ax3, 'preempt acc. (m/$s^2$)')
      plot_utils.axis_set_ylabel(fig_ax3, 'yield acc. (m/$s^2$)')

      fig_ax3.plot(g_i_preempt_acc, g_i_yield_acc, '.r', label='preempt', markersize=1)
      fig_ax3.plot(g_j_preempt_acc, g_j_yield_acc, '.b', label='yield', markersize=1)
      fig_ax3.grid(True)

      # # Create the subwindow that zooms in on a certain area
      # subax = fig_ax3.inset_axes([0.55, 0.55, 0.4, 0.4])
      # subax.plot(g_i_preempt_acc, g_i_yield_acc, '.r', markersize=1)
      # subax.plot(g_j_preempt_acc, g_j_yield_acc, '.b', markersize=1)
      # subax.grid(True)
      # subax.set_yticks([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
      # subax.set_ylim(-4.0, 6.0)
      
      plt.legend(loc='upper left')
      plot_utils.save_fig(os.path.expanduser('~') + "/accs_test", vdpi=600)
      # plt.show()

      ### rviz ax4
      fig.clf()
      fig.set_size_inches(3.5 * 2, 2.163)

      fig_ax4 = fig.add_subplot(111)
      plot_utils.subfig_reset()
      plot_utils.axis_set_title(fig_ax4, 'V0 - cvm dt', loc_y=1.0)
      plot_utils.axis_set_xlabel(fig_ax4, 'v0. (m/s)')
      plot_utils.axis_set_ylabel(fig_ax4, 'CVM dt (s)')

      fig_ax4.plot(hp_data[:, 1], hp_data[:, 2], ',r', label='preempt', markersize=1)
      fig_ax4.plot(lp_data[:, 1], lp_data[:, 2], ',b', label='yield', markersize=1)
      # # Create the subwindow that zooms in on a certain area
      # subax = fig_ax4.inset_axes([0.65, 0.65, 0.3, 0.3])
      # subax.plot(lp_data[:, 1], lp_data[:, 2], ',b', markersize=1)
      # subax.plot(hp_data[:, 1], hp_data[:, 2], ',r', markersize=1)
      # subax.set_ylim(-2.0, 2.0)
      # subax.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
      # subax.grid(True)
      
      plt.legend(loc='upper left')
      plot_utils.save_fig(os.path.expanduser('~') + "/v02cvm_dt_test", vdpi=600)
      # plt.show()

      ### rviz ax6
      fig.clf()
      plot_utils.fig_reset()
      fig.set_size_inches(3.5 * 2, 2.163)
      fig.suptitle("interaction dur. - mean acc. ")

      fig_ax6_1 = fig.add_subplot(121)
      plot_utils.subfig_reset()
      plot_utils.axis_set_xlabel(fig_ax6_1, 'preempt dur. (s)')
      plot_utils.axis_set_ylabel(fig_ax6_1, 'mean acc. (m/$s^2$)')
      fig_ax6_1.plot(hp_data[:, 0], hp_data[:, 3], '.r', label='preempt', markersize=1)
      plt.legend(loc='lower right')
      # plot_func.plot_boxplot(fig_ax6_1, hp_data[:, 0], 0.5, hp_data[:, 3])

      fig_ax6_2 = fig.add_subplot(122)
      plot_utils.subfig_reset()
      plot_utils.axis_set_xlabel(fig_ax6_2, 'yield dur. (s)')
      plot_utils.axis_set_ylabel(fig_ax6_2, 'mean acc. (m/$s^2$)')
      fig_ax6_2.plot(lp_data[:, 0], lp_data[:, 3], '.b', label='yield', markersize=1)
      plt.legend(loc='lower right')
      # plot_func.plot_boxplot(fig_ax6_2, lp_data[:, 0], 0.5, lp_data[:, 3])

      plot_utils.save_fig(os.path.expanduser('~') + "/dur2avg_acc", vdpi=600)
      # plt.show()

      ### rviz ax7
      fig.clf()
      fig.set_size_inches(3.5 * 2, 2.163)

      fig_ax7 = fig.add_subplot(111)
      plot_utils.subfig_reset()
      plot_utils.axis_set_title(fig_ax7, 'v0 - dcc. ', loc_y=1.0)
      plot_utils.axis_set_xlabel(fig_ax7, 'v0 (m/s)')
      plot_utils.axis_set_ylabel(fig_ax7, 'avg. dcc. (m/$s^2$)')

      valid_locs = (feature5[:, i_key_avg_acc] <= 0.0)
      rviz_boxplot(fig_ax7, "v02avg_dcc", feature5[valid_locs, :], 
                   i_key_v0, i_key_avg_acc, data0_reso=v_reso,
                   count_postive_bound=False,
                   protect_range=[13.5, 19.5])

      # plt.legend(loc='upper left')
      plot_utils.save_fig(os.path.expanduser('~') + "/v02avg_acc", vdpi=600)
      # plt.show()

      # #####################################################################
      # # collect probabilities
      # xlist, ylist = feature4[:, 2], feature4[:, 3]
      # xreso, yreso = t_reso, t_reso
      # flaglist = feature4[:, 8]

      # # init values
      # grid_probs = {}
      # rviz_probs = []
      # prob_blank_locs = []

      # xf, xidlist = plot_func.axis_feature(vlist=xlist, vreso=xreso)
      # yf, yidlist = plot_func.axis_feature(vlist=ylist, vreso=yreso)
      # print("xf", xf)
      # print("yf", yf)

      # _x = np.linspace(xf['vmin'], xf['vmax'], math.ceil((xf['vmax'] - xf['vmin']) / xreso))
      # _y = np.linspace(yf['vmin'], yf['vmax'], math.ceil((yf['vmax'] - yf['vmin']) / yreso))
      # _xx, _yy = np.meshgrid(_x, _y)
      # _xlist, _ylist = _xx.ravel(), _yy.ravel()

      # xsiz = int(xf['idmax']-xf['idmin'] + 1)
      # ysiz = int(yf['idmax']-yf['idmin'] + 1)

      # record_flag_num = np.zeros([xsiz, ysiz])
      # record_total_num = np.zeros_like(record_flag_num)
      # for xid, yid, flag in zip(xidlist, yidlist, flaglist):
      #   if flag > 0.5:
      #     record_flag_num[xid, yid] = record_flag_num[xid, yid] + 1.
      #   record_total_num[xid, yid] = record_total_num[xid, yid] + 1.

      # for x, y in zip(_xlist, _ylist):
      #   xid :int = math.floor((x - xf['vmin'] + xreso * 0.5) / xreso)
      #   yid :int = math.floor((y - yf['vmin'] + yreso * 0.5) / yreso)
      #   if record_total_num[xid, yid] > grid_cond_num:
      #     prob = record_flag_num[xid, yid] / record_total_num[xid, yid]
      #     grid_probs[(xid, yid)] = prob
      #     rviz_probs.append([x, y, prob])
      #   else:
      #     grid_probs[(xid, yid)] = blank_fill_value
      #     prob_blank_locs.append([x, y])

      # rviz_probs = np.array(rviz_probs)
      # prob_blank_locs = np.array(prob_blank_locs)

      # #####################################################################
      # # collect tolerant acc

      # xlist, ylist = feature5[:, 2], feature5[:, 3]
      # acclist = feature5[:, 6]

      # # init values
      # grid_accs = {}
      # rviz_accs = []
      # acc_blank_locs = []

      # record_acc = {} 
      # for x, y, acc in zip(xlist, ylist, acclist):
      #   xid :int = math.floor((x - xf['vmin'] + xreso * 0.5) / xreso)
      #   yid :int = math.floor((y - yf['vmin'] + yreso * 0.5) / yreso)
      #   if not (xid, yid) in record_acc.keys():
      #     record_acc[(xid, yid)] = []
      #   record_acc[(xid, yid)].append(acc)

      # for x, y in zip(_xlist, _ylist):
      #   xid :int = math.floor((x - xf['vmin'] + xreso * 0.5) / xreso)
      #   yid :int = math.floor((y - yf['vmin'] + yreso * 0.5) / yreso)

      #   add_blank = True
      #   if (xid, yid) in record_acc.keys():
      #     if len(record_acc[(xid, yid)]) > grid_cond_num:
      #       acc_array = np.array(record_acc[(xid, yid)])
      #       acc_array = acc_array[acc_array >= 0.0]
      #       if acc_array.shape[0] >= grid_cond_num:
      #         median_mean, intervals = boostrap_sampling(acc_array, (0.25, 0.75))
      #         grid_accs[(xid, yid)] = intervals[0]
      #         rviz_accs.append([x, y, intervals[0]])
      #         add_blank = False
      #   if add_blank:
      #     grid_accs[(xid, yid)] = blank_fill_value
      #     acc_blank_locs.append([x, y])

      # rviz_accs = np.array(rviz_accs)
      # acc_blank_locs = np.array(acc_blank_locs)

      # #####################################################################
      # # rviz
      # fig.set_size_inches(3.5 * 2, 2.163)
      
      # # ax1
      # prob_cmap_key = 'winter'
      # fig_ax1 = fig.add_subplot(121)
      # plot_utils.subfig_reset()
      # plot_utils.axis_set_title(fig_ax1, 'overtaking prob. ', loc_y=1.0)
      # # plot_utils.axis_set_xlabel(fig_ax1, 'initial speed ($m/s$)')
      # plot_utils.axis_set_xlabel(fig_ax1, 'exp. arr. time ($s$)')
      # plot_utils.axis_set_ylabel(fig_ax1, 'exp. arr. time ($s$)')

      # fig_ax1.scatter(
      #   rviz_probs[:, 0], rviz_probs[:, 1], 
      #   c=rviz_probs[:, 2],
      #   vmin=0.0, vmax=1.0, 
      #   cmap=plt.cm.get_cmap(prob_cmap_key), 
      #   marker='s')
      # fig_ax1.scatter(
      #   prob_blank_locs[:, 0], prob_blank_locs[:, 1], c='grey', marker='s')

      # prob_cmap=plt.cm.get_cmap(prob_cmap_key)
      # cmp_norm = matplotlib.colors.BoundaryNorm(
      #   np.linspace(0.0, 1.0, 6), prob_cmap.N)
      # fig.colorbar(matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=prob_cmap),
      #              ax=fig_ax1, shrink=1.0, orientation='vertical')

      # # ax2
      # acc_cmap_key = 'winter'
      # fig_ax2 = fig.add_subplot(122)
      # plot_utils.subfig_reset()
      # plot_utils.axis_set_title(fig_ax2, 'tolerant dcc. ', loc_y=1.0)
      # # plot_utils.axis_set_xlabel(fig_ax2, 'initial speed ($m/s$)')
      # plot_utils.axis_set_xlabel(fig_ax2, 'exp. arr. time ($s$)')
      # plot_utils.axis_set_ylabel(fig_ax2, 'exp. arr. time ($s$)')

      # fig_ax2.scatter(
      #   rviz_accs[:, 0], rviz_accs[:, 1], 
      #   c=rviz_accs[:, 2],
      #   vmin=0.0, vmax=2.5, 
      #   cmap=plt.cm.get_cmap(acc_cmap_key), 
      #   marker='s')
      # fig_ax2.scatter(
      #   acc_blank_locs[:, 0], acc_blank_locs[:, 1], c='grey', marker='s')

      # acc_cmap=plt.cm.get_cmap(acc_cmap_key)
      # cmp_norm = matplotlib.colors.BoundaryNorm(
      #   np.linspace(0.0, 2.5, 6), acc_cmap.N)
      # fig.colorbar(matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=acc_cmap),
      #              ax=fig_ax2, shrink=1.0, orientation='vertical')

      # # save & show
      # plot_utils.save_fig(os.path.expanduser('~') + "/scene_accs", vdpi=600)
      # plt.show()

    def plot_schedule2(self, fig, feature4):
      angle_reso = 10.0
      v_reso = 2.0
      acc_reso = 0.25
      dist_reso = 5.0
      t_reso = 1.0
      prob_cmap_key = 'winter'

      # ax1
      fig.set_size_inches(3.5, 2.163)
      fig_ax1 = fig.add_subplot(111)
      plot_utils.subfig_reset()
      # plot_utils.axis_set_title(fig_ax1, 'dur-avg_acc', loc_y=-0.5)
      # plot_utils.axis_set_xlabel(fig_ax1, 'oppo. dist. ($m$)')
      plot_utils.axis_set_xlabel(fig_ax1, 'v ($m/s$)')
      plot_utils.axis_set_ylabel(fig_ax1, 'avg. acc ($m/s^2$)')

      ###################################
      axis_example = fig_ax1
      xlist = feature4[:, 4]
      xreso = v_reso
      ylist = feature4[:, 1]
      flaglist = feature4[:, 0]

      xf, xidlist = plot_func.axis_feature(xlist, xreso)

      box_dict = dict()
      for y, xid, flag in zip(ylist, xidlist, flaglist):
        if not xid in box_dict:
          box_dict[xid] = []
        box_dict[xid].append([y, flag])
      od_box_dict = collections.OrderedDict(sorted(box_dict.items()))

      x_ticks_labels = []
      x_ticks_poses = []
      box_colors = []

      box_data = []
      get_datas = []
      for key_xid, content in od_box_dict.items():
        _x = key_xid * xreso + xf['vmin']
        content = np.array(content)

        num_sample = content.shape[0]
        yvalues = content[:, 0]
        prob = np.mean(content[:, 1])

        # if prob < 0.975: # has chance to giveway
        box_data.append(yvalues.tolist())
        get_datas.append({
          'key_int': key_xid, 
          'key_double': _x,
          'prob': prob,
          'values': yvalues, 
          'mean': np.mean(yvalues), 
          'std': np.std(yvalues)
        })

        x_ticks_labels.append("{}".format(round(_x, 1)))
        x_ticks_poses.append(_x)
        box_colors.append(plot_func.map_amount2color(num_sample))

      boxpros = dict(linestyle='-', linewidth=0.25, color='k')
      medianprops = dict(linestyle='--', linewidth=1.0, color="g")
      meanprops = dict(linestyle='-', linewidth=1.0, color="g")
      flier_marker=dict(markeredgecolor='blue', # markerfacecolor='red', 
                        marker='o', markersize=1)

      bps = axis_example.boxplot(box_data,
        boxprops=boxpros, 
        positions=x_ticks_poses,
        showfliers=True, # flier: abnormal value
        flierprops=flier_marker,
        meanprops=meanprops,
        medianprops=medianprops,
        # notch=1,
        patch_artist=True, # enable fill with color
        showmeans=True,
        meanline=True,
        widths=(1.0 * 0.4)
      )
      for patch, color in zip(bps['boxes'], box_colors):
        patch.set_facecolor(color)

      plot_utils.axis_set_xticks(axis_example, 
        tick_values=x_ticks_poses, tick_labels=x_ticks_labels)

      #################################################################
      # prob_cmap=plt.cm.get_cmap(prob_cmap_key)
      # cmp_norm = matplotlib.colors.BoundaryNorm(
      #   np.linspace(acc_min, acc_max, 6), prob_cmap.N)
      # fig.colorbar(matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=prob_cmap),
      #              ax=fig_ax1, shrink=1.0, orientation='vertical')

      for content in get_datas:
        print(content['key_double'], content['prob'])

      plot_utils.save_fig(os.path.expanduser('~') + "/[g]v02acc", vdpi=600)
      plt.show()
