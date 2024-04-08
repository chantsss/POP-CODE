#!/usr/bin/env python

import enum
import math
import os
import collections
import numpy as np
import copy
import matplotlib.pyplot as pltfrom typing import Any, DefaultDict, Dict, List, Set

import type_utils.state_trajectory as state_trajfrom utils.file_io import write_dict2bin, read_dict_from_binfrom utils.angle_operation import get_normalized_angle, get_normalized_angles

import paper_plot.utils as plot_utils
import paper_plot.functions as plot_func

# constants
M_PI = 3.14159265358979323846
double_M_PI = 6.283185307179586
TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

# parameters
grid_xy_reso = 2.0
grid_yaw_reso = 20.0 * TO_RADIAN
grid_dangle_reso = 2.0 * TO_RADIAN
grid_v_reso = 0.5
grid_t_reso = 0.5
grid_acc_reso = 0.2
max_v_id = 20 # [0, 20]
max_t_id = 8 # [0, 8]
grid_x_range = 100.0
grid_y_range = 100.0
yaw_range = 360 * TO_RADIAN

# mid variables
sample_v_length = max_v_id + 1
sample_t_length = max_t_id + 1
grid_xy_reso_2 = grid_xy_reso * 0.5
grid_yaw_reso_2 = grid_yaw_reso * 0.5
grid_dangle_reso_2 = grid_dangle_reso * 0.5
grid_v_reso_2 = grid_v_reso * 0.5
grid_t_reso_2 = grid_t_reso * 0.5
grid_acc_reso_2 = grid_acc_reso * 0.5
grid_x_range_2 = grid_x_range * 0.5
grid_y_range_2 = grid_y_range * 0.5
yaw_range_2 = yaw_range * 0.5
grid_x_len = round(grid_x_range / grid_xy_reso)
grid_y_len = round(grid_y_range / grid_xy_reso)
grid_yaw_len = round(yaw_range / grid_yaw_reso)
grid_xy_len = grid_x_len * grid_y_len

class TrajectoryAnalysis():
    '''
    A trajectory analyzer to analyze/plot features given traj_list, which
    store features in the form of
      self.float_features
      self.space_features
      self.g_g_features
    For more details, please see add_traj_list()
    '''

    def __init__(self, workspace: str) -> None:
        '''
        :param workspace: directory to store/read bin data
        '''
        self.workspace = workspace
        self.full_yaw_array = np.arange(-M_PI, M_PI, grid_yaw_reso)
        self.full_yaw_ids = [math.floor((_yaw + grid_yaw_reso_2 + yaw_range_2) / grid_yaw_reso) \
                                        for _yaw in self.full_yaw_array]
        self.reinit()

    def reinit(self):
        self.float_features: List[Any] = []         # features in floats
        self.space_features: Dict[Any, Any] = {}    # features in grids
        self.g_g_features = [[], [], [], []]        # features for g-g diagram 

    def get_path2file(self, file_name: str):
        return os.path.join(self.workspace, file_name)

    def get_indexs(self, array_x, array_y, array_yaw):
        xids = np.floor((array_x + grid_xy_reso_2) / grid_xy_reso)
        yids = np.floor((array_y + grid_xy_reso_2 + grid_y_range_2) / grid_xy_reso)
        yawids = np.floor((get_normalized_angles(\
                           array_yaw + grid_yaw_reso_2) + yaw_range_2) / grid_yaw_reso)
        return xids, yids, yawids

    def get_index3d(self, id_x: int, id_y: int, id_yaw: int):
        return (id_yaw * grid_xy_len + id_y * grid_x_len + id_x)

    def get_index2d(self, id_x: int, id_y: int):
        return (id_y * grid_x_len + id_x)

    def write_data2bin(self):
        write_dict2bin(self.float_features, self.get_path2file("float_features.bin"))
        write_dict2bin(self.space_features, self.get_path2file("space_features.bin"))
        write_dict2bin(self.g_g_features, self.get_path2file("g_g_features.bin"))

    def read_data_frome_bin(self):
        self.float_features = read_dict_from_bin(self.get_path2file("float_features.bin"))
        self.space_features = read_dict_from_bin(self.get_path2file("space_features.bin"))
        self.g_g_features = read_dict_from_bin(self.get_path2file("g_g_features.bin"))

    def add_traj_list(self, trajs_list: Dict, 
                            take_len=-1, dt=0.1,
                            filter_traj_min_num=10,
                            filter_traj_min_len=3,
                            filter_length=[3.0, 7.0],
                            ):
        '''
        Add a list of trajectories to the class
        :param trajs_list: {v0: traj}, where each traj is a list of [[agent_info], [x, y, yaw, v, time_s]_x, ...]
                                       and v's sign represents the direction.
        :param take_len: number of trajectory points being added
        :param dt: inverval time of the trajectory points
        :param filter_traj_min_num: len(trajs_list[key]) < filter_traj_min_num will be abandoned
        :param filter_traj_min_len: len(traj) < filter_traj_min_len will be abandoned
        :param filter_length: the agent with length out of filter_length will be abandoned
        '''
        od_trajs_list = collections.OrderedDict(sorted(trajs_list.items()))

        trajs_total_number = 0
        for key_start_v, trajs in od_trajs_list.items():
          traj_num = len(trajs)
          if (traj_num < filter_traj_min_num): continue

          traj_i = 0
          for traj in trajs:
            print("start_v={}m/s with traj_num={}/{};".format(key_start_v, 
                                                              traj_i, traj_num))
            traj_i += 1

            cache = state_traj.StateTrajectory()
            cache.set_trajectory_list(traj)
            info = cache.get_info()

            this_shape = [info.length, info.width, 1.0]
            traj_len = traj.shape[0] - 1
            if traj_len < filter_traj_min_len:
              continue # skip too short traj

            length = round((this_shape[0] + 0.1) / 0.2) * 0.2
            dict_length = math.floor((length + 0.5) / 1.0) * 1.0
            if (dict_length < filter_length[0]) or (dict_length > filter_length[1]):
              continue # skip when out of len limits

            trajs_total_number += 1
            tstampmax = (traj_len - 1) * dt
            length = dict_length # set as dict_length

            # init data
            if not length in self.space_features.keys():
              self.space_features[length] = [0, dict(), []]
            self.space_features[length][0] += 1
            grids_map3d = self.space_features[length][1]
            grids_features = self.space_features[length][2]

            if take_len < 0:
              start_state = traj[1, :]
              end_state = traj[-1, :]
              xy_list = np.array(traj[1:, 0:2])
              yaw_list = np.array(traj[1:, 2])
              v_list = np.array(traj[1:, 3])
              ts_list = np.array(traj[1:, 4])
            else:
              start_state = traj[1, :]
              if take_len >= traj_len:
                end_state = traj[-1, :]
              else:
                end_state = traj[take_len-1, :]
              xy_list = np.array(traj[1:take_len, 0:2])
              yaw_list = np.array(traj[1:take_len, 2])
              v_list = np.array(traj[1:take_len, 3])
              ts_list = np.array(traj[1:take_len, 4])

            # fill data
            euler_dists = np.power((xy_list[1:] - xy_list[0:-1]), 2)
            euler_dists = np.sqrt(euler_dists[:, 0] + euler_dists[:, 1])
            sum_s = np.sum(euler_dists)
            sum_t = ts_list[-1] - ts_list[0]

            sum_s_list = np.zeros([1])
            sum_s_list = np.hstack((sum_s_list, euler_dists))
            for i in range(1, len(sum_s_list)):
              sum_s_list[i] = sum_s_list[i-1] + sum_s_list[i]
            
            diff_v = v_list[-1] - v_list[0]
            start_v = start_state[3]
            avg_v = np.mean(v_list)
            var_v = np.var(v_list)
            min_v = np.min(v_list)
            max_v = np.max(v_list)

            INTERVAL = 5
            dtt = dt * INTERVAL
            len_list = len(v_list)
            a_lon_list = [0.0]
            a_lat_list = [0.0]
            a_lat_corr_v_list = [0.0] # v list to calculate a_lat 
            a_lat_corr_kappa_list = [0.0]  # kappa list to calculate a_lat 
            if (len_list > INTERVAL):
              a_lon_list = []
              a_lat_list = []
              a_lat_corr_v_list = []
              a_lat_corr_kappa_list = []
              cache = v_list[::INTERVAL]
              dv_list = cache[1:] - cache[0:-1]
              cache = yaw_list[::INTERVAL]
              dyaw_list = cache[1:] - cache[0:-1]
              cache = sum_s_list[::INTERVAL]
              ds_list = cache[1:] - cache[0:-1]

              # print(len(v_list[::INTERVAL]), len(dv_list), len(ds_list))
              # len(v_list[::INTERVAL]) == len(dv_list) + 1
              for dv in dv_list:
                a_lon_list.append(dv / dtt)

              for v, dyaw, ds in zip(v_list[::INTERVAL], dyaw_list, ds_list):
                ds_flag: bool = (ds < 0.1)
                abs_kapaa = 0.0
                a_lat = 0.0
                cal_v = v
                if not ds_flag:
                  abs_kapaa = math.fabs(dyaw / ds)
                  a_lat = v * v * abs_kapaa

                a_lat_list.append(a_lat)
                a_lat_corr_v_list.append(cal_v)
                a_lat_corr_kappa_list.append(abs_kapaa)

            avg_a = 0.0
            min_a, max_a = 0.0, 0.0
            start_a = a_lon_list[0]

            avg_dradian = get_normalized_angle(end_state[2] - start_state[2])
            abs_avg_dradian = abs(avg_dradian)
            abs_avg_dradian_m = 0.
            if (sum_s > 1e-1):
              abs_avg_dradian_m = abs_avg_dradian / sum_s
            abs_avg_dradian_s = abs_avg_dradian / sum_t

            # <format> of self.float_features
            if (start_v >= 1e-3) and (avg_v >= 1e-3):
              self.float_features.append([length, start_v, start_a,             # 0, 1, 2
                                          sum_s, sum_t,                         # 3, 4
                                          abs_avg_dradian_s, abs_avg_dradian_m, # 5, 6
                                          avg_v, min_v, max_v, var_v,           # 7, 8, 9, 10
                                          diff_v, avg_a]                        # 11, 12
              )
              check_bool = len(a_lon_list) == len(a_lat_list)
              if check_bool:
                a_lon_list = np.array(a_lon_list)
                a_lat_list = np.array(a_lat_list)
                a_lat_corr_v_list = np.array(a_lat_corr_v_list)
                a_lat_corr_kappa_list = np.array(a_lat_corr_kappa_list)

                for a_lon, a_lat, a_lat_corr_v, a_lat_corr_kappa in\
                    zip(a_lon_list, a_lat_list, a_lat_corr_v_list, a_lat_corr_kappa_list):
                  self.g_g_features[0].append(a_lon)
                  self.g_g_features[1].append(a_lat)
                  self.g_g_features[2].append(a_lat_corr_v)
                  self.g_g_features[3].append(a_lat_corr_kappa)
              else:
                print("warning, check_bool == false, with {}/{}/{}".format(
                  len(a_lon_list), len(a_lat_list))
                )

            # traverse the trajectory
            # @format tpoint = [x, y, yaw, v]
            xids, yids, yawids = self.get_indexs(traj[1:, 0], traj[1:, 1], traj[1:, 2])
            vids = np.floor((traj[1:, 3] + grid_v_reso_2) / grid_v_reso)
            tstamp = 0.0 # reinit to zero
            for tsum_s, tv, xid, yid, yawid, v_id in zip(sum_s_list, traj[1:, 3],
                                                         xids, yids, yawids, vids):
              # yids = np.floor((array_y + grid_xy_reso_2 + grid_y_range_2) / grid_xy_reso)
              _x = xid * grid_xy_reso
              _y = yid * grid_xy_reso + (-grid_y_range_2)
              _yaw = yawid * grid_yaw_reso + (-yaw_range_2)
              v_id = min(int(v_id), max_v_id) # get v_id
              t_id = round(tstamp / grid_t_reso) # get t_id
              t_id = min(int(t_id), max_t_id)
              tstamp = tstamp + dt

              # array_id2d = [int(xid), int(yid)]
              # id2d = self.get_index2d(array_id2d[0], array_id2d[1])
              array_id3d = [int(xid), int(yid), int(yawid)]
              id3d = self.get_index3d(array_id3d[0], array_id3d[1], array_id3d[2])

              # <format> of self.space_features[length]
              if not id3d in grids_map3d.keys():
                grids_map3d[id3d] = len(grids_features)
                # cache statistics feature values
                #   [start_v, x, y, yaw, vmax, tmin, tmax, smin, smax]
                grids_features.append([start_v, _x, _y, _yaw, 
                                       0.0, 1e+3, 0.0, 1e+3, 0.0])

              gdta = grids_features[grids_map3d[id3d]]
              grids_features[grids_map3d[id3d]][4] = max(gdta[4], tv) # vmax
              grids_features[grids_map3d[id3d]][5] = min(gdta[5], tstamp) # tmin
              grids_features[grids_map3d[id3d]][6] = max(gdta[6], tstamp) # tmax
              grids_features[grids_map3d[id3d]][7] = min(gdta[7], tsum_s) # smin
              grids_features[grids_map3d[id3d]][8] = max(gdta[8], tsum_s) # smax

    def plot_v0_dradian_features(self, fig_path=None):
        v_reso = 0.5
        v_reso_2 = v_reso * 0.5

        cmap_mode='rainbow'
        min_num_samples = 100

        # [length, start_v, start_a,             # 0, 1, 2
        #  sum_s, sum_t,                         # 3, 4
        #  abs_avg_dradian_s, abs_avg_dradian_m, # 5, 6
        #  avg_v, min_v, max_v, var_v,           # 7, 8, 9, 10
        #  diff_v, avg_a]                        # 11, 12
        features = np.array(self.float_features) # N x feature_num

        dt_v0_to_features: Dict[int, List[float]] = {}
        i = 0
        num = features.shape[0]
        for feature in features:
          print("\rplot, processing features {}/{};".format(i, num), end="")
          i += 1
          start_v = feature[1]
          avg_v = feature[7]
          abs_avg_dradian_m = feature[6]
          if (start_v < 1e-3): 
            continue
          if (avg_v < 0):
            print("unexpected avg_v < 0", start_v, avg_v)
          key_start_vi = int(math.floor((start_v + v_reso_2) / v_reso))

          if not key_start_vi in dt_v0_to_features:
            dt_v0_to_features[key_start_vi] = []
          dt_v0_to_features[key_start_vi].append([avg_v, abs_avg_dradian_m])
        print("")
        
        fig = plt.figure()
        fig.set_size_inches(3.5, 2.163) # golden ratio= 0.618
        plot_utils.fig_reset()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        plot_utils.subfig_reset()
        plot_utils.axis_set_xlabel(ax1, labelpad=1, label="$v_0$ (m/s)")
        plot_utils.axis_set_ylabel(ax1, labelpad=1, label="Avg.$v$ (m/s)")
        plot_utils.axis_set_ylabel(ax2, labelpad=1, label="Max. $v_\kappa$ (m/s)")
        curve_v0_max_k_v = []
        for key_v0_i, features in dt_v0_to_features.items():
          print("\rplot, plotting key index={};".format(key_v0_i), end="")
          key_v0 = (key_v0_i + 1) * v_reso
          features = np.array(features)

          avg_v_ids = np.floor((features[:, 0] + v_reso_2) / v_reso)
          id_max = int(np.max(avg_v_ids) + 1)
          id_min = int(np.min(avg_v_ids) + 1)
          
          avg_v_grid_num = int((id_max - id_min) + 1)
          avg_v_grids = np.zeros(avg_v_grid_num)
          
          avg_v0_points_num = avg_v_ids.shape[0]
          for id in avg_v_ids:
            avg_v_grids[int(id - id_min)] += 1
          avg_v_grids = avg_v_grids / float(avg_v0_points_num)

          # plot v0-avg_v distribution
          x = np.ones_like(avg_v_grids) * (key_v0_i + 1)
          y = np.arange(avg_v_grids.shape[0]) + id_min
          plt.scatter(x, y, c=avg_v_grids,
                      vmin=0.0, vmax=1.0, cmap=plt.cm.get_cmap(cmap_mode), 
                      marker='s')

          # plot avg_v-max_abs dradian
          # print(avg_v_ids.shape, features.shape) # (659,) (659, 2)
          dt: Dict[int, List[float]] = {}
          for i, avg_v_id in enumerate(avg_v_ids):
            abs_avg_dradian_m = features[i, 1] # avg_kappa
            if not avg_v_id in dt:
              dt[avg_v_id] = []
            dt[avg_v_id].append(abs_avg_dradian_m)

          num_samples = avg_v_ids.shape[0]
          mean = np.mean(features[:, 1])
          std = np.std(features[:, 1])
          cache_max = np.max(features[:, 1]) * 0.95
          max_kappa = cache_max # min(mean + 3.0 * std, cache_max)

          if num_samples >= min_num_samples:
            corr_avg_v = None
            corr_avg_v_id = None
            for avg_v_id, list_floats in dt.items():
              avg_v = (avg_v_id + 1) * v_reso
              list_floats = np.array(list_floats)

              nums = np.sum(list_floats >= max_kappa)
              if nums > 0:
                corr_avg_v_id = avg_v_id
                corr_avg_v = avg_v
                break

            # print("check=", key_v0, corr_avg_v, max_kappa)
            curve_v0_max_k_v.append([key_v0_i+1, corr_avg_v_id+1])

        # print("")
        curve_v0_max_k_v = np.array(curve_v0_max_k_v)
        ax2.plot(curve_v0_max_k_v[:, 0], curve_v0_max_k_v[:, 1], '.r')
        ax2.plot(curve_v0_max_k_v[:, 0], curve_v0_max_k_v[:, 0], ':k')
        plt.colorbar(orientation="horizontal", pad=0.1)
        plt.show()

    def plot_gg_diagram(self, acc_reso=0.2):
      features = np.array(self.g_g_features)

      fig = plt.figure()
      fig.set_size_inches(3.5, 2.163) # golden ratio= 0.618
      plot_utils.fig_reset()
      fig_ax1 = fig.add_subplot(111)
      plot_utils.subfig_reset()

      plot_utils.axis_set_xlabel(fig_ax1, "lon. acc. m/s$^2$")
      plot_utils.axis_set_ylabel(fig_ax1, "lat. acc. m/s$^2$")
      plot_func.plot_heatmap2d(features[0, :], acc_reso,
                               features[1, :], acc_reso)

      plt.colorbar()
      plt.show()

    def plot_v_a_lat(self, v_reso=0.5, acc_reso=0.2):
      features = np.array(self.g_g_features)

      fig = plt.figure()
      fig.set_size_inches(3.5, 2.163) # golden ratio= 0.618
      plot_utils.fig_reset()
      fig_ax1 = fig.add_subplot(111)
      plot_utils.subfig_reset()

      plot_utils.axis_set_xlabel(fig_ax1, "velocity m/s")
      plot_utils.axis_set_ylabel(fig_ax1, "lat. acc. m/s$^2$")
      plot_func.plot_heatmap2d(features[2, :], v_reso,
                               features[1, :], acc_reso,
                               full_color_num=1000)
      # plot_func.plot_boxplot(fig_ax1, features[2, :], v_reso, features[1, :])

      plt.show()

    def plot_v_kappa(self, v_reso=0.5, kappa_reso=0.001):
      features = np.array(self.g_g_features)

      fig = plt.figure()
      fig.set_size_inches(3.5, 2.163) # golden ratio= 0.618
      plot_utils.fig_reset()
      fig_ax1 = fig.add_subplot(111)
      plot_utils.subfig_reset()

      plot_utils.axis_set_xlabel(fig_ax1, "velocity m/s")
      plot_utils.axis_set_ylabel(fig_ax1, "kappa rad./m")
      plot_func.plot_heatmap2d(features[2, :], v_reso,
                               features[3, :], kappa_reso,
                               full_color_num=1000)

      plt.show()

    # def plot_v_lat_v(self, v_reso=0.5, lat_v_reso=0.2):
    #   features = np.array(self.g_g_features)

    #   lat_v = []

    #   fig = plt.figure()
    #   fig.set_size_inches(3.5, 2.163) # golden ratio= 0.618
    #   plot_utils.fig_reset()
    #   fig_ax1 = fig.add_subplot(111)
    #   plot_utils.subfig_reset()

    #   plot_utils.axis_set_xlabel(fig_ax1, "velocity m/s")
    #   plot_utils.axis_set_ylabel(fig_ax1, "kappa rad./m")
    #   plot_func.plot_heatmap2d(features[2, :], v_reso,
    #                            features[3, :], kappa_reso,
    #                            full_color_num=1000)

    #   plt.show()

