from typing import Dict, List, Tuple, Union
import numpy as np
import math
import copy
import abc
from planners.st_search.config import MAP_MODE2WEIGHTSfrom planners.st_search.zone_stv_search import ZoneStvGraphSearchfrom planners.interaction_space import InteractionFormat, InteractionSpacefrom planners.st_search.sti_node import StiNodefrom type_utils.agent import EgoAgent

class ZoneStvGraphSearchContingency(ZoneStvGraphSearch):
  '''
  Speed search based on the S-t-v graph, with interaction zone 
  modelling and priority determination relying on collision checking with prediction results
  '''
  def __init__(self, ego_agent: EgoAgent, 
                     ispace: InteractionSpace, 
                     s_samples: np.ndarray, 
                     xyyawcur_samples: np.ndarray,
                     start_sva: Tuple[float, float, float],
                     search_horizon_T: float, 
                     planning_dt: float,
                     prediction_dt: float,
                     s_end_need_stop: bool,
                     friction_coef: float = 0.35,
                     path_s_interval: float = 1.0,
                     prediction_num: int = 0.0,
                     reaction_conditions: Dict = None,
                     enable_debug_rviz: bool = False) -> None:
    '''
    :param ispace: interaction space used for cost evaluation
    :param s_samples: discretized s values of samples
    :param start_sva: initial s,v,a values of AV: corresponding to self.s_samples[0]
    :param xyyawcur_samples: [[x, y, yaw, curvature], ...] values at s_samples
    :param search_horizon_T: search(plan) horizon T
    :param planning_dt: planning node interval timestamp
    :param prediction_dt: prediction trajectory node interval timestamp
    :param s_end_need_stop: when AV reaches s_samples[-1], is need to stop or change lane in advance.
    :param friction_coef: friction coefficient to calculate speed limit for bended path
    :param path_s_interval: interval s value for s sampling
    '''
    mode_st_coefficents = reaction_conditions['st_coefficents']
    if mode_st_coefficents <= 0:
      super().__init__(
        ego_agent, ispace, s_samples, xyyawcur_samples, start_sva, search_horizon_T, 
        planning_dt, prediction_dt, s_end_need_stop, friction_coef,
        path_s_interval, # one_predtraj_multi_izone=False,
        enable_debug_rviz=enable_debug_rviz)
    else:
      super().__init__(
        ego_agent, ispace, s_samples, xyyawcur_samples, start_sva, search_horizon_T, 
        planning_dt, prediction_dt, s_end_need_stop, friction_coef,
        path_s_interval, # one_predtraj_multi_izone=False,
        svaj_cost_weights=MAP_MODE2WEIGHTS[mode_st_coefficents],
        enable_debug_rviz=enable_debug_rviz)

    self.__short_term_cond_s = 3.0
    self.__prediction_num = int(prediction_num)

  def _edge_is_valid(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the edge's child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      aid_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_idx')
      tid_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_traj_idx')
      s_index = self.zone_part_data_len + InteractionFormat.iformat_index('av_s')
      t_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_t')

      # relation calculations
      # check_izone_array with shape = (edge_check_point_num, zone_info_num)
      # check_stva_array with shape = (child_num, edge_check_point_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, s_index])

      # child_izone_array with shape = (child_num, edge_check_point_num, zone_info_num)
      child_izone_array = np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # relation flag is treated as 
      # 1. relation_not_determined: no_collision with certain zone
      # 2. relation_influ: collision with certain zone
      # 3. other realtions are not used, as contigency-planner default disable the IRD (initial relation determination)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notcollided_yet = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations)

      ################################################################
      # not collided locations
      if np.sum(locs_notcollided_yet) > 0:
        plan_t_array = check_stva_array[locs_notcollided_yet, 1]
        pred_t_array = child_izone_array[locs_notcollided_yet, t_index]

        collision_locs = np.logical_and(
          (pred_t_array - self.safe_time_gap) < plan_t_array,
          plan_t_array < (pred_t_array + self.yield_safe_time_gap)
        )

        valids_locs[locs_notcollided_yet] = np.logical_not(collision_locs)

        # update relationship
        get_relations[locs_notcollided_yet] = collision_locs * StiNode.relation_preempt()
        
        self._update_relations(child_nodes, range_iinfo_dict, get_relations)

        # if parent_node[self._node_key_t] < 1e-2:
        #   print("relation_indexes", range_iinfo_dict['relation_indexes'])
        #   print("relations-1", get_relations.shape, get_relations)
        #   print("relations-2", child_nodes[:, range_iinfo_dict['relation_indexes']])

      ################################################################
      # child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']] # update variables

      ################################################################
      # set valids at long-short distinguished situations
      def update_func(_atinfo, _rr, _stva, _updates):
        if (self.__prediction_num == 1) or (_stva[1] <= self.__short_term_cond_s):
          _updates[0].append(_rr)
        else:
          aidx = int(_atinfo[0])
          tidx = int(_atinfo[1])
          if not aidx in _updates[1]:
            _updates[1][aidx] = [[], [], []]
          _updates[1][aidx][tidx].append(_rr)
        return 0.

      # shapes
      # 1. child_nodes:       shape = (child_num, node_state_num)
      # 2. relations          shape = (child_num, edge_check_point_num)
      # 3. valid_locs         shape = (child_num, edge_check_point_num)
      # 4. check_stva_array   shape = (child_num, edge_check_point_num, 4)
      # 5. valids_locs        shape = (child_num, edge_check_point_num)
      # @note edge_check_point_num == interpolated point num of the edge == relation number
      for child_i, relation in enumerate(get_relations):
        agent_traj_idx = child_izone_array[child_i, :, [aid_index, tid_index]].transpose() # (edge_check_point_num, 2)
        plan_stva = check_stva_array[child_i, :, :] # (edge_check_point_num, 4)

        shortterm_cache = []
        longterm_cache = {}
        # cache: records informations for the edge to this child_node[child_i]
        # cache = [
        #   relation_i : [
        #     []: list of [pred_agent_index, pred_traj_index, relation_result, plan_t_value]
        #   ]
        # ]
        # _ = [[update_func(atinfo, rr, stva, [shortterm_cache, longterm_cache]) for atinfo, rr, stva in zip(
        #     agent_traj_idx[seq_locs, :], relation[seq_locs], plan_stva[seq_locs, :])] \
        #   for relation_i, seq_locs in range_iinfo_dict['relation2ipoint_locs'].items()]
        for relation_i, seq_locs in range_iinfo_dict['relation2ipoint_locs'].items():
          for atinfo, rr, stva in zip(agent_traj_idx[seq_locs, :], relation[seq_locs], plan_stva[seq_locs, :]): 
            update_func(atinfo, rr, stva, [shortterm_cache, longterm_cache])
            # if parent_node[self._node_key_t] < 1e-2:
            #   print(relation_i, stva[1], atinfo, rr)

        # if parent_node[self._node_key_t] < 1e-2:
        #   print("shortterm_cache=", shortterm_cache, valids_locs[child_i, 0])
        #   print("longterm_cache=", longterm_cache, len(longterm_cache))

        if (len(longterm_cache) == 0):
          # illegal when relations of any pred-K has collisions
          valids_locs[child_i, :] = (sum(shortterm_cache) < 1e-2) * 1.0 # not collided, is valid

        else:
          # check long-term when len(short-term) == 0, or short-term all not collided
          enable_checks = (len(shortterm_cache) == 0) or (sum(shortterm_cache) < 1e-2)

          is_illegal = (sum(shortterm_cache) > 1e-2) # short-term exists collisions

          if enable_checks and (is_illegal == False):
            for agent_idx, preds_relations in longterm_cache.items():
              # illegal only when relations of all pred-K has collisions
              is_illegal = (sum(preds_relations[0]) > 1e-2) and\
                          (sum(preds_relations[1]) > 1e-2) and\
                          (sum(preds_relations[2]) > 1e-2)
              if is_illegal:
                break
          if not is_illegal:
            valids_locs[child_i, :] = 1.0

      # print("************************************")

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs
