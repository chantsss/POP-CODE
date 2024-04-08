from typing import Dict, List, Tuple, Union
import numpy as np
import math
import copy
import abc
from planners.st_search.config import MAP_MODE2WEIGHTSfrom planners.st_search.zone_stv_search import ZoneStvGraphSearchfrom planners.interaction_space import InteractionFormat, InteractionSpacefrom planners.st_search.sti_node import StiNodefrom type_utils.agent import EgoAgentfrom models.ipm.dt_model import IPMDtModel
from files.react_t2acc_table import assert_dict_name_is_legal, get_reaction_acc_values

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

class ZoneStvGraphSearchIZoneRelations(ZoneStvGraphSearch):
  '''
  Speed search based on the S-t-v graph, with interaction zone 
  modelling and priority determination relying on longitudinal responses along prediction results
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
                     algo_vars: Tuple = [0.0, 0.0, 0.0],
                     prediction_num: int = 1,
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
    :param prediction_num: variable for algorithm
    :param reaction_conditions: reaction conditions for agents
    '''
    tmode = reaction_conditions['traj_mode']
    judge_init_relation = True
    if ('#' in tmode):
      tmode = tmode[:-1]
      judge_init_relation = False # disable judge init relation.
    self.__ignore_influ_cons = reaction_conditions['ignore_influ_cons']

    mode_st_coefficents = reaction_conditions['st_coefficents']
    if mode_st_coefficents <= 0:
      super().__init__(
        ego_agent, ispace, s_samples, xyyawcur_samples, start_sva, search_horizon_T, 
        planning_dt, prediction_dt, s_end_need_stop, friction_coef,
        path_s_interval, 
        enable_judge_init_relation=judge_init_relation,
        enable_debug_rviz=enable_debug_rviz)
    else:
      super().__init__(
        ego_agent, ispace, s_samples, xyyawcur_samples, start_sva, search_horizon_T, 
        planning_dt, prediction_dt, s_end_need_stop, friction_coef,
        path_s_interval, svaj_cost_weights=MAP_MODE2WEIGHTS[mode_st_coefficents],
        enable_judge_init_relation=judge_init_relation,
        enable_debug_rviz=enable_debug_rviz)
  
    # Algorithm variable
    self.__ireact_gap_cond_t = algo_vars[0]
    self.__ireact_delay_s = algo_vars[1]
    self.__influ_react_min_acc = algo_vars[2]
    # self.__ireact_coef = algo_vars[?] / float(prediction_num)
    
    self.__reaction_cond_acc = reaction_conditions['acc_const_value']
    self.__relation_cond_func = None
    self.__cond_low_speed = 0.2
    self.__ignore_cond_iangle = 155 * TO_RADIAN

    valid_modes = ['cvm', 'cam', 'pred', 'pred_cvm', 'irule']

    assert tmode in valid_modes,\
      "fatal value overtake_giveway_mode, {} / {}".format(tmode, valid_modes)
    self.__using_cvm = (tmode == 'cvm')
    self.__using_cam = (tmode == 'cam')
    self.__using_pred = (tmode == 'pred')
    self.__using_pred_cvm = (tmode == 'pred_cvm')
    self.__using_irule = (tmode == 'irule')

    if reaction_conditions['acc_mode'] == 'const':
      self.__relation_cond_func = self.__get_constant_giveway_cond_acc
    elif 'v0-acc' in reaction_conditions['acc_mode']:
      acc_key = reaction_conditions['acc_mode'].split('-')
      acc_key = acc_key[-1]
      self.__relation_cond_func = self.__get_poly_v0_to_acc

      self.__poly_bounds = np.array([13.5, 19.5])
      self.__poly_v02acc_2sigma = np.array(
        [-2.62101720e-04, 1.40339451e-02, -2.52707644e-01, 1.74195825e+00, -4.79683323e+00])
      self.__poly_v02acc_1sigma = np.array(
        [-1.82504921e-04, 9.52942765e-03, -1.66791717e-01, 1.10976350e+00, -2.99202548e+00])
      self.__poly_v02acc_05sigma = np.array(
        [-1.42706521e-04, 7.27716892e-03, -1.23833754e-01, 7.93666123e-01, -2.08962160e+00])

      if acc_key == 'sigma2':
        self.__poly_v02acc_coefs = self.__poly_v02acc_2sigma
      elif acc_key == 'sigma1':
        self.__poly_v02acc_coefs = self.__poly_v02acc_1sigma
      elif acc_key == 'sigma05':
        self.__poly_v02acc_coefs = self.__poly_v02acc_05sigma
      else:
        raise NotImplementedError("acc mode value illegal.")

      self.__poly_bvalues = np.polyval(self.__poly_v02acc_coefs, self.__poly_bounds)
    elif 'idur-acc' in reaction_conditions['acc_mode']:
      self.__idur2acc_min_t = 0.0
      self.__idur2acc_max_t = 6.0
      self.__idur2acc_min_acc = -4.0
      self.__idur2acc_max_acc = self.__reaction_cond_acc
      self.__idur2acc_coef = 4.0
      self.__idur2acc_dist_norm =\
        np.power(self.__idur2acc_max_t - self.__idur2acc_min_t, self.__idur2acc_coef)
      self.__idur2acc_value_norm = (2.0 / (1 + np.exp(self.__idur2acc_coef * self.__idur2acc_max_t)) - 1.0)

      self.__relation_cond_func = self.__get_poly_idur2acc
    else:
      key_mode = reaction_conditions['acc_mode']
      assert_dict_name_is_legal(key_mode)
      self.__table_key_mode = key_mode

      self.__relation_cond_func = self.__get_table_giveway_cond_acc

    self.__ipm_dt = IPMDtModel()

    self.__s_index = self.zone_part_data_len + InteractionFormat.iformat_index('av_s')
    self.__pred_s_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_s')
    self.__v0_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_v0')
    self.__t_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_t')
    self.__acc_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_acc')
    self.__iangle_index = self.zone_part_data_len + InteractionFormat.iformat_index('iangle')

    self.__agent_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_idx')
    self.__agent_traj_index = self.zone_part_data_len + InteractionFormat.iformat_index('agent_traj_idx')
    self.__tsv_indexs = StiNode.tsv_record_indexs_tiled()
    self.__tsv_t_indexs, self.__tsv_s_indexs, self.__tsv_v_indexs = StiNode.tsv_relative_idexes()

    self.__edge_check_func = None
    if self.__using_cvm:
      self.__edge_check_func = self.__edge_is_valid_cvm
    elif self.__using_pred:
      self.__edge_check_func = self.__edge_is_valid_pred
    elif self.__using_pred_cvm:
      self.__edge_check_func = self.__edge_is_valid_predcvm
    elif self.__using_cam:
      self.__edge_check_func = self.__edge_is_valid_cam
    elif self.__using_irule:
      self.__edge_check_func = self.__edge_is_valid_irule

  def __get_constant_giveway_cond_acc(self, pred_v0_array: np.ndarray, idur_array: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Return constant giveway condition acc value
    '''
    return self.__reaction_cond_acc

  def __get_poly_v0_to_acc(self, pred_v0_array: np.ndarray, idur_array: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Calculate and return polyfited giveway condition acc value
    '''
    cond_accs = np.polyval(self.__poly_v02acc_coefs, pred_v0_array)
    cond_accs[cond_accs <= self.__poly_bounds[0]] = self.__poly_bvalues[0]
    cond_accs[cond_accs >= self.__poly_bounds[1]] = self.__poly_bvalues[1]

    return cond_accs

  def __get_poly_idur2acc(self, pred_v0_array: np.ndarray, idur_array: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Calculate and return polyfited giveway condition acc value
    '''
    idur_array[idur_array < self.__idur2acc_min_t] = self.__idur2acc_min_t
    idur_array[idur_array > self.__idur2acc_max_t] = self.__idur2acc_max_t

    dist2_min_t = np.power((idur_array - self.__idur2acc_min_t), self.__idur2acc_coef)
    dist2_max_t = np.power((self.__idur2acc_max_t - idur_array), self.__idur2acc_coef)
    
    cond_acc = 1.0 - (2.0 / (1 + np.exp(self.__idur2acc_coef * idur_array)) - 1.0) / self.__idur2acc_value_norm
    cond_acc = 0.0 * dist2_min_t / self.__idur2acc_dist_norm + 1.0 * dist2_max_t / self.__idur2acc_dist_norm
    cond_acc = cond_acc * (self.__idur2acc_min_acc - 1.0 * self.__idur2acc_max_acc) + self.__idur2acc_max_acc
    return cond_acc

  def __get_table_giveway_cond_acc(self, 
      plan_v0: float, plan_s_array: np.ndarray, pred_v0_array: np.ndarray, pred_s_array: np.ndarray) -> Union[float, np.ndarray]:
    '''
    Return constant giveway condition acc value according to TABLE(plan_v0, plan_s, pred_v0, pred_s)
    '''
    raise NotImplementedError("__get_table_giveway_cond_acc::not implemented()")

    return get_reaction_acc_values(
      self.__table_key_mode, 
      agent_i_v0=pred_v0_array, agent_i_move_s=pred_s_array, 
      agent_j_v0=plan_v0, agent_j_move_s=plan_s_array, 
      default_acc=self.__reaction_cond_acc)

  def __get_tsv_record(self, child_nodes: np.ndarray, inquiry_s: np.ndarray) -> np.ndarray:
    '''
    Interpolate node value given s values.
    :param parent_node: parent node of the graph edge in searching
    :param child_nodes: child nodes of the graph edge in searching
    :param inquiry_s: array like inquiry s values, values should inside [paren_node.s, np.max(child_nodes.s)]
    :return: [[s, t, v, a], ...] values given inquiry_s for [child0, child1, ...]
    '''
    cache = child_nodes[:, self.__tsv_indexs]
    record_tsv = np.repeat(cache[:, np.newaxis, :], inquiry_s.shape[0], axis=1)

    # print("record_tsv=", record_tsv.shape)
    # print(record_tsv[:, :, [1, 4, 7]])
    return record_tsv

  def _edge_is_valid(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    return self.__edge_check_func(range_iinfo_dict, parent_node, child_nodes)

  def __process_influ_constraints(self, locs_influ: np.ndarray, 
      check_stva_array: np.ndarray, child_izone_array: np.ndarray, valids_locs: np.ndarray):

    if np.sum(locs_influ) > 0:
      if self.__ignore_influ_cons:
        valids_locs[locs_influ] = True # set all true
      else:
        plan_t_array = check_stva_array[locs_influ, 1]
        pred_v0_array = child_izone_array[locs_influ, self.__v0_index]
        pred_s_array = child_izone_array[locs_influ, self.__pred_s_index]

        if (math.fabs(self.__influ_react_min_acc) >= 1e-1):
          # using cam
          ddd = np.square(pred_v0_array) + 2.0*self.__influ_react_min_acc*pred_s_array
          ddd_valids = ddd >= 0.0 # give dcc, vt >= 0.0
          iacc_arrive_t = np.ones_like(pred_v0_array) * 1e+3 # ddd_invalids are inf
          iacc_arrive_t[ddd_valids] = (-pred_v0_array[ddd_valids] + np.sqrt(ddd[ddd_valids])) / self.__influ_react_min_acc

          valids_locs[locs_influ] =\
            (iacc_arrive_t >= (plan_t_array + self.yield_safe_time_gap))
        else:
          # using cvm
          pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
          pred_cvm_t = pred_s_array / pred_v0_array

          valids_locs[locs_influ] =\
            (pred_cvm_t >= (plan_t_array + self.yield_safe_time_gap))

        # #####################################
        # # calculate influ cost
        # tmp_d1 = np.zeros_like(check_stva_array)
        # tmp_d1[notdter_set_inlu_locs, 1] = 1.0
        # new_set_influ_locs = tmp_d1[locs_influ, 1] > 0.5

        # agent_yield_min_dts = plan_t_array + self.yield_safe_time_gap
        # agent_yield_react_acc = 2.0 *\
        #   (pred_s_array - pred_v0_array * agent_yield_min_dts) /\
        #     np.square(agent_yield_min_dts)

        # # when t = agent_yield_min_dts, a = agent_yield_react_acc
        # # s = 0.5*(v0 + vt) * t > check vt value
        # cache_s = pred_s_array
        # cache_dur = agent_yield_min_dts
        # cache_square_v0 = np.square(pred_v0_array)

        # cache_react_vt = 2.0 * cache_s / cache_dur - pred_v0_array
        # assign_t_valids = cache_react_vt >= 0.0 # t == agent_yield_min_dts, is the minimum arrival time given agent_yield_react_acc
        # # assign_t_invalids = cache_react_vt < 0.0 # exist solution of t < agent_yield_min_dts, that acc = agent_yield_react_acc

        # react_dccs = np.ones_like(pred_v0_array) * self.search_acc_bounds[0]
        # ddd = cache_square_v0 + 2.0*react_dccs*cache_s
        # ddd_valids = ddd >= 0.0 # give react_dccs, vt >= 0.0
        # ddd_invalids = np.logical_not(ddd_valids) # give react_dccs, vt < 0.0

        # # update acc values at assign_t_valids == false
        # react_dccs[ddd_invalids] = -cache_square_v0[ddd_invalids] / (2.0 * cache_s[ddd_invalids])
        # react_dccs[assign_t_valids] = agent_yield_react_acc[assign_t_valids] # copy the valid original acc values
        # agent_yield_react_acc = react_dccs

        # # update t values at assign_t_valids == false
        # ref_dt = np.ones_like(pred_v0_array) * 1e+3 # ddd_invalids are inf
        # ref_dt[ddd_valids] = (-pred_v0_array[ddd_valids] + np.sqrt(ddd[ddd_valids])) / react_dccs[ddd_valids]
        # ref_dt[assign_t_valids] = agent_yield_min_dts[assign_t_valids] # copy the valid original t values
        # agent_yield_dts = ref_dt

        # # calculate reaction cost of agents
        # norm_t = (plan_t_array - parent_node[self._node_key_t]) / self.search_horizon_T
        # # new influ locs with full horizon
        # norm_t[new_set_influ_locs] = plan_t_array[new_set_influ_locs] / self.search_horizon_T

        # cost_react_acc = agent_yield_react_acc
        # cost_react_acc[cost_react_acc > 0.0] = 0.0 # only dcc values introduces costs
        # costs_locs[locs_influ] =\
        #   self.__ireact_coef * norm_t * np.square(cost_react_acc) * agent_yield_dts

  def __process_relation_conflicts(self, range_iinfo_dict: Dict, get_relations: np.ndarray, valids_locs: np.ndarray) -> None:
    # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)

    for child_i, relation in enumerate(get_relations):
      # relation = [1, zone_and_interp_num], records relation_location_at_node x zone_and_interp_num
      #
      # cache shape = [zone_num, zone's multiple relation determination results]
      #   relation[ipoint_locs] returns [relation value] for each relation (zone)
      # 
      # > 0.3: aims to remove == 0.0, 0.25 relation (0.25 is influ relation)
      cache = [[v for v in relation[ipoint_locs] if math.fabs(v) > 0.3] \
        for relation_i, ipoint_locs in range_iinfo_dict['relation2ipoint_locs'].items()]
      # extract unqiue relations, e.g. [-1.0], [1.0], [-1.0, 1.0]
      # where [-1.0, 1.0] means have preempt and yield at meantime.

      # child_nodes.shape = (child_num, node_state_num)
      # valids_locs.shape = (child_num, zone_and_interp_num)
      # print("CONFLICT::DEBUG", child_nodes.shape, valids_locs.shape, len(cache))
      # print(cache)
      # example = (4, 21) (4, 13) 2,
      #         = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]]
      # two interaction zones with [8, 5] overlaps with AV's planned trajectory,
      #   and their relations are [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0] respectively

      # check ipoints corresponding to each relation_index has unique preempt/yeild relation 
      # influ is ignored when couting the relation number
      cache_unique_len = []
      for cc in cache:
        unique_cc = np.unique(cc)
        unique_cc_list = unique_cc.tolist()
        len_unique_values :int= unique_cc.shape[0]
        if (StiNode.relation_influ() in unique_cc_list) and (StiNode.relation_preempt() in unique_cc_list):
          len_unique_values -= 1
        cache_unique_len.append(len_unique_values)

      max_unique_num = max(cache_unique_len)
      if max_unique_num >= 2:
        valids_locs[child_i, :] = 0.0 # set all invalid along of this edge check point

  def __edge_is_valid_cvm(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        pred_v0_array = child_izone_array[locs_notdeter, self.__v0_index]
        pred_s_array = child_izone_array[locs_notdeter, self.__pred_s_index]

        pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / pred_v0_array

        preempt_locs = plan_t_array <= (pred_cvm_t - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_cvm_t + self.yield_safe_time_gap)

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() +\
          yield_locs * StiNode.relation_yield()

        self._update_relations(child_nodes, range_iinfo_dict, get_relations)
        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, yield_locs) * 1.0 # set valids

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_v0_array = child_izone_array[locs_preempt, self.__v0_index]
        pred_s_array = child_izone_array[locs_preempt, self.__pred_s_index]

        pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / pred_v0_array

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition
        valids_locs[locs_preempt] = np.logical_and(
          plan_t_array <= (pred_cvm_t - self.safe_time_gap), 
          not2stop_array)

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_v0_array = child_izone_array[locs_yield, self.__v0_index]
        pred_s_array = child_izone_array[locs_yield, self.__pred_s_index]

        pred_v0_array[pred_v0_array < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / pred_v0_array

        valids_locs[locs_yield] = plan_t_array >= (pred_cvm_t + self.yield_safe_time_gap)
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs

  def __edge_is_valid_pred(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        pred_t_array = child_izone_array[locs_notdeter, self.__t_index]

        preempt_locs = plan_t_array <= (pred_t_array - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() + yield_locs * StiNode.relation_yield()

        self._update_relations(child_nodes, range_iinfo_dict, get_relations)
        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, yield_locs) * 1.0 # set valids

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_t_array = child_izone_array[locs_preempt, self.__t_index]

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition
        valids_locs[locs_preempt] = np.logical_and(
          plan_t_array <= (pred_t_array - self.safe_time_gap), 
          not2stop_array)

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_t_array = child_izone_array[locs_yield, self.__t_index]

        valids_locs[locs_yield] =\
          (plan_t_array >= (pred_t_array + self.yield_safe_time_gap))
  
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs

  def __edge_is_valid_predcvm(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        plan_v_array = check_stva_array[locs_notdeter, 2]
        pred_v0_array = child_izone_array[locs_notdeter, self.__v0_index]
        pred_s_array = child_izone_array[locs_notdeter, self.__pred_s_index]

        protect_v0 = pred_v0_array.copy()
        protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / protect_v0
        
        pred_t_array = child_izone_array[locs_notdeter, self.__t_index]
        replace_locs = pred_v0_array < self.__cond_low_speed
        pred_t_array[replace_locs] = pred_cvm_t[replace_locs]

        preempt_locs = plan_t_array <= (pred_t_array - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() +\
          yield_locs * StiNode.relation_yield()

        self._update_relations(child_nodes, range_iinfo_dict, get_relations)
        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, yield_locs) * 1.0 # set valids

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_v0_array = child_izone_array[locs_preempt, self.__v0_index]
        pred_acc0_array = child_izone_array[locs_preempt, self.__acc_index]
        pred_s_array = child_izone_array[locs_preempt, self.__pred_s_index]

        protect_v0 = pred_v0_array.copy()
        protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / protect_v0

        pred_t_array = child_izone_array[locs_preempt, self.__t_index]
        replace_locs = pred_v0_array < self.__cond_low_speed
        pred_t_array[replace_locs] = pred_cvm_t[replace_locs]

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition
        valids_locs[locs_preempt] =\
          (np.logical_and(plan_t_array <= (pred_t_array - self.safe_time_gap), not2stop_array)) * 1.0
        # print("locs_preempt", valids_locs[locs_preempt])

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_v0_array = child_izone_array[locs_yield, self.__v0_index]
        pred_s_array = child_izone_array[locs_yield, self.__pred_s_index]

        # yield conditions:
        protect_v0 = pred_v0_array.copy()
        protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / protect_v0

        pred_t_array = child_izone_array[locs_yield, self.__t_index]
        replace_locs = pred_v0_array < self.__cond_low_speed
        pred_t_array[replace_locs] = pred_cvm_t[replace_locs]

        valids_locs[locs_yield] =\
          (plan_t_array >= (pred_t_array + self.yield_safe_time_gap))
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs

  def __edge_is_valid_cam(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        pred_v0_array = child_izone_array[locs_notdeter, self.__v0_index]
        pred_acc0_array = child_izone_array[locs_notdeter, self.__acc_index]
        pred_s_array = child_izone_array[locs_notdeter, self.__pred_s_index]

        protect_v0 = pred_v0_array.copy()
        protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / protect_v0

        dd = np.square(pred_v0_array) + 2.0*pred_acc0_array*pred_s_array
        acc_nonzero_locs = np.abs(pred_acc0_array) > 1e-1
        acc_zero_locs = np.logical_not(acc_nonzero_locs)

        legal_locs = np.logical_and(acc_nonzero_locs, dd >= 0.0)

        pred_cam_t_array = np.ones_like(dd) * 1e+3 # default with inf time (situ: braking to stop)
        pred_cam_t_array[legal_locs] =\
          (-pred_v0_array[legal_locs] + np.sqrt(dd[legal_locs])) / pred_acc0_array[legal_locs] # acc not zero & legal situation
        pred_cam_t_array[acc_zero_locs] = pred_cvm_t[acc_zero_locs] # acc = 0 using cvm

        preempt_locs = plan_t_array <= (pred_cam_t_array - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_cam_t_array + self.yield_safe_time_gap)

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() +\
          yield_locs * StiNode.relation_yield()

        self._update_relations(child_nodes, range_iinfo_dict, get_relations)
        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, yield_locs) * 1.0 # set valids

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_v0_array = child_izone_array[locs_preempt, self.__v0_index]
        pred_acc0_array = child_izone_array[locs_preempt, self.__acc_index]
        pred_s_array = child_izone_array[locs_preempt, self.__pred_s_index]

        protect_v0 = pred_v0_array.copy()
        protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / protect_v0

        dd = np.square(pred_v0_array) + 2.0*pred_acc0_array*pred_s_array
        acc_nonzero_locs = np.abs(pred_acc0_array) > 1e-1
        acc_zero_locs = np.logical_not(acc_nonzero_locs)

        legal_locs = np.logical_and(acc_nonzero_locs, dd >= 0.0)

        pred_cam_t_array = np.ones_like(dd) * 1e+3 # default with inf time (situ: braking to stop)
        pred_cam_t_array[legal_locs] =\
          (-pred_v0_array[legal_locs] + np.sqrt(dd[legal_locs])) / pred_acc0_array[legal_locs] # acc not zero & legal situation
        pred_cam_t_array[acc_zero_locs] = pred_cvm_t[acc_zero_locs] # acc = 0 using cvm

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition
        valids_locs[locs_preempt] = np.logical_and(
          plan_t_array <= (pred_cam_t_array - self.safe_time_gap), not2stop_array)

        # print("locs_preempt", valids_locs[locs_preempt])

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_v0_array = child_izone_array[locs_yield, self.__v0_index]
        pred_s_array = child_izone_array[locs_yield, self.__pred_s_index]
        pred_acc0_array = child_izone_array[locs_yield, self.__acc_index]

        # yield conditions:
        protect_v0 = pred_v0_array.copy()
        protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        pred_cvm_t = pred_s_array / protect_v0

        dd = np.square(pred_v0_array) + 2.0*pred_acc0_array*pred_s_array
        acc_nonzero_locs = np.abs(pred_acc0_array) > 1e-1
        acc_zero_locs = np.logical_not(acc_nonzero_locs)

        legal_locs = np.logical_and(acc_nonzero_locs, dd >= 0.0)

        pred_cam_t_array = np.ones_like(dd) * 1e+3 # default with inf time (situ: braking to stop)
        pred_cam_t_array[legal_locs] =\
          (-pred_v0_array[legal_locs] + np.sqrt(dd[legal_locs])) / pred_acc0_array[legal_locs] # acc not zero & legal situation
        pred_cam_t_array[acc_zero_locs] = pred_cvm_t[acc_zero_locs] # acc = 0 using cvm

        valids_locs[locs_yield] =\
          (plan_t_array >= (pred_cam_t_array + self.yield_safe_time_gap))
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid

    return valid_costs

  def __edge_is_valid_irule(self, range_iinfo_dict: Dict, parent_node: np.ndarray, child_nodes: np.ndarray) -> np.ndarray:
    '''
    check whether the edge is valid to add to open list
    :param range_iinfo_dict: range interaction information dict from _extract_range_iinfo()
    :param parent_node: the edge's parent node
    :param child_nodes: the parent node's multiple child nodes
    :return: [[is_valid_flag, reaction cost], ...]
    '''
    valid_costs = np.zeros((child_nodes.shape[0], 2))
    valid_costs[:, 0] = 1.0 # default with all edge valids

    plan_v0 = self.start_sva[1]
    if range_iinfo_dict['has_interaction'] and (child_nodes.shape[0] > 0):
      # prepare data
      check_izone_array = range_iinfo_dict['izone_data']
      # area_protect_s_extended = range_iinfo_dict['area_protect_s_extended']
      # print(check_izone_array[:, [0, 1, 3, 4, 6, 7]]) agent/agent_traj s/t values overlapped with path point of AV
      if check_izone_array.shape[0] == 0:
        return valid_costs # no interactions, return all valids

      # relation calculations
      # check_izone_array with shape = (zone_and_interp_num, zone_info_dim)
      # check_stva_array with shape = (child_num, zone_and_interp_num, 4)
      # record_tsv_array with shape = (child_num, zone_and_interp_num, record_num x 3)
      # child_nodes with shape (child_num, node_state_num)
      check_stva_array = self._interpolate_nodes_given_s(
        parent_node, child_nodes, inquiry_s=check_izone_array[:, self.__s_index])
      # record_tsv_array = self.__get_tsv_record(
      #   child_nodes=child_nodes, inquiry_s=check_izone_array[:, self.__s_index])

      # child_izone_array with shape = (child_num, zone_and_interp_num, zone_info_dim)
      child_izone_array =\
        np.repeat(np.array([check_izone_array.tolist()]), child_nodes.shape[0], axis=0)

      # child_relations with shape = (child_num, zone_and_interp_num)
      child_relations = child_nodes[:, range_iinfo_dict['relation_indexes']]
      locs_notdeter = np.abs(child_relations - StiNode.relation_not_determined()) < 1e-3
      # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
      # # example = (4, 13) (4, 13, 11) 13
      # #         = (13, 11)
      # print(child_relations.shape, child_izone_array.shape, len(range_iinfo_dict['relation_indexes']))
      # print(check_izone_array.shape)

      # init relations with original values
      get_relations = child_relations.copy()
      valids_locs = np.zeros_like(child_relations) # default all are invalids
      costs_locs = np.zeros_like(child_relations)
      notdter_set_inlu_locs = valids_locs < -1e+3 # init with all false

      ################################################################
      # not determined locations
      if np.sum(locs_notdeter) > 0:
        plan_t_array = check_stva_array[locs_notdeter, 1]
        plan_v_array = check_stva_array[locs_notdeter, 2]
        pred_v0_array = child_izone_array[locs_notdeter, self.__v0_index]
        pred_s_array = child_izone_array[locs_notdeter, self.__pred_s_index]

        # protect_v0 = pred_v0_array.copy()
        # protect_v0[protect_v0 < self._cvm_protect_min_v] = self._cvm_protect_min_v
        # pred_cvm_t = pred_s_array / protect_v0
        
        pred_t_array = child_izone_array[locs_notdeter, self.__t_index]
        # predcvm_t_array = pred_t_array.copy()
        # replace_locs = pred_v0_array < self.__cond_low_speed
        # predcvm_t_array[replace_locs] = pred_cvm_t[replace_locs]

        ddd = np.square(pred_v0_array) + 2.0*self.__reaction_cond_acc*pred_s_array
        ddd_valids = ddd >= 0.0 # give dcc, vt >= 0.0
        iacc_arrive_t = np.ones_like(pred_v0_array) * 1e+3 # ddd_invalids are inf
        iacc_arrive_t[ddd_valids] = (-pred_v0_array[ddd_valids] + np.sqrt(ddd[ddd_valids])) / self.__reaction_cond_acc

        influ_locs = np.logical_and(
          (plan_t_array + self.__ireact_delay_s / (1e-3 + plan_v_array)) <= (pred_t_array - self.__ireact_gap_cond_t),
          iacc_arrive_t >= (plan_t_array + self.yield_safe_time_gap)
        )

        preempt_locs = plan_t_array <= (pred_t_array - self.safe_time_gap)
        yield_locs = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)
        preempt_locs[influ_locs] = False
        yield_locs[influ_locs] = False

        valid_array1 = np.logical_or(yield_locs, influ_locs)
        notdter_set_inlu_locs[locs_notdeter] = influ_locs

        valids_locs[locs_notdeter] =\
          np.logical_or(preempt_locs, valid_array1) * 1.0 # set valids

        # update relationship
        #   preempt_locs with relation_preempt
        #   yield_locs with relation_yield
        #   others with original values
        get_relations[locs_notdeter] =\
          preempt_locs * StiNode.relation_preempt() +\
          influ_locs * StiNode.relation_influ() +\
          yield_locs * StiNode.relation_yield()
        self._update_relations(child_nodes, range_iinfo_dict, get_relations)

        # if self.enable_debug_rviz:
        #   debug_a_array = child_izone_array[locs_notdeter, :][:, self.__agent_index]
        #   debug_t_array = child_izone_array[locs_notdeter, :][:, self.__agent_traj_index]
        #   # debug_av_s_array = child_izone_array[locs_notdeter, :][:, self.__s_index]
        #   # debug_av_t_array = check_stva_array[locs_notdeter, 1]
        #   # debug_av_v_array = check_stva_array[locs_notdeter, 2]
        #   # debug_agent_v0_array = child_izone_array[locs_notdeter, :][:, self.__v0_index]
        #   # debug_agent_acc_array = child_izone_array[locs_notdeter, :][:, self.__acc_index]
        #   # debug_agent_s_array = child_izone_array[locs_notdeter, :][:, self.__pred_s_index]
        #   # debug_agent_t_array = child_izone_array[locs_notdeter, :][:, self.__t_index]
        #   debug_idx :int= 304625
        #   if (debug_idx in np.unique(debug_a_array)):
        #     print("AAA=", influ_locs)
        #     show_locs1 = debug_a_array[influ_locs] == debug_idx
        #     show_locs2 = debug_a_array[preempt_locs] == debug_idx
        #     show_locs3 = debug_a_array[yield_locs] == debug_idx
        #     print("parent_node s=", parent_node[self._node_key_s])
        #     print("influ agents", np.unique(debug_a_array[influ_locs][show_locs1]),
        #           "influ trajs", np.unique(debug_t_array[influ_locs][show_locs1]))
        #     print("preempt agents", np.unique(debug_a_array[preempt_locs][show_locs2]),
        #           "preempt trajs", np.unique(debug_t_array[preempt_locs][show_locs2]))
        #     print("yield agents", np.unique(debug_a_array[yield_locs][show_locs3]),
        #           "yield trajs", np.unique(debug_t_array[yield_locs][show_locs3]))
        #     # print("av-s=", debug_av_s_array[debug_locs][show_locs1])
        #     # print("av-t=", debug_av_t_array[debug_locs][show_locs1])
        #     # print("av-v", debug_av_v_array[debug_locs][show_locs1])
        #     # print("i-v0=", debug_agent_v0_array[debug_locs][show_locs1])
        #     # print("i-acc0=", debug_agent_acc_array[debug_locs][show_locs1])
        #     # print("i-s=", debug_agent_s_array[debug_locs][show_locs1])
        #     # print("i-t=", debug_agent_t_array[debug_locs][show_locs1])
        #     print(" ")

      ################################################################
      # influ locations
      locs_influ = np.abs(get_relations - StiNode.relation_influ()) < 1e-3

      self.__process_influ_constraints(
        locs_influ, check_stva_array, child_izone_array, valids_locs)

      ################################################################
      # preempt locations
      locs_preempt = np.abs(get_relations - StiNode.relation_preempt()) < 1e-3

      if np.sum(locs_preempt) > 0:
        plan_t_array = check_stva_array[locs_preempt, 1]
        plan_v_array = check_stva_array[locs_preempt, 2]
        pred_t_array = child_izone_array[locs_preempt, self.__t_index]

        # preempt means that agent's t exist > plan_t, if av plan to stop, it would be invalid
        not2stop_array = plan_v_array > self.stop_v_condition

        valids_locs[locs_preempt] =\
          np.logical_and(plan_t_array <= (pred_t_array - self.safe_time_gap), 
          not2stop_array)
        # print("locs_preempt", valids_locs[locs_preempt])

      ################################################################
      # yield locations
      locs_yield = np.abs(get_relations - StiNode.relation_yield()) < 1e-3

      if np.sum(locs_yield) > 0:
        plan_t_array = check_stva_array[locs_yield, 1]
        pred_t_array = child_izone_array[locs_yield, self.__t_index]

        valids_locs[locs_yield] = plan_t_array >= (pred_t_array + self.yield_safe_time_gap)
        # set valids when is2stop and collisions at any future time stamp
        # because: any stop is valid as it already arrives after other agents

      ################################################################
      # set invalids when some edges have conflicted relations (e.g., have preempt and yield at meantime)
      self.__process_relation_conflicts(range_iinfo_dict, get_relations, valids_locs)

      # set valid & costs
      illegal_child_locs = np.sum(valids_locs, axis=1) < valids_locs.shape[1]
      valid_costs[illegal_child_locs, 0] = 0.0 # set invalid
      valid_costs[:, 1] = np.sum(costs_locs, axis=1) # fill costs

    return valid_costs
