import files.module_path

import os
import torch
import numpy as np
import torch
import torch.nn as nnfrom typing import Tuple, Dict, Union
from models.ipm.dt_model import IPMDtModel

# import torch.optim as optim
# import networks.torchbnn as bnn
# import torch.nn.functional as F

TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREE

class IPMPriorityModel():
  '''
  3-layer MLP of for interaction priority classification
  '''
  def __init__(self, device :str= 'cpu', 
                     input_dim :int= 2, 
                     midlayer_dim :int= 5, 
                     output_dim :int= 2) -> None:
    self.device = device
    self.dt_model = IPMDtModel()

    # init file directory
    file_dir = os.path.dirname(files.module_path.__file__)
    self.model_file = os.path.join(file_dir, "ipmodel.ckpt")
    self.script_model_file = os.path.join(file_dir, "ipmodel.pt")

    # network
    self.input_dim = input_dim
    self.mid_dim = midlayer_dim
    self.output_dim = output_dim

    self.model = nn.Sequential(
      nn.Linear(input_dim, midlayer_dim),
      nn.Linear(midlayer_dim, midlayer_dim),
      nn.Tanh(),
      nn.Linear(midlayer_dim, output_dim),
      nn.Softmax(dim=1)
    ).to(device)

    self.model_is_loaded = False

  def load_model_from_file(self) -> bool:
    '''
    Return true if model is successfully loaded from the model_file
    '''
    load_successfully = False
    try:
      self.model.load_state_dict(torch.load(self.model_file))
      load_successfully = True
      self.model_is_loaded = True
    except:
      pass
    return load_successfully

  def forward_mlp(self, input_vec :torch.Tensor) -> torch.Tensor:
    '''
    Run MLP given input vector with shape (N, input_dim)
    '''
    assert self.model_is_loaded == True, "Error, model is not loaded"
    input_vec = input_vec.to(self.device)
    pred_vec = self.model(input_vec)
    return pred_vec
  
  def __get_overtaking_giving_way_abilities(self, 
      av_v0 :float, av_st_array :np.ndarray, 
      other_agents_v0si_array :np.ndarray,
      acc_bounds: Tuple[float, float],
      agents_maximal_v: float) -> Dict:
    '''
    Return overtaking ability and giving way ability: array of [M^-, M^+], and other intermediate variables
    :param av_v0, initial speed of the AV (initial state)
    :param av_st_array, AV's (s, t) values to reach the 's' location along the AV's path (planned trajectory)
    :param other_agents_v0si_array, other agents' (v0, s, interaction angle) values, 
           s indicates the distance to reach interaction points (along predictions)
    :param acc_bounds, defined acc limits for all agents, e.g., (-3, 3) m/s^2
    :param agents_maximal_v, estimated maximal speed for all agents
    :return: dict of results
    '''
    ## AV relevant calculation
    # arrival time of the AV along planned trajectory
    av_min_arrival_ts = av_st_array[:, 1]
    av_move_s = av_st_array[:, 0]

    # maximum arrival time of the AV along planned trajectory
    # vt**2 = v0**2 + 2.0 * a * s
    dd = av_v0**2 + 2.0 * acc_bounds[0] * av_move_s
    dd[dd < 0.0] = 0.0
    av_min_vts = np.sqrt(dd)
    av_max_arrival_ts = 2.0 * av_move_s / (1e-2 + av_min_vts + av_v0)

    ## agents relevant calculation
    agents_v0 = other_agents_v0si_array[:, 0]
    agents_s = other_agents_v0si_array[:, 1]
    iangles = other_agents_v0si_array[:, 2]

    # minimum arrival time of agents along their predictions to reach the AV's path
    # vt**2 = v0**2 + 2.0 * a * s
    dd = agents_v0**2 + 2.0 * acc_bounds[1] * agents_s
    agents_max_vts = np.sqrt(dd)
    agents_max_vts[agents_max_vts > agents_maximal_v] = agents_maximal_v
    agents_min_arrival_ts = (agents_max_vts - agents_v0) / acc_bounds[1]

    # maximal arrival time of agents along their predictions to reach the AV's path
    dd = agents_v0**2 + 2.0 * acc_bounds[0] * agents_s
    dd[dd < 0.0] = 0.0
    agents_min_vts = np.sqrt(dd)
    agents_max_arrival_ts = 2.0 * agents_s / (1e-3 + agents_min_vts + agents_v0)

    ## get M^- and M^+
    overtake_dt_bounds = self.dt_model.overtake_dt_upper_bound(iangles)
    giveway_dt_bounds = self.dt_model.giveway_dt_lower_bound(iangles)

    overtake_M = av_min_arrival_ts - agents_min_arrival_ts + np.fabs(overtake_dt_bounds)
    giveway_M = av_max_arrival_ts - agents_max_arrival_ts - np.fabs(giveway_dt_bounds)
    overtake_giveway_Ms = np.vstack((overtake_M, giveway_M)).transpose() # (N, 2), N is the number of interaction points

    return {
      'overtake_giveway_Ms': overtake_giveway_Ms,
      'overtake_dt_bounds': overtake_dt_bounds,
      'giveway_dt_bounds': giveway_dt_bounds,
    }

  def ipmnet_get_priorities_and_time_gaps(self, 
      av_v0 :float, av_st_array :np.ndarray, 
      other_agents_v0si_array :np.ndarray,
      acc_bounds: Tuple[float, float],
      agents_maximal_v: float) -> Dict:  
    '''
    Return interaction priorities and time gap bounds accroding to trained ipm network model
    :param av_v0, initial speed of the AV (initial state)
    :param av_st_array, AV's (s, t) values to reach the 's' location along the AV's path (planned trajectory)
    :param other_agents_v0si_array, other agents' (v0, s, interaction angle) values, 
           s indicates the distance to reach interaction points (along predictions)
    :param acc_bounds, defined acc limits for all agents, e.g., (-3, 3) m/s^2
    :param agents_maximal_v, estimated maximal speed for all agents
    :return: dict of results
    '''
    dict_results = self.__get_overtaking_giving_way_abilities(
      av_v0, av_st_array, other_agents_v0si_array, 
      acc_bounds, agents_maximal_v)

    overtake_M = dict_results['overtake_giveway_Ms'][:, 0]
    giveway_M = dict_results['overtake_giveway_Ms'][:, 1]

    tm_conds = 0.5 * av_st_array[:, 1] + 1.5
    tm_conds[tm_conds > 6.0] = 6.0
    force_set_low_pri_locs = other_agents_v0si_array[:, 1] < 1e-1
    np.logical_or(
      overtake_M >= tm_conds,
      other_agents_v0si_array[:, 1] < 1e-1
    )

    priorities = self.forward_mlp(
      torch.from_numpy(dict_results['overtake_giveway_Ms']).float()).detach().numpy()
    priorities[force_set_low_pri_locs, :] = np.array([0., 1.0])

    return {
      'priorities': priorities, 
      'overtake_dt_bounds': dict_results['overtake_dt_bounds'],
      'giveway_dt_bounds': dict_results['giveway_dt_bounds'],
    }

  def ipmpred_get_high_priorities_locations(self, 
      av_t_array :Union[np.ndarray, float], agents_t_array: Union[np.ndarray, float]) -> Union[np.ndarray, bool]:
    '''
    Return locations where av's priority is higher than agents'
    '''
    return av_t_array < agents_t_array

  def ipmpred_get_low_priorities_locations(self, 
      av_t_array :Union[np.ndarray, float], agents_t_array: Union[np.ndarray, float]) -> Union[np.ndarray, bool]:
    '''
    Return locations where av's priority is lower than agents'
    '''
    return av_t_array >= agents_t_array

  def ipmpred_get_dt_bounds(self, iangles: np.ndarray) -> np.ndarray:
    '''
    Return time gap bounds given interaction angles
    :param iangles: interaction angles in radian, with shape = (N, 1)
    '''
    return {
      'overtake_dt_bounds': self.dt_model.overtake_dt_upper_bound(iangles),
      'giveway_dt_bounds': self.dt_model.giveway_dt_lower_bound(iangles),
    }

  def ipmpred_get_priorities_and_time_gaps(self, 
        av_st_array :np.ndarray, other_agents_v0stiv_array :np.ndarray,
        enable_return_dt_bounds: bool=False) -> Dict:
    '''
    :param av_st_array, AV's (s, t) values to reach the 's' location along the AV's path (planned trajectory)
    :param other_agents_v0stiv_array, other agents' (v0, s, t, interaction_angle, v) values, 
           s indicates the distance to reach interaction points (along predictions), t and v are
           the corresponding time and velocity values at the s location
    :param enable_return_dt_bounds: enable to return the dt bounds or not
    '''
    assert av_st_array.shape[0] == other_agents_v0stiv_array.shape[0], 'Fatal error, row num unequal.'
    
    # update priority: set plan_t < pred_t with high priorities 
    priorities = np.zeros_like(av_st_array)
    priorities[av_st_array[:, 1] < other_agents_v0stiv_array[:, 2]] = np.array([1.0, 0.0])

    if enable_return_dt_bounds:
      iangles = other_agents_v0stiv_array[:, 3]
      overtake_dt_bounds = self.dt_model.overtake_dt_upper_bound(iangles)
      giveway_dt_bounds = self.dt_model.giveway_dt_lower_bound(iangles)

      return {
        'priorities': priorities, 
        'overtake_dt_bounds': overtake_dt_bounds,
        'giveway_dt_bounds': giveway_dt_bounds,
      }
    else:
      return {
        'priorities': priorities, 
        'overtake_dt_bounds': None,
        'giveway_dt_bounds': None,
      } 
