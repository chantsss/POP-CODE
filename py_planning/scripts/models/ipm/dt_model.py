import os
import math
import numpy as np

M_PI = 3.14159265358979323846

class IPMDtModel:
  def __init__(self) -> None:
    self.poly_radian2maxdt = [-0.17106961, 0.09499304, -1.2734661 ]
    self.poly_dv2maxdt = [-3.73132913e-05, 7.21087592e-06, -1.23657133e-02, 3.38347447e-02, -1.16506280e+00]
    self.poly_p_radian2maxdt = [0.18237564, -0.14051625, 1.40760894]
    self.poly_p_dv2maxdt = [6.26375360e-05, 3.43049200e-04, 1.32386831e-02, 3.97324890e-02, 1.27294369e+00]

    self.modified_poly_iradian2dt = [0.01333489, -0.13840551, 0.59530217, -0.33279808, 0.55909382]

  def overtake_dt_upper_bound(self, dradian :np.ndarray, dv :np.ndarray= None):
    '''
    get overtake interaction delta t, it is a upper bound value 
    since the interaction t is always < 0 (ego go first, and other traffic agent)

    :param dradian: the interaction angle (radian) in interaction zone 
    :param dv: the differential speed (m/s) in interaction zone
    :return: the upper bound of interaction delta t
    '''
    if dv == None:
      dv = np.zeros_like(dradian)

    abs_dradian = np.fabs(dradian)
    upper_bound1 = np.polyval(self.poly_radian2maxdt, abs_dradian)
    upper_bound2 = np.polyval(self.poly_dv2maxdt, dv)

    # get_common_bound = min(upper_bound1, upper_bound2)
    get_common_bound = upper_bound1
    locs = get_common_bound > upper_bound2
    get_common_bound[locs] = upper_bound2[locs]
  
    get_common_bound = upper_bound1 + 0.5
    get_common_bound[get_common_bound >= 0.0] = 0.0

    return get_common_bound

  def giveway_dt_lower_bound(self, dradian :np.ndarray, dv :np.ndarray= None):
    '''
    get giveway interaction delta t, it is a lower bound value 
    since the interaction t is always > 0 (ego go after other traffic agent)

    :param dradian: the interaction angle (radian) in interaction zone 
    :param dv: the differential speed (m/s) in interaction zone
    :return: the lower bound of interaction delta t
    '''
    if dv == None:
      dv = np.zeros_like(dradian)

    abs_dradian = np.fabs(dradian)
    lowerbound1 = np.polyval(self.poly_p_radian2maxdt, abs_dradian)
    lowerbound2 = np.polyval(self.poly_p_dv2maxdt, dv)

    # get_common_bound = max(lowerbound1, lowerbound2)
    get_common_bound = lowerbound1
    locs = get_common_bound < lowerbound2
    get_common_bound[locs] = lowerbound2[locs]

    return get_common_bound

  def giveway_lower_bound_cmr(self, dradian: np.ndarray) -> np.ndarray:
    '''
    get giveway interaction delta t, it is a lower bound value 
    since the interaction t is always > 0 (ego go after other traffic agent)

    :param dradian: the interaction angle (radian) in interaction zone 
    '''
    abs_dradian = np.fabs(dradian)
    lowerbound = np.polyval(self.modified_poly_iradian2dt, abs_dradian)

    return lowerbound
