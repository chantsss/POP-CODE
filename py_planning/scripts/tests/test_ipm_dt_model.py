import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
TO_DEGREE = 57.29577951471995
TO_RADIAN = 1.0 / TO_DEGREEfrom models.ipm.dt_model import IPMDtModel

import matplotlib
import matplotlib.pyplot as plt
import paper_plot.utils as plot_utils
import paper_plot.functions as plot_func

angles = np.arange(0.0, 180.1, 1.0)
radians = angles * TO_RADIAN

# ipm model from interaction dataset
model = IPMDtModel()
ttt1 = model.giveway_dt_lower_bound(radians)

# modified version
poly_radians = np.array(
    [0.0, 25.0, 50.0, 100.0, 150.0, 180.0]) * TO_RADIAN
poly_ts = np.array(
    [0.55, 0.55, 0.6, 1.2, 1.9, 2.4])
poly_mat = np.polyfit(poly_radians, poly_ts, 4)

ttt2 = np.polyval(poly_mat, radians)

fig = plt.figure()
plot_utils.fig_reset()

# ax1
fig_ax1 = fig.add_subplot(111)
plot_utils.subfig_reset()
plot_utils.axis_set_xlabel(fig_ax1, 'iangle (degree)')
plot_utils.axis_set_ylabel(fig_ax1, 'dt (s)')

fig_ax1.plot(angles, ttt1, 'b', label='dataset')
plt.legend()
fig_ax1.plot(angles, ttt2, 'g', label='modified')
plt.legend()

print("poly_mat=", poly_mat)
plt.show()
