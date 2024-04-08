import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math

import matplotlib
import matplotlib.pyplot as plt
import paper_plot.utils as plot_utils
import paper_plot.functions as plot_func

##############################################################
# collecting data

v_reso = 0.25
dist_reso = 1.0
ref_dcc = -4.0
max_t = 10.0

v0_array = np.linspace(0.0, 20.0, round(20.0/v_reso))
s_array = np.linspace(0.0, 50.0, round(50.0/dist_reso))

v0_s_t_iacc = []
v0_s_dt_iacc_cvm = []
for v0 in v0_array:
  for s in s_array:
    ddd = (v0**2) + 2.0*ref_dcc*s
    t_iacc = max_t
    if ddd >= 0.0:
      t_iacc = (-v0 + math.sqrt(ddd)) / ref_dcc
    t_cvm = s / max(1e-1, v0)

    v0_s_t_iacc.append([v0, s, t_iacc])
    v0_s_dt_iacc_cvm.append([v0, s, max(t_iacc - t_cvm, 0.0)])

v0_s_t_iacc = np.array(v0_s_t_iacc)
v0_s_dt_iacc_cvm = np.array(v0_s_dt_iacc_cvm)

##############################################################
# visualization

fig = plt.figure()
plot_utils.fig_reset()

# ax1
fig_ax1 = fig.add_subplot(121)
plot_utils.subfig_reset()
plot_utils.axis_set_xlabel(fig_ax1, 'v0 (m/s)')
plot_utils.axis_set_ylabel(fig_ax1, 's (m)')

rviz_cmap=plt.cm.get_cmap('Accent')
fig_ax1.scatter(v0_s_t_iacc[:, 0], v0_s_t_iacc[:, 1], 
                c=v0_s_t_iacc[:, 2], 
                vmin=0.0, vmax=max_t, 
                cmap=rviz_cmap, 
                marker='s', s=3)

cmp_norm = matplotlib.colors.BoundaryNorm(
  np.linspace(0.0, max_t, rviz_cmap.N), rviz_cmap.N)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=rviz_cmap),
             ax=fig_ax1, shrink=1.0, orientation='vertical')

# ax2
fig_ax2 = fig.add_subplot(122)
plot_utils.subfig_reset()
plot_utils.axis_set_xlabel(fig_ax2, 'v0 (m/s)')
plot_utils.axis_set_ylabel(fig_ax2, 's (m)')

rviz_cmap=plt.cm.get_cmap('Accent')
fig_ax2.scatter(v0_s_dt_iacc_cvm[:, 0], v0_s_dt_iacc_cvm[:, 1], 
                c=v0_s_dt_iacc_cvm[:, 2], 
                vmin=0.0, vmax=max_t, 
                cmap=rviz_cmap, 
                marker='s', s=3)

cmp_norm = matplotlib.colors.BoundaryNorm(
  np.linspace(0.0, max_t, rviz_cmap.N), rviz_cmap.N)
fig.colorbar(matplotlib.cm.ScalarMappable(norm=cmp_norm, cmap=rviz_cmap),
             ax=fig_ax2, shrink=1.0, orientation='vertical')


plt.show()